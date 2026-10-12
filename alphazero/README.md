# AlphaZero Studio

A real, local AlphaZero-style implementation: a residual policy/value network learns from MCTS-generated self-play, with no human games, handcrafted evaluator or pretrained weights. The default adapter plays **exactly the territory Pentomino rules in the parent web application**. A second adapter for tic-tac-toe demonstrates that the search and training code do not depend on Pentomino.

This is a small research implementation, not DeepMind's original code or a claim of comparable playing strength.

## Run

Python 3.11+ is required; this workspace was tested with Python 3.14 and the exact dependencies in `requirements.lock.txt`.

```sh
cd alphazero
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m az.server
```

Open **http://127.0.0.1:8787**. Select a game, iteration count and search budget, then start. The dashboard shows real self-play positions, training losses, replay-buffer size, evaluation results and the latest complete game replay. It supports pause/checkpoint and resume. A completed run can be extended; a paused run continues its remaining iteration budget. Closing the browser does not stop training. Stop the studio process to stop its trainer gracefully. The studio binds only to loopback and does not submit cloud jobs.

CLI alternative:

```sh
.venv/bin/python -m az.train --config configs/local.json --run-dir runs/pentomino-local
.venv/bin/python -m az.train --config configs/local.json --run-dir runs/pentomino-local --resume
.venv/bin/python -m az.train --config configs/smoke.json --run-dir runs/tictactoe-smoke
.venv/bin/python -m pytest -q
```

Use Ctrl+C to pause and save. Do not launch CLI training into a directory already in use by the studio. Resume restores network, optimizer, replay memory, NumPy RNG and CPU Torch RNG. Increase `iterations` in the original config to extend a completed run; incompatible training settings are rejected. On interruption, completed games are retained; an unfinished self-play game is discarded and the partial iteration's game collection restarts. CPU reproducibility is tested; results across devices/library versions need not be bitwise identical.

## Common interface

`az/game.py` defines the `Game[State]` protocol:

- Immutable state, initial state and player to move.
- Stable integer action IDs and the exact legal-action list.
- State transition and terminal result (+1/0/−1 from the player-to-move perspective).
- A Markov-complete `float32` tensor `(channels, height, width)` and action-space size.
- Serialization, readable action decoding and a finite game horizon.

Implement an adapter, register it in `az/games/__init__.py`, and use its name in a training config. No changes to MCTS, the network, replay or trainer are needed. The generic network accepts different rectangular board dimensions and action counts. Models are trained separately for each game; this is not one pretrained universal model.

**Scope:** deterministic, alternating, two-player, perfect-information, zero-sum games with discrete actions. Chance nodes, hidden information, simultaneous moves, multiplayer and general-sum rewards need different search/state semantics; the interface does not pretend to solve those unchanged.

## Structure

```text
az/game.py             adapter contract and value conventions
az/games/pentomino.py  bitboard rules, action codec and browser-state bridge
az/games/tictactoe.py  second independent game adapter
az/network.py         residual trunk, policy head, tanh value head, evaluator
az/mcts.py            PUCT expansion, root noise, visit targets, value backup
az/selfplay.py        self-play trajectories and independent random-opponent evaluation
az/replay.py          bounded position replay and sparse target storage
az/train.py           SGD learner, schedules, metrics, atomic resumable checkpoints
az/server.py          local job controls and read-only monitoring endpoints
web/                  standalone responsive studio UI
configs/              local, smoke and scale-reference configurations
tests/                search, game, cross-language and checkpoint integration tests
runs/                 ignored generated results, one folder per experiment
```

## Algorithm and fidelity

Reference: [Silver et al., 2018, *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf), especially the algorithm description and Methods.

Implemented:

- Random initialization; training exclusively from self-play terminal outcomes.
- Convolutional residual blocks with batch normalization and separate policy/value heads.
- Legal-action masking and renormalization during search.
- PUCT using `Q + C(N) P sqrt(N) / (1 + n)`, with `C(N) = log((N + 19652 + 1) / 19652) + 1.25`.
- Dirichlet noise mixed into root priors only during training (25% noise); local Pentomino alpha 0.03 is an adaptation, not a value tuned or reported for this game by DeepMind.
- Value backup with explicit player perspective. Leaves use the network or exact terminal result; no random rollouts or handcrafted strategic evaluator.
- Search visit distribution as the policy target, and final win/draw/loss as the value target. Training moves are sampled proportionally to visits; evaluation uses maximum visits without noise.
- Cross-entropy policy loss + mean-squared value loss + explicit L2 penalty; SGD with momentum 0.9 and a step-based learning-rate schedule. Policy loss uses the full action logits, not just legal logits.
- Latest network used for subsequent self-play; no AlphaGo Zero 55% promotion gate and no board-symmetry augmentation.

Adaptations and limits:

- Default: 2 residual blocks, 32 channels, 64 simulations, batch 32, learning rate 0.02, replay capacity 5,000 positions and two CPU threads. These are practical development settings, not the paper's training budget.
- Sequential game generation and learning, unbatched leaf inference, no distributed actors, no virtual loss, no tree reuse, and no resignation. These favor clarity over throughput. No fixed-time equivalence to DeepMind's TPU experiment is implied.
- State features contain occupancy, both inventories, current player, passes and move count. History planes are unnecessary for this game's Markov rules. The fixed policy contains 12 pieces × 8 transform slots × 144 anchors + pass = 13,825 outputs. Duplicate geometric transforms are canonically mapped and masked; off-board and illegal slots remain masked at search time.
- `configs/paper-scale.json` is an **unexecuted reference**, with 800 simulations, 19 residual blocks/256 channels and batch 4096. It is not a reproduction of the original distributed system; its replay size is in positions, not the paper's games, and it requires substantial GPU memory/resources. Do not run it expecting laptop-scale costs.
- The web app's room server does not run PyTorch. Training stays in this separate local service; its adapter, state bridge and parity tests are the integration with Pentomino.

## Outputs and interpretation

Each run writes `config.json`, live `status.json`, `metrics.jsonl`, full self-play `games.jsonl`, `latest-game.json`, and atomic `checkpoint.pt`. Checkpoints include model, SGD state, replay, RNG and progress; generated files are ignored by Git and omitted from web deployment.

Evaluation alternates colors against a uniform legal random player using a fixed seed and the same MCTS budget, first with the initial network and then periodically. The dashboard's percentage is `(wins + 0.5 × draws) / games`, not an Elo rating. The default six-game evaluation is a sanity check only. Falling training loss or a higher score in a tiny sample is not proof of improved strength. A serious claim requires many held-out evaluation games, confidence intervals and comparisons against earlier checkpoints and stronger baselines. No such strength claim is made here.

### Use a checkpoint from another interface

The independent inference entry point accepts the web game's JSON state and returns a legal move using the trained network and MCTS:

```sh
.venv/bin/python -m az.predict --checkpoint runs/YOUR_RUN/checkpoint.pt --state position.json --simulations 128
```

The state fields are `board` (144 entries, `null`/0/1), `remaining` (two arrays of piece names), `turn`, `passes` and `moves`. This is the same shape used by the web room API. The response contains the move (`piece`, `rotation`, `flipped`, `x`, `y`, or a pass), search value and training provenance. It does not silently replace MCTS with a greedy neural-policy lookup.

## Overnight experiment (M4, 16 GB)

`configs/overnight.json` defines a new, independently seeded run from random weights. The earlier 400-game checkpoint remains untouched. The run targets **10,000 self-play games** with the same 32-channel/two-block network, 64 simulations, batch 32 and two CPU threads. It stops and saves at a **12-hour cumulative runtime budget**; reaching the budget is a pause, not a claim of successful training.

```sh
.venv/bin/python -m az.train --config configs/overnight.json --run-dir runs/overnight-01
```

The local studio also accepts `POST /api/start` with JSON `{"preset":"overnight"}`. It prevents idle system sleep on macOS only while its training subprocess exists; it does not change power settings. Keep the Mac powered and the lid open. Pause/resume remains available through the studio. Extending a time-budget-exhausted run requires increasing `max_wall_seconds` in that run's config before resuming.

Evaluation is separate from self-play and never changes policy targets or selects a model for training:

- Before learning and every 1,000 self-play games: **200 games per opponent**, alternating colors, with checkpoint-dependent seeds.
- Opponents: uniform random legal play; a deterministic scoring heuristic that preserves open corner frontiers (random tie-breaking); and the **frozen initial network with the same MCTS budget**.
- The heuristic is solely an evaluation baseline. Its geometric features do not enter the learner or its search value function.
- After training: **400 games per opponent** on a separate held-out seed range. If the time budget expires first, this final evaluation can be incomplete and no final strength claim should be made.
- Score means `(wins + 0.5*draws)/games`. Conservative 95% Hoeffding intervals use the mean scores of color-swapped seed pairs as bounded observations; draws are not treated as wins. Repeated monitoring evaluations are not independent significance tests.

Raw outcomes and seeds are written to `evaluations.jsonl`; compact summaries appear in the studio. `initial-model.pt` preserves the baseline; `models/iteration-*.pt` preserves evaluated model snapshots. Full resumable checkpoints are written every 100 completed self-play games, on graceful pause/failure, and at completion. An abrupt crash can lose progress since the last checkpoint; partially completed iterations repeat collection when resumed. This does not promise 10,000 unique retained games after interruptions.

The run records the source Git commit and Torch version. The primary question is whether final held-out score against the frozen initial model exceeds 50% with its interval above 50%, alongside results against random and heuristic opponents. This is a predeclared comparison, not a promise of expert or superhuman play. Do not tune on the held-out results and continue calling them held out.
