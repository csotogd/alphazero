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

## Second-player-focused continuation

The default learner trains on both players. `training_player: 1` is an explicit
experimental adaptation: both seats still use the latest network and noisy MCTS
in self-play, but **only positions with player 1 to move enter replay**. Rewards
remain the terminal win/draw/loss from that player's perspective. This does not
move the second player first, train against random play, or reproduce the paper's
balanced data distribution. Specialization can reduce first-player strength.

```sh
.venv/bin/python -m az.train --config configs/second-player.json \
  --initialize-from runs/YOUR_ORIGINAL_RUN/checkpoint.pt \
  --run-dir runs/second-player
# Resume this continuation, not the source run:
.venv/bin/python -m az.train --config configs/second-player.json \
  --run-dir runs/second-player --resume
```

The supplied config collects 320 games (40 iterations × 8), with 1,000 simulations
per move, 1,280 SGD updates, learning rate 0.002, and a two-hour elapsed-time cap.
Every iteration saves a resumable checkpoint. `--initialize-from` imports **only
weights**; optimizer, replay, RNG and counters start fresh. The source file stays
unchanged, and its hash and training counts are recorded in `initialization.json`.
Resume restores this continuation's complete state and rejects a change of seat.
To extend a time-limited run, increase its elapsed-time cap before resuming.

Evaluation records `by_player` win/draw/loss counts for both seats. Monitoring uses
40 games on fixed seeds before learning and every 10 iterations. These repeated
monitoring games are not held-out evidence. The final test compares the learner
and the frozen source on the same 200 fresh games at the same search budget, with
100 games per seat; the time cap can pause the run before final evaluation.
Generated models and live logs remain local and are excluded from Git.

The README image can be regenerated with `python scripts/plot_diagnosis.py` from
this directory, using an environment with the optional `matplotlib` package.
It reads the committed diagnostic JSON; it does not run training or invent data.

## Game adapter interface

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

## Controlled diagnostic study

`az.diagnose` tests seven families of explanations for a plateau using copies of an
existing Pentomino checkpoint. It does not modify the source checkpoint, the
production training defaults, or the browser game. The study is finite and can
take tens of minutes on a laptop; it uses two CPU threads.

```sh
.venv/bin/python -m az.diagnose --source runs/YOUR_RUN --output runs/diagnosis
.venv/bin/python -m az.endgame_audit --source runs/YOUR_RUN --output runs/diagnosis/endgames.json
.venv/bin/python -m az.diagnosis_report --study runs/diagnosis --output reports/diagnosis.html
```

The source must have the overnight run's full checkpoint, game log and snapshots
at iterations 100, 300 and 500. The harness writes its protocol before testing.
Default screening uses 80 color-alternated games per arm with shared seeds:

- Frozen-model comparisons: greedy policy, 16/64/256 MCTS simulations, increased
  PUCT exploration, zero neural value, uniform policy prior, reflected input,
  reflection ensemble, earlier checkpoints, and two reference opponents.
- Five continuation arms, each with two training seeds and 80 new self-play games:
  current learning rate 0.002, 0.0002, 0.02, historical value rehearsal, and more
  diffuse Dirichlet noise. Each starts with the same model, optimizer momentum
  and replay; training uses 320 gradient steps per arm.
- The best non-control continuation by mean screening score across both seeds is
  compared with its controls on 200 new games per seed. Both training seeds are
  retained. The original model with 64 versus 256 simulations is independently
  compared on that same new seed range.
- A bounded exact endgame solver checks whether selected small positions contain
  avoidable decision errors. It refuses to return an exact answer after a node
  cutoff and does not supply training labels or gameplay hints to the learner.

Results include raw game outcomes, points, color splits, timings, model copies,
and exploratory 99% paired seed-cluster bootstrap intervals. The intervals are
not simultaneous multiple-comparison guarantees. Short continuations with two
training seeds cannot establish a globally optimal learning rate or exclude
long-run architecture/data limitations.

Only the current replay stores historical MCTS distributions. Older game logs
store moves and outcomes, so the rehearsal arm uses **value labels only**, never
fabricated policy targets. It also changes batch composition/BatchNorm updates;
it is not a pure replay-capacity ablation. Reflection ensembling doubles network
evaluations per search leaf and is not a fixed-time comparison. Game outcomes
against random play are not exact minimax labels for the value network.

Completed match arms and complete continuation models are reused on rerun after
protocol validation. An interrupted continuation restarts from the source model;
there is no mid-arm resume. Keep the original source checkpoint unchanged during
a study. `--phase` runs an individual stage after its prerequisites exist.

The October diagnostic study also records explicitly labelled adaptive follow-ups.
They use new confirmation seeds and leave the initial protocol identifiable:

```sh
.venv/bin/python -m az.prior_diagnosis --source runs/YOUR_RUN --output runs/diagnosis/prior-followup
.venv/bin/python -m az.diagnose_followups --source runs/YOUR_RUN --output runs/diagnosis --phase symmetry
.venv/bin/python -m az.diagnose_followups --source runs/YOUR_RUN --output runs/diagnosis --phase capacity
.venv/bin/python -m az.diagnose_followups --source runs/YOUR_RUN --output runs/diagnosis --phase capacity-confirmation
.venv/bin/python -m az.diagnose_followups --source runs/YOUR_RUN --output runs/diagnosis --phase entropy
.venv/bin/python -m az.diagnose_followups --source runs/YOUR_RUN --output runs/diagnosis --phase export
.venv/bin/python -m az.diagnose_followups --source runs/YOUR_RUN --output runs/diagnosis --phase search1000
```

The capacity arm isolates replay eviction by retaining the original 5,000 examples
while collecting 80 further games, reaching about 6,500 positions under a 20,000
cap. It does not test a full 20,000-position buffer. Its confirmation runs only if
both screening seeds improve win counts and their mean score gain is at least
0.025. The source run must remain unchanged across all phases. Regenerate the
HTML report after all requested follow-ups finish; it includes their results and
distinguishes exploratory from confirmation comparisons.

The export phase packages both selected training seeds for `az.predict`, including
cumulative training counters and architecture metadata. These files are inference
snapshots, not resumable training checkpoints; the original full checkpoint is
preserved.

The user-requested `search1000` phase evaluates the unchanged original model for
200 games with 1,000 simulations, paired against the existing 64/256 confirmation
arms. It records move timings and paired differences. Because it was requested
after observing the earlier results and reuses their seeds, this is an adaptive
follow-up, not a fresh independent confirmation. It performs no training.
