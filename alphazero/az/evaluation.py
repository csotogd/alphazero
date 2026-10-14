"""Evaluation-only opponents and paired, reproducible matches.

Handcrafted features below belong exclusively to the baseline opponent; they
never enter self-play, network inputs, search evaluation or training targets.
"""
from collections import Counter
from math import log, sqrt
import numpy as np
from .mcts import search, choose_action


def search_agent(game, evaluator, simulations):
    def act(state, rng):
        result = search(game, state, evaluator, simulations, rng, noise=False)
        return choose_action(result, rng, temperature=0.)
    return act


def random_agent(game):
    return lambda state, rng: int(rng.choice(game.legal_actions(state)))


def frontier_agent(game):
    """Pentomino baseline: preserve open corner frontier and reduce rival space."""
    if game.name != 'pentomino':
        raise ValueError('The frontier heuristic is specific to Pentomino')
    from .games.pentomino import corners, neighbors, FULL, SIZE

    def act(state, rng):
        legal = game.legal_actions(state)
        if len(legal) == 1:
            return int(legal[0])
        player = game.current_player(state)
        scores = []
        for action in legal:
            next_state = game.next_state(state, int(action))
            own, other = next_state.boards[player], next_state.boards[1-player]
            occupied = own | other
            own_frontier = corners(own) & ~(occupied | neighbors(own)) & FULL
            rival_frontier = corners(other) & ~(occupied | neighbors(other)) & FULL
            move = game.decode_action(int(action))
            advance = move['x'] + move['y']
            if player == 1:
                advance = 2 * (SIZE - 1) - advance
            scores.append(own_frontier.bit_count() - .25*rival_frontier.bit_count() + .01*advance)
        scores = np.asarray(scores)
        return int(legal[rng.choice(np.flatnonzero(scores == scores.max()))])
    return act


def summarize(outcomes, seed):
    """Conservative bound on independent seed-pair scores in [0,1]."""
    if not outcomes or len(outcomes) % 2:
        raise ValueError('Evaluation requires complete color-swapped pairs')
    counts = Counter(outcomes)
    scores = (np.asarray(outcomes, dtype=np.float64) + 1.) / 2.
    pairs = scores.reshape(-1, 2).mean(axis=1)
    radius = sqrt(log(2/.05)/(2*len(pairs)))
    low, high = max(0.,float(pairs.mean())-radius),min(1.,float(pairs.mean())+radius)
    return {'games':len(outcomes), 'wins':counts[1.], 'draws':counts[0.],
            'losses':counts[-1.], 'score_rate':float(scores.mean()),
            'score_ci95':[float(low),float(high)],
            'interval_method':'95% Hoeffding bound on seed-pair mean scores; conservative',
            'note':'Descriptive interval; repeated evaluations are not independent significance tests.'}


def evaluate_match(game, candidate, opponent, games, seed, progress=None):
    if games < 2 or games % 2:
        raise ValueError('Use an even number of evaluation games >= 2')
    outcomes = []
    for index in range(games):
        # Independent RNGs per agent; swap colors under each pair's seed.
        pair_seed = seed + index // 2
        candidate_rng = np.random.default_rng(pair_seed)
        opponent_rng = np.random.default_rng(pair_seed + 1_000_000)
        candidate_player = index % 2
        state = game.initial_state()
        for _ in range(game.max_moves):
            if game.terminal_value(state) is not None:
                break
            action = (candidate(state,candidate_rng) if game.current_player(state)==candidate_player
                      else opponent(state,opponent_rng))
            state = game.next_state(state, int(action))
        value = game.terminal_value(state)
        if value is None:
            raise RuntimeError('Evaluation exceeded declared game horizon')
        outcome = value if game.current_player(state)==candidate_player else -value
        outcomes.append(float(outcome))
        if progress:
            progress(index+1, games, game.serialize(state))
    by_player={}
    for player in (0,1):
        values=outcomes[player::2];counts=Counter(values);n=len(values)
        by_player[str(player)]={'games':n,'wins':counts[1.],'draws':counts[0.],
            'losses':counts[-1.],'win_rate':counts[1.]/n,
            'score_rate':(counts[1.]+.5*counts[0.])/n}
    return {**summarize(outcomes, seed+2_000_000), 'seed':seed, 'outcomes':outcomes,
            'by_player':by_player}


def evaluate_suite(game, evaluator, initial_evaluator, simulations, games, seed,
                   opponents=('random','frontier','initial'), progress=None):
    candidate = search_agent(game,evaluator,simulations)
    agents = {'random':random_agent(game),
              'initial':search_agent(game,initial_evaluator,simulations)}
    if 'frontier' in opponents:
        agents['frontier'] = frontier_agent(game)
    results=[]
    for offset,name in enumerate(opponents):
        if name not in agents:
            raise ValueError(f'Unknown evaluation opponent: {name}')
        callback = (lambda done,total,state,n=name:progress(n,done,total,state)) if progress else None
        result=evaluate_match(game,candidate,agents[name],games,seed+offset*100_000,callback)
        results.append({'opponent':name,'simulations':simulations,**result})
    return results
