from .mcts import search,choose_action
from .replay import Sample

def self_play(game,evaluate,config,rng,progress=None):
    state=game.initial_state();pending=[];trajectory=[]
    for ply in range(game.max_moves):
        if game.terminal_value(state) is not None:break
        result=search(game,state,evaluate,config['simulations'],rng,noise=True,dirichlet_alpha=config['dirichlet_alpha'])
        pending.append((game.encode(state),result.actions,result.probabilities.astype('float32'),game.current_player(state)))
        # AlphaZero's paper samples proportional to visits in training. No greedy
        # early cutoff, symmetry augmentation, or AlphaGo Zero promotion gate.
        action=choose_action(result,rng,temperature=1.)
        state=game.next_state(state,action)
        trajectory.append({'state':game.serialize(state),'action':game.decode_action(action),'value':result.value,'simulations':result.simulations})
        if progress:progress(ply+1,trajectory[-1])
    value=game.terminal_value(state)
    if value is None:raise RuntimeError('Game exceeded its declared horizon; refusing to label a truncation as a draw')
    terminal_player=game.current_player(state)
    samples=[Sample(obs,actions,policy,value if player==terminal_player else -value) for obs,actions,policy,player in pending]
    outcome0=value if terminal_player==0 else -value
    return samples,{'moves':len(trajectory),'outcome_player0':outcome0,'final':game.serialize(state),'trajectory':trajectory}

def evaluate_random(game,evaluate,simulations,games,seed):
    import numpy as np
    rng=np.random.default_rng(seed);wins=draws=losses=0
    for index in range(games):
        model_player=index%2;state=game.initial_state()
        for _ in range(game.max_moves):
            if game.terminal_value(state) is not None:break
            if game.current_player(state)==model_player:
                result=search(game,state,evaluate,simulations,rng,noise=False)
                action=choose_action(result,rng,temperature=0.)
            else:action=int(rng.choice(game.legal_actions(state)))
            state=game.next_state(state,action)
        value=game.terminal_value(state)
        if value is None:raise RuntimeError('Evaluation exceeded game horizon')
        value=value if game.current_player(state)==model_player else -value
        wins+=value>0;draws+=value==0;losses+=value<0
    # Wilson interval for the decisive win rate; draws separately reported.
    return {'opponent':'uniform legal random','games':games,'wins':int(wins),'draws':int(draws),'losses':int(losses),'score_rate':(wins+.5*draws)/games,'seed':seed,'simulations':simulations,'note':'Small diagnostic sample; not evidence of strong play.'}
