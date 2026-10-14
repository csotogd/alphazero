"""Small exact endgame oracle for diagnosis, never used to train the network."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .games.pentomino import Pentomino
from .mcts import search, choose_action
from .network import Evaluator, PolicyValueNet
from .train import atomic_json, load_checkpoint


class NodeLimit(Exception):
    pass


def solve_actions(game, state, max_nodes=2000):
    """Return exact minimax action values or raise; never label a cutoff exact."""
    nodes=0
    @lru_cache(maxsize=None)
    def value(s):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:raise NodeLimit
        terminal=game.terminal_value(s)
        if terminal is not None:return terminal
        best=-1.
        for action in game.legal_actions(s):
            child=game.next_state(s,int(action))
            score=value(child)*(1 if game.current_player(child)==game.current_player(s) else -1)
            best=max(best,score)
            if best==1.:break
        return best
    result={}
    for action in game.legal_actions(state):
        child=game.next_state(state,int(action))
        result[int(action)]=value(child)*(1 if game.current_player(child)==game.current_player(state) else -1)
    return result,nodes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();source=Path(args.source);output=Path(args.output)
    torch.set_num_threads(1)
    saved=load_checkpoint(source/'checkpoint.pt');cfg=saved['config'];game=Pentomino()
    net=PolicyValueNet(game.observation_shape,game.action_size,cfg['channels'],cfg['blocks'])
    net.load_state_dict(saved['model']);ev=Evaluator(net,game,'cpu')
    candidates=[]
    with (source/'games.jsonl').open() as file:
        for i,line in enumerate(file):
            if i%100:continue
            record=json.loads(line)
            for frame in record['trajectory'][-7:-2]:
                state=game.from_web(frame['state'])
                legal=game.legal_actions(state)
                if 2<=len(legal)<=16:candidates.append(state)
    rng=np.random.default_rng(421);rng.shuffle(candidates)
    rows=[];skipped=0
    for state in candidates[:80]:
        try:values,nodes=solve_actions(game,state)
        except NodeLimit:skipped+=1;continue
        optimal=max(values.values());legal=game.legal_actions(state);priors,v=ev(state,legal)
        actions={'policy':int(legal[np.argmax(priors)])}
        coverage={}
        for sims in (64,256):
            result=search(game,state,ev,sims,np.random.default_rng(74))
            actions[str(sims)]=choose_action(result,np.random.default_rng(74),0.)
            coverage[str(sims)]=sum(int(n) for a,n in zip(result.actions,result.visits) if values[int(a)]==optimal)
        uniform=lambda s,a: (np.full(len(a),1/len(a)),ev(s,a)[1])
        actions['uniform64']=choose_action(search(game,state,uniform,64,np.random.default_rng(74)),np.random.default_rng(74),0.)
        rows.append({'legal':len(legal),'nodes':nodes,'optimal_value':optimal,'network_value':v,
                     'has_suboptimal_actions':min(values.values())<optimal,
                     'optimal_prior_mass':float(sum(p for a,p in zip(legal,priors) if values[int(a)]==optimal)),
                     'optimal_action_visits':coverage,
                     'state':game.serialize(state),
                     'chosen_exact_values':{k:values[a] for k,a in actions.items()}})
    informative=[r for r in rows if r['has_suboptimal_actions']]
    report={'attempted':min(80,len(candidates)),'solved':len(rows),'node_limit_skips':skipped,
            'decision_relevant_positions':len(informative),
            'suboptimal_choices':{k:sum(r['chosen_exact_values'][k]<r['optimal_value'] for r in informative)
                                  for k in ('policy','64','256','uniform64')},
            'value_mse_vs_exact':float(np.mean([(r['network_value']-r['optimal_value'])**2 for r in rows])) if rows else None,
            'rows':rows,'caveat':'Selected small historical endgames, not a representative game-strength estimate.'}
    atomic_json(output,report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
