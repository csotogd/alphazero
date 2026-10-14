"""Follow-up ablation prompted by exact-endgame failures of near-zero priors.

No oracle, handcrafted evaluation, or training is used in these match agents.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .diagnose import Study, make_agent, paired_delta
from .network import Evaluator
from .train import atomic_json


def soften(evaluate, mode):
    def wrapped(state, legal):
        priors, value=evaluate(state,legal)
        if mode=='sqrt':
            priors=np.sqrt(priors);priors/=priors.sum()
        elif mode=='floor10':
            priors=.9*priors+.1/len(priors)
        else:raise ValueError(mode)
        return priors,value
    return wrapped


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();study=Study(args.source,args.output)
    protocol={'motivation':'Exact small endgames: winning actions with near-zero learned priors remained unvisited.',
              'screening_seed':20_000_023,'screening_games':80,'simulations':64,
              'variants':['sqrt','floor10'],'selection':'highest screening score, ties use declaration order',
              'confirmation_seed':60_000_023,'confirmation_games':200,
              'notes':'Adaptive follow-up, separate from initial protocol; frozen model; no exact oracle in gameplay.'}
    atomic_json(Path(args.output)/'protocol.json',protocol)
    ev=Evaluator(study.model(),study.game,'cpu')
    results={}
    for mode in protocol['variants']:
        results[mode]=study.evaluate(mode,make_agent(study.game,soften(ev,mode)))
    best=max(protocol['variants'],key=lambda mode:results[mode]['score_rate'])
    atomic_json(Path(args.output)/'selection.json',{'mode':best,'criterion':protocol['selection']})
    control=study.evaluate('confirm_control',make_agent(study.game,ev),games=200,seed=60_000_023)
    variant=study.evaluate('confirm_softened',make_agent(study.game,soften(ev,best)),games=200,seed=60_000_023)
    comparison=paired_delta(variant['outcomes'],control['outcomes'])
    atomic_json(Path(args.output)/'comparison.json',{'selected':best,**comparison})
    study.progress(phase='complete',status='complete')


if __name__=='__main__':main()
