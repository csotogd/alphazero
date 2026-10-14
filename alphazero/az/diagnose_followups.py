"""Reproduce the explicitly labelled adaptive follow-ups of a diagnostic study."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .diagnose import Study, make_agent, paired_delta
from .network import Evaluator
from .train import atomic_json, load_checkpoint


def matches(path):
    return {r['name']:r for r in map(json.loads,Path(path).read_text().splitlines())}


def symmetry(source, base):
    out=base/'symmetry-confirmation';s=Study(source,out)
    protocol={'motivation':'Reflection ensemble had the highest frozen-model screening score.',
              'games':200,'seed':40_000_023,'comparison':'Main study confirm_last64 on identical seeds',
              'cost':'Two network evaluations per leaf, not equal time.',
              'type':'Adaptive follow-up, independent evaluation seeds'}
    atomic_json(out/'protocol.json',protocol)
    ev=Evaluator(s.model(),s.game,'cpu')
    s.evaluate('symmetry_ensemble',make_agent(s.game,s.symmetry.evaluator(ev,True)),games=200,seed=40_000_023)
    s.progress(phase='complete',status='complete')


def search1000(source, base):
    """User-requested adaptive comparison against the existing 64/256 arms."""
    out=base/'search1000';s=Study(source,out)
    references=matches(base/'matches.jsonl')
    protocol={'type':'User-requested adaptive follow-up after observing 256 simulations',
              'source':str(Path(source).resolve()),'source_steps':s.saved['steps'],
              'simulations':1000,'games':200,'seed':40_000_023,'threads':2,
              'training':False,'noise':False,'temperature':0,'c_init':1.25,
              'references':['confirm_last64','confirm_last256'],
              'limitation':'Reuses previous confirmation seeds for paired comparisons; not a new independent confirmation of a selected budget.'}
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Existing search1000 protocol differs')
    atomic_json(protocol_path,protocol)
    for name in protocol['references']:
        if references[name]['games']!=200 or references[name]['seed']!=40_000_023:
            raise ValueError('Reference evaluation does not match protocol')
    ev=Evaluator(s.model(),s.game,'cpu')
    agent=make_agent(s.game,ev,simulations=1000)
    durations=[]
    def timed_agent(state,rng):
        started=time.perf_counter()
        action=agent(state,rng)
        durations.append(time.perf_counter()-started)
        return action
    result=s.evaluate('last1000',timed_agent,games=200,seed=40_000_023)
    if durations:
        atomic_json(out/'timing.json',{'moves':len(durations),
          'mean_seconds_per_move':float(np.mean(durations)),
          'median_seconds_per_move':float(np.median(durations)),
          'p90_seconds_per_move':float(np.quantile(durations,.9)),
          'total_candidate_seconds':sum(durations)})
    atomic_json(out/'comparison.json',{
        name:paired_delta(result['outcomes'],references[name]['outcomes'])
        for name in protocol['references']})
    s.progress(phase='complete',status='complete')


def capacity(source,base):
    out=base/'replay-capacity';s=Study(source,out)
    protocol={'motivation':'Isolate replay eviction from value-only historical rehearsal.',
              'source_steps':s.saved['steps'],'training_seeds':[811,977],
              'selfplay_games_per_seed':80,'learning_rate':.002,'capacity':20000,
              'initial_positions':5000,'screening_games':80,'screening_seed':20_000_023,
              'control':'Main study control_seed811 and control_seed977; same initial replay, optimizer, seeds and update schedule.',
              'limitation':'Short continuation fills only about 6500 positions; tests retaining oldest examples, not a full 20000-position replay.'}
    atomic_json(out/'protocol.json',protocol)
    for seed in (811,977):
        name=f'capacity20000_seed{seed}'
        saved=s.continuation(name,seed,.002,replay_capacity=20000)
        ev=Evaluator(s.model(saved['model']),s.game,'cpu')
        s.evaluate(name,make_agent(s.game,ev))
    s.progress(phase='complete',status='complete')


def capacity_confirmation(source,base):
    out=base/'replay-capacity-confirmation';out.mkdir(parents=True,exist_ok=True)
    protocol_path=out/'protocol.json'
    if not protocol_path.exists():
        atomic_json(protocol_path,{'type':'Adaptive confirmation of replay retention',
          'condition':'Both training seeds improve win count versus their matching control and the mean score improvement is at least 0.025',
          'training_seeds':[811,977],'games_per_seed':200,'seed':40_000_023,
          'reference':'Main study confirm_control_seed811 and confirm_control_seed977 on identical evaluation seeds',
          'no_training':'Evaluate the two already saved capacity models without further updates'})
    controls=matches(base/'matches.jsonl')
    rows=list(matches(base/'replay-capacity/matches.jsonl').values())
    if len(rows)!=2:raise ValueError('Both replay-capacity screening arms must finish first')
    passed=all(r['wins']>controls['control_seed'+r['name'].split('_seed')[-1]]['wins'] for r in rows)
    delta=float(np.mean([r['score_rate']-controls['control_seed'+r['name'].split('_seed')[-1]]['score_rate'] for r in rows]))
    atomic_json(out/'decision.json',{'condition_met':passed and delta>=.025,'mean_screening_score_delta':delta})
    if not (passed and delta>=.025):return
    s=Study(source,out)
    for seed in (811,977):
        saved=load_checkpoint(base/f'replay-capacity/capacity20000_seed{seed}.pt')
        ev=Evaluator(s.model(saved['model']),s.game,'cpu')
        s.evaluate(f'capacity_seed{seed}',make_agent(s.game,ev),games=200,seed=40_000_023)
    s.progress(phase='complete',status='complete')


def entropy(source,base):
    s=Study(source,base/'entropy-work');torch.set_num_threads(1)
    states=[]
    with (Path(source)/'games.jsonl').open() as file:
        for i,line in enumerate(file):
            if i%200:continue
            record=json.loads(line)
            for frame in record['trajectory'][2::6]:
                state=s.game.from_web(frame['state']);legal=s.game.legal_actions(state)
                if len(legal)>1:states.append((state,legal))
    rows=[]
    for iteration in (100,200,300,400,500,600):
        saved=load_checkpoint(Path(source)/f'models/iteration-{iteration:05d}.pt')
        ev=Evaluator(s.model(saved['model']),s.game,'cpu');entropies=[];peaks=[]
        for state,legal in states:
            p,_=ev(state,legal)
            entropies.append(float(-(p*np.log(p+1e-30)).sum()/np.log(len(legal))))
            peaks.append(float(p.max()))
        rows.append({'training_games':iteration*10,'states':len(states),
                     'mean_normalized_policy_entropy':float(np.mean(entropies)),
                     'mean_max_prior':float(np.mean(peaks))})
    atomic_json(base/'policy_entropy.json',rows)


def export(source,base):
    """Export both selected training seeds for the existing inference bridge.

    These are inference snapshots; the source full checkpoint stays resumable.
    Counts include the base training, unlike per-arm diagnostic step counts.
    """
    initial=load_checkpoint(Path(source)/'checkpoint.pt')
    label=json.loads((base/'selection.json').read_text())['selected_training_variant']
    output=base/'inference-models';output.mkdir(exist_ok=True)
    records=[]
    for seed in (811,977):
        arm=load_checkpoint(base/f'{label}_seed{seed}.pt')
        config=dict(initial['config'])
        config.update(learning_rate=arm['learning_rate'],lr_drop_steps=[],
                      dirichlet_alpha=arm['dirichlet_alpha'])
        result={'model':arm['model'],'config':config,
                'steps':initial['steps']+arm['steps'],
                'games_played':initial['games_played']+arm['selfplay_games'],
                'inference_only':True,'base_run':str(Path(source).resolve()),
                'experimental_variant':arm['name'],'continuation_steps':arm['steps'],
                'continuation_games':arm['selfplay_games'],
                'note':'Selected on screening mean across both seeds; neither training seed was selected over the other. Not a full resumable checkpoint.'}
        path=output/f'{label}_seed{seed}.pt';torch.save(result,path)
        records.append({k:v for k,v in result.items() if k not in ('model','config')})
    atomic_json(output/'manifest.json',records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=['symmetry','capacity','capacity-confirmation','entropy','export','search1000'],required=True)
    args=parser.parse_args()
    functions={'symmetry':symmetry,'capacity':capacity,'capacity-confirmation':capacity_confirmation,'entropy':entropy,'export':export,'search1000':search1000}
    functions[args.phase](args.source,Path(args.output))


if __name__=='__main__':main()
