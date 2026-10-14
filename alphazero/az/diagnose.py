"""Reproducible, bounded hypothesis tests on copies of a Pentomino checkpoint.

This is an experimental harness, not a change to the production learner. All
screening arms share evaluation seeds. Confirmation uses new seeds. Short
continuations cannot establish long-run hyperparameter optima.
"""
import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .evaluation import frontier_agent, random_agent, summarize
from .games.pentomino import Pentomino, Position, PASS, SIZE
from .mcts import search, choose_action
from .network import PolicyValueNet, Evaluator
from .replay import ReplayBuffer, Sample
from .selfplay import self_play
from .train import atomic_json, load_checkpoint


def transpose_bits(bits):
    result = 0
    while bits:
        bit = bits & -bits
        index = bit.bit_length() - 1
        y, x = divmod(index, SIZE)
        result |= 1 << (x * SIZE + y)
        bits ^= bit
    return result


class Transpose:
    """Main-diagonal reflection preserves both designated starting corners."""
    def __init__(self, game):
        lookup = {(piece, mask): action for action, (piece, mask) in game.by_action.items()}
        self.actions = {a: lookup[(p, transpose_bits(mask))]
                        for a, (p, mask) in game.by_action.items()}
        self.actions[PASS] = PASS

    def state(self, state):
        return Position(tuple(transpose_bits(b) for b in state.boards),
                        state.remaining, state.turn, state.passes, state.moves)

    def evaluator(self, evaluate, ensemble=False):
        def wrapped(state, legal):
            reflected = np.array([self.actions[int(a)] for a in legal])
            p, v = evaluate(self.state(state), reflected)
            if ensemble:
                p0, v0 = evaluate(state, legal)
                return (p+p0)/2, (v+v0)/2
            return p, v
        return wrapped


def make_agent(game, evaluate, simulations=64, c_init=1.25, mode='normal'):
    if mode == 'zero_value':
        original = evaluate
        evaluate = lambda s, a: (original(s, a)[0], 0.)
    elif mode == 'uniform_prior':
        original = evaluate
        evaluate = lambda s, a: (np.full(len(a), 1/len(a)), original(s, a)[1])
    def act(state, rng):
        if simulations == 0:
            legal = game.legal_actions(state)
            p, _ = evaluate(state, legal)
            return int(legal[rng.choice(np.flatnonzero(p == p.max()))])
        result = search(game, state, evaluate, simulations, rng, noise=False, c_init=c_init)
        return choose_action(result, rng, temperature=0.)
    return act


def paired_delta(candidate, reference, seed=731):
    """Seed-pair cluster bootstrap, preserving color pairing and arm pairing.

    99% intervals are exploratory, not familywise-adjusted significance tests.
    Confirmation results, not screening maxima, should drive conclusions.
    """
    a, b = np.asarray(candidate), np.asarray(reference)
    if len(a) != len(b) or len(a) % 2:
        raise ValueError('Matched complete color pairs required')
    scores = ((a-b)/2).reshape(-1, 2).mean(1)
    wins = ((a == 1).astype(float)-(b == 1)).reshape(-1, 2).mean(1)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scores), size=(10000, len(scores)))
    def interval(values):
        return np.quantile(values[indices].mean(1), [.005, .995]).tolist()
    return {'score_delta': float(scores.mean()), 'score_delta_ci99': interval(scores),
            'win_rate_delta': float(wins.mean()), 'win_rate_delta_ci99': interval(wins),
            'method': 'paired seed-cluster percentile bootstrap; exploratory 99%, no multiple-test guarantee'}


def match(game, candidate, opponent, games, seed, progress=None):
    if games < 2 or games % 2:
        raise ValueError('An even game count is required')
    rows = []
    for index in range(games):
        pair_seed = seed + index//2
        crng, orng = np.random.default_rng(pair_seed), np.random.default_rng(pair_seed+1_000_000)
        player = index % 2
        state = game.initial_state()
        for _ in range(game.max_moves):
            if game.terminal_value(state) is not None:
                break
            action = candidate(state, crng) if state.turn == player else opponent(state, orng)
            state = game.next_state(state, int(action))
        value = game.terminal_value(state)
        if value is None:
            raise RuntimeError('Nonterminal evaluation result')
        outcome = float(value if state.turn == player else -value)
        rows.append({'outcome': outcome, 'player': player,
                     'candidate_points': state.boards[player].bit_count(),
                     'opponent_points': state.boards[1-player].bit_count(),
                     'both_all_pieces': state.remaining == (0, 0)})
        if progress:
            progress(index+1, games)
    result = summarize([r['outcome'] for r in rows], seed)
    result.update(seed=seed, rows=rows, outcomes=[r['outcome'] for r in rows],
                  win_rate=result['wins']/games,
                  mean_candidate_points=float(np.mean([r['candidate_points'] for r in rows])),
                  mean_opponent_points=float(np.mean([r['opponent_points'] for r in rows])),
                  full_board_inventory_draws=sum(r['both_all_pieces'] for r in rows),
                  by_color={str(p): dict(Counter(str(r['outcome']) for r in rows if r['player']==p)) for p in (0,1)})
    return result


class Study:
    def __init__(self, source, output, games=80, continuation_games=80):
        self.source, self.output = Path(source), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.games, self.continuation_games = games, continuation_games
        self.game = Pentomino()
        self.saved = load_checkpoint(self.source/'checkpoint.pt')
        self.config = self.saved['config']
        torch.set_num_threads(2)
        self.symmetry = Transpose(self.game)
        self.started = time.monotonic()
        self.rows = []
        self.results_path = self.output/'matches.jsonl'
        if self.results_path.exists():
            self.rows = [json.loads(line) for line in self.results_path.read_text().splitlines()]
        self.done = {row['name']: row for row in self.rows}

    def progress(self, **data):
        atomic_json(self.output/'status.json', {'elapsed_seconds': time.monotonic()-self.started,
                    'completed_matches': len(self.rows), **data})

    def model(self, weights=None):
        net = PolicyValueNet(self.game.observation_shape, self.game.action_size,
                             self.config['channels'], self.config['blocks'])
        net.load_state_dict(self.saved['model'] if weights is None else weights)
        return net

    def evaluate(self, name, agent, opponent=None, games=None, seed=20_000_023):
        if name in self.done:
            return self.done[name]
        self.progress(phase='evaluation', name=name, completed=0, total=games or self.games)
        started = time.monotonic()
        result = match(self.game, agent, opponent or random_agent(self.game), games or self.games, seed,
                       lambda n,total: self.progress(phase='evaluation',name=name,completed=n,total=total))
        result.update(name=name, seconds=time.monotonic()-started)
        with self.results_path.open('a') as file:
            file.write(json.dumps(result)+'\n')
        self.rows.append(result); self.done[name] = result
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','outcomes')}),flush=True)
        return result

    def audit(self):
        self.progress(phase='audit')
        metrics = [json.loads(line) for line in (self.source/'metrics.jsonl').read_text().splitlines()]
        rate_changes=[]
        for item in metrics:
            if not rate_changes or item['learning_rate'] != rate_changes[-1]['learning_rate']:
                rate_changes.append({k:item[k] for k in ('games','steps','learning_rate')})
        groups=defaultdict(list); sampled=[]
        with (self.source/'games.jsonl').open() as file:
            for i,line in enumerate(file):
                record=json.loads(line)
                final=record['final']
                groups[(i//1000)*1000].append((record['outcome_player0'], len(final['remaining'][0]), len(final['remaining'][1])))
                if i % 100 == 0:
                    sampled.append(record)
        net=self.model(); ev=Evaluator(net,self.game,'cpu')
        counts=[]; covered=[]; top_prior=[]; values=[]; policy_js=[]; value_sym=[]; legal_mass=[]
        t_ev=self.symmetry.evaluator(ev)
        with torch.inference_mode():
            for record in sampled:
                for frame in record['trajectory'][2::6]:
                    state=self.game.from_web(frame['state'])
                    if self.game.terminal_value(state) is not None:
                        continue
                    legal=self.game.legal_actions(state)
                    transformed=self.symmetry.state(state)
                    assert set(self.symmetry.actions[int(a)] for a in legal)==set(self.game.legal_actions(transformed))
                    assert self.symmetry.state(transformed)==state
                    p,v=ev(state,legal); q,w=t_ev(state,legal)
                    mean=(p+q)/2
                    js=.5*np.sum(p*np.log((p+1e-30)/(mean+1e-30)))+.5*np.sum(q*np.log((q+1e-30)/(mean+1e-30)))
                    policy_js.append(float(js));value_sym.append(abs(v-w))
                    logits,_=net(torch.from_numpy(self.game.encode(state)).unsqueeze(0))
                    legal_mass.append(float(torch.softmax(logits,1)[0,legal].sum()))
                    counts.append(len(legal));top_prior.append(float(p.max()))
                    value_target=record['outcome_player0']*(1 if state.turn==0 else -1)
                    values.append((v,value_target))
                    if len(covered)<48 and len(legal)>1:
                        r=search(self.game,state,ev,64,np.random.default_rng(71),noise=False)
                        covered.append({'legal':len(legal),'visited':int((r.visits>0).sum()),
                                        'max_visits':int(r.visits.max()),'prior_max':float(p.max())})
        records=[]
        for start,items in groups.items():
            n=len(items)
            records.append({'start_game':start+1,'games':n,'draw_rate':sum(x[0]==0 for x in items)/n,
                'full_inventory_draw_rate':sum(x[1]==x[2]==0 for x in items)/n,
                'mean_remaining_pieces':float(np.mean([x[1]+x[2] for x in items]))})
        replay=self.saved['replay']
        heads={name:sum(p.numel() for p in module.parameters()) for name,module in
               [('trunk',net.trunk),('policy',net.policy),('value',net.value)]}
        data={'rate_changes':rate_changes,'selfplay_windows':records,'replay_positions':len(replay),
              'replay_outcomes':dict(Counter(str(s['outcome']) for s in replay)),
              'approx_replay_games':len(replay)/(sum(len(r['trajectory']) for r in sampled)/len(sampled)),
              'legal_actions':{'median':float(np.median(counts)),'p90':float(np.quantile(counts,.9)),'max':max(counts)},
              'search_coverage_samples':covered,'mean_max_prior':float(np.mean(top_prior)),
              'mean_probability_mass_on_legal_actions':float(np.mean(legal_mass)),
              'diagonal_symmetry':{'states':len(policy_js),'mean_policy_js':float(np.mean(policy_js)),
                                   'mean_abs_value_difference':float(np.mean(value_sym))},
              'historical_value_mse':float(np.mean([(v-z)**2 for v,z in values])),
              'parameter_counts':heads,
              'caveat':'Historical value targets come from older self-play policies, not minimax truth.'}
        atomic_json(self.output/'audit.json',data)
        print(json.dumps({'audit':data}),flush=True)

    def inference(self):
        ev=Evaluator(self.model(),self.game,'cpu')
        variants=[('last_64',ev,64,1.25,'normal'),('last_policy_only',ev,0,1.25,'normal'),
                  ('last_16',ev,16,1.25,'normal'),('last_256',ev,256,1.25,'normal'),
                  ('last_explore5',ev,64,5.,'normal'),('last_zero_value',ev,64,1.25,'zero_value'),
                  ('last_uniform_prior',ev,64,1.25,'uniform_prior'),
                  ('last_transposed',self.symmetry.evaluator(ev),64,1.25,'normal'),
                  ('last_symmetry_ensemble',self.symmetry.evaluator(ev,True),64,1.25,'normal')]
        for name,evaluator,sims,c,mode in variants:
            self.evaluate(name,make_agent(self.game,evaluator,sims,c,mode))
        for iteration in (100,300,500):
            saved=load_checkpoint(self.source/f'models/iteration-{iteration:05d}.pt')
            evaluator=Evaluator(self.model(saved['model']),self.game,'cpu')
            self.evaluate(f'checkpoint_{iteration*10}',make_agent(self.game,evaluator))
        self.evaluate('random_vs_random',random_agent(self.game),games=400)
        self.evaluate('frontier_vs_random',frontier_agent(self.game),games=200)

    def historical_replay(self):
        """Uniform reservoir of original self-play positions with exact saved targets.

        Full MCTS policies weren't logged historically. Only the current replay
        has those. Reconstructing policy targets from chosen actions would change
        the objective. Instead, history is used for value-only rehearsal below.
        """
        rng=np.random.default_rng(404)
        records=[];seen=0
        with (self.source/'games.jsonl').open() as file:
            for i,line in enumerate(file):
                if i>=5000:break
                if i%5:continue
                record=json.loads(line);state=self.game.initial_state()
                for frame in record['trajectory']:
                    z=record['outcome_player0']*(1 if state.turn==0 else -1)
                    item=(self.game.encode(state),z)
                    seen+=1
                    if len(records)<5000:records.append(item)
                    else:
                        j=int(rng.integers(seen))
                        if j<5000:records[j]=item
                    state=self.game.from_web(frame['state'])
        return records

    def continuation(self, name, seed, learning_rate, *, history=None, noise_alpha=None, replay_capacity=None):
        path=self.output/f'{name}.pt'
        if path.exists():return load_checkpoint(path)
        net=self.model()
        optimizer=torch.optim.SGD(net.parameters(),lr=learning_rate,momentum=self.config['momentum'])
        optimizer.load_state_dict(copy.deepcopy(self.saved['optimizer']))
        for group in optimizer.param_groups:group['lr']=learning_rate
        replay=ReplayBuffer(replay_capacity or self.config['replay_capacity']);replay.load_state_dict(self.saved['replay'])
        rng=np.random.default_rng(seed);train_rng=np.random.default_rng(seed+500_000)
        torch.manual_seed(seed)
        ev=Evaluator(net,self.game,'cpu');cfg=dict(self.config)
        if noise_alpha is not None:cfg['dirichlet_alpha']=noise_alpha
        losses=[];outcomes=[];started=time.monotonic()
        for iteration in range(self.continuation_games//10):
            for g in range(10):
                self.progress(phase='continuation',name=name,games=iteration*10+g,total=self.continuation_games)
                samples,record=self_play(self.game,ev,cfg,rng)
                replay.extend(samples);outcomes.append(record['outcome_player0'])
            net.train()
            for _ in range(40):
                observations,targets,z=replay.batch(32,self.game.action_size,train_rng,'cpu')
                logits,v=net(observations)
                policy_loss=-(targets*F.log_softmax(logits,1)).sum(1).mean()
                value_loss=F.mse_loss(v,z)
                # Fixed extra update term, never fabricated historical policy labels.
                if history is not None:
                    indices=train_rng.choice(len(history),32,replace=False)
                    obs=torch.from_numpy(np.stack([history[j][0] for j in indices]))
                    target=torch.tensor([history[j][1] for j in indices],dtype=torch.float32)
                    _,oldv=net(obs)
                    value_loss=.5*value_loss+.5*F.mse_loss(oldv,target)
                regularization=self.config['l2']*sum(p.square().sum() for p in net.parameters())
                loss=policy_loss+value_loss+regularization
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite experimental loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                losses.append(float(loss.detach()))
        saved={'model':net.state_dict(),'name':name,'seed':seed,'learning_rate':learning_rate,
               'selfplay_games':len(outcomes),'steps':len(losses),'mean_final_loss':float(np.mean(losses[-40:])),
               'seconds':time.monotonic()-started,'history_value_rehearsal':history is not None,
               'replay_capacity':replay.samples.maxlen,'final_replay_positions':len(replay),
               'dirichlet_alpha':cfg['dirichlet_alpha']}
        torch.save(saved,path)
        atomic_json(self.output/f'{name}.json',{k:v for k,v in saved.items() if k!='model'})
        return saved

    def training(self):
        history=self.historical_replay()
        for seed in (811,977):
            for label,lr,kwargs in [('control',.002,{}),('lr_low',.0002,{}),('lr_high',.02,{}),
                                    ('history_value',.002,{'history':history}),
                                    ('noise_spread',.002,{'noise_alpha':.3})]:
                name=f'{label}_seed{seed}'
                saved=self.continuation(name,seed,lr,**kwargs)
                ev=Evaluator(self.model(saved['model']),self.game,'cpu')
                self.evaluate(name,make_agent(self.game,ev))

    def confirmation(self):
        # Predeclared selection: highest screening score among trained low/high-LR,
        # rehearsal and noise variants (average both training seeds). Compare each
        # seed against its matched control, without selecting a lucky seed.
        labels=['lr_low','lr_high','history_value','noise_spread']
        best=max(labels,key=lambda name:np.mean([self.done[f'{name}_seed{s}']['score_rate'] for s in (811,977)]))
        atomic_json(self.output/'selection.json',{'selected_training_variant':best,
                    'criterion':'highest mean screening score across both training seeds',
                    'confirmation_seed':40_000_023,'games_per_seed':200,
                    'warning':'All screening arms are exploratory. Never select the better training seed.'})
        for label in ('control',best):
            for seed in (811,977):
                saved=load_checkpoint(self.output/f'{label}_seed{seed}.pt')
                ev=Evaluator(self.model(saved['model']),self.game,'cpu')
                self.evaluate(f'confirm_{label}_seed{seed}',make_agent(self.game,ev),games=200,seed=40_000_023)
        ev=Evaluator(self.model(),self.game,'cpu')
        self.evaluate('confirm_last64',make_agent(self.game,ev),games=200,seed=40_000_023)
        self.evaluate('confirm_last256',make_agent(self.game,ev,256),games=200,seed=40_000_023)

    def report(self):
        base=self.done['last_64']['outcomes']
        contrasts={row['name']:paired_delta(row['outcomes'],base) for row in self.rows
                   if row['seed']==20_000_023 and row['games']==self.games and row['name']!='last_64'}
        for seed in (811,977):
            control=self.done[f'control_seed{seed}']['outcomes']
            for label in ('lr_low','lr_high','history_value','noise_spread'):
                name=f'{label}_seed{seed}'
                contrasts[f'{name}_vs_control']=paired_delta(self.done[name]['outcomes'],control)
        selected=json.loads((self.output/'selection.json').read_text())['selected_training_variant']
        for seed in (811,977):
            a=self.done[f'confirm_{selected}_seed{seed}']['outcomes']
            b=self.done[f'confirm_control_seed{seed}']['outcomes']
            contrasts[f'confirm_seed{seed}']=paired_delta(a,b)
        contrasts['confirm_search256_vs64']=paired_delta(self.done['confirm_last256']['outcomes'],self.done['confirm_last64']['outcomes'])
        atomic_json(self.output/'comparisons.json',contrasts)
        self.progress(phase='complete',status='complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--games',type=int,default=80)
    parser.add_argument('--continuation-games',type=int,default=80)
    parser.add_argument('--phase',choices=['all','audit','inference','training','confirmation','report'],default='all')
    args=parser.parse_args()
    if args.games<2 or args.games%2 or args.continuation_games<10 or args.continuation_games%10:
        parser.error('Even evaluation count >=2; continuation count must be a positive multiple of 10')
    study=Study(args.source,args.output,args.games,args.continuation_games)
    protocol={'source':str(study.source.resolve()),'source_steps':study.saved['steps'],
              'screening_games':args.games,'continuation_games_per_arm':args.continuation_games,
              'training_seeds':[811,977],'screening_seed':20_000_023,'confirmation_seed':40_000_023,
              'confirmation_games_per_arm':200,'threads':2,
              'hypotheses':['learning_rate','search_budget_and_exploration','history_forgetting',
                            'value_and_prior_quality','diagonal_symmetry','draws_and_color','implementation'],
              'limitations':['short continuations, not retraining from scratch','two training seeds only',
                 'history is value-only rehearsal: historical policy distributions were not logged',
                 'symmetry ensemble costs two evaluations per leaf',
                 'value predictions against random are not exact minimax values',
                 'screening comparisons are exploratory; independent confirmation tests one selected training variant and search budget']}
    protocol_path=study.output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Study protocol changed; use a new output directory')
    atomic_json(protocol_path,protocol)
    for phase in ('audit','inference','training','confirmation','report'):
        if args.phase in ('all',phase):getattr(study,phase)()


if __name__=='__main__':main()
