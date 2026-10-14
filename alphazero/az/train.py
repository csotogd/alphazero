"""Finite, resumable self-play training. No external services or training data."""
import argparse
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib, json, os, signal, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from .games import make_game
from .network import PolicyValueNet,Evaluator
from .replay import ReplayBuffer
from .selfplay import self_play,evaluate_random
from .evaluation import evaluate_suite

ROOT=Path(__file__).resolve().parents[1]
@dataclass
class Config:
    game: str='pentomino'
    iterations: int=10
    games_per_iteration: int=4
    simulations: int=64
    train_steps: int=16
    batch_size: int=32
    replay_capacity: int=5000
    channels: int=32
    blocks: int=2
    learning_rate: float=.02
    momentum: float=.9
    l2: float=.0001
    dirichlet_alpha: float=.03
    eval_games: int=6
    eval_every: int=2
    seed: int=7
    device: str='cpu'
    threads: int=2
    lr_drop_steps: tuple=(10000,30000,50000)
    eval_opponents: tuple=()
    final_eval_games: int=0
    checkpoint_every: int=1
    max_wall_seconds: float=0.
    training_player: int | None=None
    fixed_eval_seed: int | None=None
    compare_source_final: bool=False


def atomic_json(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)

def load_checkpoint(path,device='cpu'):
    return torch.load(path,map_location=device,weights_only=True)

def run(config,run_dir,resume=False,initialize_from=None):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    config_dict=asdict(config)
    if resume and initialize_from:raise ValueError('Use either resume or initialize_from, not both')
    if config.training_player not in (None,0,1):raise ValueError('training_player must be null, 0 or 1')
    if config.fixed_eval_seed is not None and config.fixed_eval_seed<0:raise ValueError('Invalid fixed evaluation seed')
    if config.compare_source_final and (not config.eval_opponents or not config.final_eval_games):
        raise ValueError('Source comparison requires eval_opponents and final_eval_games')
    if config.iterations<1 or config.games_per_iteration<1 or config.simulations<1 or config.batch_size<1 or config.train_steps<1 or config.eval_games<2 or config.eval_games%2 or config.eval_every<1:raise ValueError('Positive training parameters and an even evaluation game count >= 2 are required')
    if config.checkpoint_every < 1 or config.max_wall_seconds < 0 or (config.final_eval_games and (config.final_eval_games < 2 or config.final_eval_games % 2)):
        raise ValueError('Invalid checkpoint, time budget or final evaluation settings')
    torch.set_num_threads(config.threads);torch.manual_seed(config.seed)
    rng=np.random.default_rng(config.seed)
    device=torch.device(config.device)
    if device.type=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS is not available on this machine')
    game=make_game(config.game)
    net=PolicyValueNet(game.observation_shape,game.action_size,config.channels,config.blocks).to(device)
    # Explicit c||theta||² loss below; no optimizer weight decay (avoid double L2).
    optimizer=torch.optim.SGD(net.parameters(),lr=config.learning_rate,momentum=config.momentum)
    replay=ReplayBuffer(config.replay_capacity)
    iteration=games_played=steps=0;history=[];evaluations=[];started=time.time();previous_seconds=0.
    checkpoint=run_dir/'checkpoint.pt'
    initial_path=run_dir/'initial-model.pt'
    initialization=None
    if resume:
        saved=load_checkpoint(checkpoint,device)
        defaults=asdict(Config())
        for key in ('game','channels','blocks','simulations','seed','batch_size','learning_rate','momentum','l2','dirichlet_alpha','train_steps','games_per_iteration','replay_capacity','eval_games','eval_every','lr_drop_steps','eval_opponents','final_eval_games','training_player','fixed_eval_seed','compare_source_final'):
            if json.dumps(saved['config'].get(key,defaults[key]))!=json.dumps(config_dict[key]):raise ValueError(f'Resume configuration mismatch: {key}')
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);replay.load_state_dict(saved['replay'])
        rng.bit_generator.state=json.loads(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        iteration=saved['iteration'];games_played=saved['games_played'];steps=saved['steps'];history=saved['history'];evaluations=saved['evaluations'];previous_seconds=saved['elapsed_seconds']
        initialization=saved.get('initialization')
    elif checkpoint.exists():raise ValueError('Checkpoint already exists; use --resume or a fresh run directory')
    elif initialize_from:
        source=Path(initialize_from).resolve();saved=load_checkpoint(source,device)
        for key in ('game','channels','blocks'):
            if saved['config'][key]!=config_dict[key]:raise ValueError(f'Initialization mismatch: {key}')
        net.load_state_dict(saved['model'])
        initialization={'source_checkpoint':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                        'source_steps':saved.get('steps'),'source_games':saved.get('games_played'),
                        'method':'Weights only; fresh optimizer, empty replay, new RNG and run-local counters.'}
    if initialization:atomic_json(run_dir/'initialization.json',initialization)
    atomic_json(run_dir/'config.json',config_dict)
    evaluator=Evaluator(net,game,device)
    initial_evaluator=None
    if config.eval_opponents:
        if not resume:
            torch.save(net.state_dict(),initial_path)
        if not initial_path.exists():
            raise ValueError('Frozen initial network is missing; cannot run a valid comparison')
        # Building the reference must not perturb the learner's Torch RNG.
        with torch.random.fork_rng(devices=[]):
            initial=PolicyValueNet(game.observation_shape,game.action_size,config.channels,config.blocks).to(device)
        initial.load_state_dict(load_checkpoint(initial_path,device))
        initial_evaluator=Evaluator(initial,game,device)
    try:
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        source_commit=None
    status={'run':run_dir.name,'game':game.name,'status':'running','phase':'initializing','config':config_dict,'pid':os.getpid(),'parameters':sum(p.numel() for p in net.parameters()),'device':str(device),'source_commit':source_commit,'torch_version':torch.__version__}
    def report(**changes):
        status.update(iteration=iteration,games_played=games_played,steps=steps,replay_positions=len(replay),elapsed_seconds=previous_seconds+time.time()-started,history=history,evaluations=evaluations)
        status.update(changes);atomic_json(run_dir/'status.json',status)
    def save():
        temporary=checkpoint.with_suffix('.tmp')
        torch.save({'format_version':1,'config':config_dict,'initialization':initialization,'model':net.state_dict(),'optimizer':optimizer.state_dict(),'replay':replay.state_dict(),'numpy_rng':json.dumps(rng.bit_generator.state),'torch_rng':torch.get_rng_state(),'iteration':iteration,'games_played':games_played,'steps':steps,'history':history,'evaluations':evaluations,'elapsed_seconds':previous_seconds+time.time()-started},temporary)
        temporary.replace(checkpoint)
        if iteration and (iteration%config.eval_every==0 or iteration==config.iterations):
            model_dir=run_dir/'models';model_dir.mkdir(exist_ok=True)
            model_path=model_dir/f'iteration-{iteration:05d}.pt'
            if not model_path.exists():torch.save({'config':config_dict,'model':net.state_dict(),'steps':steps,'games_played':games_played},model_path)
    def check_budget():
        if config.max_wall_seconds and previous_seconds+time.time()-started>=config.max_wall_seconds:
            status['pause_reason']='time_budget'
            raise KeyboardInterrupt
    def evaluation(held_out=False):
        check_budget()
        games=config.final_eval_games if held_out else config.eval_games
        # Monitoring rotates by default, or reuses fixed seeds for paired trends.
        # Final seeds are separate from monitoring in either mode.
        seed=(config.fixed_eval_seed+(9_000_000 if held_out else 0) if config.fixed_eval_seed is not None
              else config.seed+(9_000_000 if held_out else 10_000+iteration*1000))
        def evaluation_progress(opponent,done,total,state):
            report(evaluation_opponent=opponent,evaluation_completed=done,evaluation_total=total,
                   current_frame={'state':state},evaluation_split='held-out' if held_out else 'monitoring')
            check_budget()
        report(evaluation_candidate='learner')
        if config.eval_opponents:
            results=evaluate_suite(game,evaluator,initial_evaluator,config.simulations,games,seed,
                                   config.eval_opponents,evaluation_progress)
        else:
            results=[evaluate_random(game,evaluator,config.simulations,games,seed)]
        for result in results:result['candidate']='learner'
        if held_out and config.compare_source_final:
            report(evaluation_candidate='frozen_source')
            reference=evaluate_suite(game,initial_evaluator,initial_evaluator,config.simulations,games,seed,
                                     config.eval_opponents,evaluation_progress)
            for result in reference:result['candidate']='frozen_source'
            results.extend(reference)
        for result in results:
            result.update(iteration=iteration,split='held-out' if held_out else 'monitoring')
            with (run_dir/'evaluations.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
            evaluations.append({k:v for k,v in result.items() if k!='outcomes'})
    def stop(_sig,_frame):raise KeyboardInterrupt
    old_int=signal.signal(signal.SIGINT,stop);old_term=signal.signal(signal.SIGTERM,stop)
    report()
    try:
        if not evaluations:
            report(phase='baseline evaluation')
            save()
            evaluation();save();report()
        while iteration<config.iterations:
            check_budget()
            for game_index in range(config.games_per_iteration):
                report(phase='self-play',game_in_iteration=game_index+1,ply=0)
                def progress(ply,frame):
                    report(ply=ply,current_frame=frame)
                    check_budget()
                samples,record=self_play(game,evaluator,config_dict,rng,progress)
                replay.extend(samples);games_played+=1
                atomic_json(run_dir/'latest-game.json',record)
                with (run_dir/'games.jsonl').open('a') as log:log.write(json.dumps({'game_number':games_played,**record})+'\n')
                report(last_outcome=record['outcome_player0'])
            report(phase='learning')
            losses=[];net.train()
            for _ in range(config.train_steps):
                check_budget()
                # Explicit schedule in gradient steps, independent of iteration size.
                learning_rate=config.learning_rate*(.1**sum(steps>=milestone for milestone in config.lr_drop_steps))
                for group in optimizer.param_groups:group['lr']=learning_rate
                observations,targets,outcomes=replay.batch(config.batch_size,game.action_size,rng,device)
                logits,values=net(observations)
                policy_loss=-(targets*F.log_softmax(logits,dim=1)).sum(dim=1).mean()
                value_loss=F.mse_loss(values,outcomes)
                regularization=config.l2*sum(parameter.square().sum() for parameter in net.parameters())
                loss=policy_loss+value_loss+regularization
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss; training stopped')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();steps+=1
                losses.append([loss.item(),policy_loss.item(),value_loss.item(),regularization.item()])
            iteration+=1
            mean=np.mean(losses,axis=0)
            metric={'iteration':iteration,'games':games_played,'steps':steps,'loss':float(mean[0]),'policy_loss':float(mean[1]),'value_loss':float(mean[2]),'l2_loss':float(mean[3]),'learning_rate':learning_rate,'elapsed_seconds':previous_seconds+time.time()-started}
            history.append(metric)
            with (run_dir/'metrics.jsonl').open('a') as log:log.write(json.dumps(metric)+'\n')
            print(json.dumps(metric),flush=True)
            if iteration%config.eval_every==0 or iteration==config.iterations:
                report(phase='evaluation')
                evaluation()
            if iteration%config.checkpoint_every==0 or iteration==config.iterations:save()
            report()
        if config.final_eval_games and not any(e.get('split')=='held-out' and e['iteration']==iteration for e in evaluations):
            report(phase='evaluation')
            evaluation(held_out=True);save()
        report(status='complete',phase='complete');print(f'Completed. Checkpoint: {checkpoint}',flush=True)
    except KeyboardInterrupt:
        save();report(status='paused',phase='paused');print('Paused; checkpoint saved.',flush=True)
    except Exception as error:
        save();report(status='failed',phase='failed',error=str(error));raise
    finally:
        signal.signal(signal.SIGINT,old_int);signal.signal(signal.SIGTERM,old_term)
    return status

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/local.json')
    parser.add_argument('--run-dir',type=Path,default=None)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--initialize-from',type=Path,help='Start a fresh run from saved weights, with empty replay and fresh optimizer')
    args=parser.parse_args();config=Config(**json.loads(args.config.read_text()))
    run(config,args.run_dir or ROOT/'runs'/f'{config.game}-{time.strftime("%Y%m%d-%H%M%S")}',args.resume,args.initialize_from)
if __name__=='__main__':main()
