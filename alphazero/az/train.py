"""Finite, resumable self-play training. No external services or training data."""
import argparse
from dataclasses import dataclass,asdict
from pathlib import Path
import json, os, signal, time
import numpy as np
import torch
import torch.nn.functional as F
from .games import make_game
from .network import PolicyValueNet,Evaluator
from .replay import ReplayBuffer
from .selfplay import self_play,evaluate_random

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

def atomic_json(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)

def load_checkpoint(path,device='cpu'):
    return torch.load(path,map_location=device,weights_only=True)

def run(config,run_dir,resume=False):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    config_dict=asdict(config)
    if config.iterations<1 or config.games_per_iteration<1 or config.simulations<1 or config.batch_size<1 or config.train_steps<1 or config.eval_games<2 or config.eval_games%2 or config.eval_every<1:raise ValueError('Positive training parameters and an even evaluation game count >= 2 are required')
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
    if resume:
        saved=load_checkpoint(checkpoint,device)
        for key in ('game','channels','blocks','simulations','seed','batch_size','learning_rate','momentum','l2','dirichlet_alpha','train_steps','games_per_iteration','replay_capacity','eval_games','eval_every','lr_drop_steps'):
            if json.dumps(saved['config'][key])!=json.dumps(config_dict[key]):raise ValueError(f'Resume configuration mismatch: {key}')
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);replay.load_state_dict(saved['replay'])
        rng.bit_generator.state=json.loads(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        iteration=saved['iteration'];games_played=saved['games_played'];steps=saved['steps'];history=saved['history'];evaluations=saved['evaluations'];previous_seconds=saved['elapsed_seconds']
    elif checkpoint.exists():raise ValueError('Checkpoint already exists; use --resume or a fresh run directory')
    atomic_json(run_dir/'config.json',config_dict)
    evaluator=Evaluator(net,game,device)
    status={'run':run_dir.name,'game':game.name,'status':'running','phase':'initializing','config':config_dict,'pid':os.getpid(),'parameters':sum(p.numel() for p in net.parameters()),'device':str(device)}
    def report(**changes):
        status.update(iteration=iteration,games_played=games_played,steps=steps,replay_positions=len(replay),elapsed_seconds=previous_seconds+time.time()-started,history=history,evaluations=evaluations)
        status.update(changes);atomic_json(run_dir/'status.json',status)
    def save():
        temporary=checkpoint.with_suffix('.tmp')
        torch.save({'format_version':1,'config':config_dict,'model':net.state_dict(),'optimizer':optimizer.state_dict(),'replay':replay.state_dict(),'numpy_rng':json.dumps(rng.bit_generator.state),'torch_rng':torch.get_rng_state(),'iteration':iteration,'games_played':games_played,'steps':steps,'history':history,'evaluations':evaluations,'elapsed_seconds':previous_seconds+time.time()-started},temporary)
        temporary.replace(checkpoint)
    def stop(_sig,_frame):raise KeyboardInterrupt
    old_int=signal.signal(signal.SIGINT,stop);old_term=signal.signal(signal.SIGTERM,stop)
    report()
    try:
        if not evaluations:
            report(phase='baseline evaluation')
            baseline=evaluate_random(game,evaluator,config.simulations,config.eval_games,config.seed+1000)
            evaluations.append({'iteration':0,**baseline});save();report()
        while iteration<config.iterations:
            for game_index in range(config.games_per_iteration):
                report(phase='self-play',game_in_iteration=game_index+1,ply=0)
                def progress(ply,frame):report(ply=ply,current_frame=frame)
                samples,record=self_play(game,evaluator,config_dict,rng,progress)
                replay.extend(samples);games_played+=1
                atomic_json(run_dir/'latest-game.json',record)
                with (run_dir/'games.jsonl').open('a') as log:log.write(json.dumps({'game_number':games_played,**record})+'\n')
                report(last_outcome=record['outcome_player0'])
            report(phase='learning')
            losses=[];net.train()
            for _ in range(config.train_steps):
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
                evaluations.append({'iteration':iteration,**evaluate_random(game,evaluator,config.simulations,config.eval_games,config.seed+1000)})
            save();report()
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
    args=parser.parse_args();config=Config(**json.loads(args.config.read_text()))
    run(config,args.run_dir or ROOT/'runs'/f'{config.game}-{time.strftime("%Y%m%d-%H%M%S")}',args.resume)
if __name__=='__main__':main()
