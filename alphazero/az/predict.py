"""Use a trained checkpoint on a game state exported by the web engine."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .games import make_game
from .network import PolicyValueNet,Evaluator
from .mcts import search,choose_action
from .train import load_checkpoint

def predict(checkpoint,state,simulations=128,seed=0):
    saved=load_checkpoint(checkpoint);config=saved['config'];torch.set_num_threads(config['threads'])
    game=make_game(config['game'])
    if not hasattr(game,'from_web'):raise ValueError('This adapter does not expose a browser-state bridge')
    position=game.from_web(state)
    if game.terminal_value(position) is not None:raise ValueError('The game is finished')
    network=PolicyValueNet(game.observation_shape,game.action_size,config['channels'],config['blocks'])
    network.load_state_dict(saved['model']);rng=np.random.default_rng(seed)
    result=search(game,position,Evaluator(network,game,'cpu'),simulations,rng,noise=False)
    action=choose_action(result,rng,0)
    return {'move':game.decode_action(action),'value':result.value,'simulations':simulations,'trained_steps':saved['steps'],'trained_games':saved['games_played']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--state',type=Path,required=True);parser.add_argument('--simulations',type=int,default=128)
    args=parser.parse_args();print(json.dumps(predict(args.checkpoint,json.loads(args.state.read_text()),args.simulations)))
if __name__=='__main__':main()
