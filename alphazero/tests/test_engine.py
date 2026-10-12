import json,subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from az.games import make_game
from az.games.pentomino import Pentomino,PASS,Position
from az.games.tictactoe import TicTacToe,Position as TTTPosition
from az.mcts import search,choose_action
from az.network import PolicyValueNet,Evaluator
from az.selfplay import self_play
from az.train import Config,run,load_checkpoint

def uniform(state,legal):return np.ones(len(legal))/len(legal),0.

@pytest.mark.parametrize('name',['pentomino','tictactoe'])
def test_adapter_game_terminates_and_observations_are_complete(name):
    game=make_game(name);s=game.initial_state();rng=np.random.default_rng(42)
    for ply in range(game.max_moves):
        assert game.encode(s).shape==game.observation_shape
        assert np.isfinite(game.encode(s)).all()
        if game.terminal_value(s) is not None:break
        legal=game.legal_actions(s);assert len(legal)>0;assert len(set(legal))==len(legal)
        previous=s;s=game.next_state(s,int(rng.choice(legal)))
        assert game.current_player(s)==1-game.current_player(previous)
    assert game.terminal_value(s) in (-1,0,1)
    assert len(game.legal_actions(s))==0
    with pytest.raises(ValueError):game.next_state(s,0)

def test_mcts_finds_forced_win_and_backs_up_correct_player_sign():
    game=TicTacToe();s=TTTPosition((0,0,-1,1,1,-1,-1,-1,-1),0)
    result=search(game,s,uniform,100,np.random.default_rng(3))
    assert choose_action(result,np.random.default_rng(3),0)==2
    assert result.value>0
    assert result.visits.sum()==100
    assert np.isclose(result.probabilities.sum(),1)
    assert set(result.actions)==set(game.legal_actions(s))

def test_mcts_blocks_immediate_opponent_win():
    game=TicTacToe();s=TTTPosition((1,1,-1,0,-1,-1,-1,0,-1),0)
    result=search(game,s,uniform,300,np.random.default_rng(1))
    assert choose_action(result,np.random.default_rng(1),0)==2

def test_forced_pass_and_terminal_scoring():
    game=Pentomino()
    with pytest.raises(ValueError):game.next_state(game.initial_state(),PASS)
    s=Position(boards=(0,1),remaining=(0,0),turn=0)
    assert game.terminal_value(s)==-1
    s=Position(remaining=(0,1),turn=0)
    assert game.legal_actions(s).tolist()==[PASS]
    s=game.next_state(s,PASS);assert s.turn==1 and s.passes==1

def test_selfplay_targets_are_terminal_outcomes_not_network_guesses():
    game=TicTacToe();samples,record=self_play(game,uniform,{'simulations':16,'dirichlet_alpha':.3},np.random.default_rng(4))
    assert len(samples)==record['moves']
    for i,s in enumerate(samples):
        assert s.outcome==record['outcome_player0']*((-1)**i)
        assert np.isclose(s.policy.sum(),1)

def test_python_matches_authoritative_typescript_over_complete_games():
    game=Pentomino();rng=np.random.default_rng(31);cases=[]
    for _ in range(3):
        s=game.initial_state()
        while game.terminal_value(s) is None:
            legal=game.legal_actions(s)
            cases.append({'state':game.serialize(s),'legal':[game.decode_action(int(a)) for a in legal]})
            action=int(rng.choice(legal));s=game.next_state(s,action)
        cases.append({'state':game.serialize(s),'legal':[]})
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run(['node','--experimental-strip-types',str(root/'tests/parity.mjs')],input=json.dumps(cases),text=True,capture_output=True,cwd=root)
    assert result.returncode==0,result.stderr
    assert 'parity verified' in result.stdout

def test_training_updates_weights_and_checkpoint_resumes(tmp_path):
    config=Config(game='tictactoe',iterations=1,games_per_iteration=1,simulations=8,train_steps=2,batch_size=4,channels=8,blocks=1,eval_games=2,eval_every=1)
    torch.manual_seed(config.seed);initial=PolicyValueNet((3,3,3),9,8,1)
    status=run(config,tmp_path);assert status['status']=='complete'
    saved=load_checkpoint(tmp_path/'checkpoint.pt')
    assert any(not torch.equal(initial.state_dict()[k],v) for k,v in saved['model'].items())
    assert saved['steps']==2 and len(saved['replay'])>0
    config.iterations=2;status=run(config,tmp_path,resume=True)
    assert status['steps']==4 and status['iteration']==2 and status['games_played']==2

def test_browser_bridge_roundtrip_and_action_codec():
    game=Pentomino();s=game.initial_state();rng=np.random.default_rng(8)
    for _ in range(15):
        assert game.from_web(game.serialize(s))==s
        if game.terminal_value(s) is not None:break
        for a in game.legal_actions(s):assert game.encode_action(game.decode_action(int(a)))==a
        s=game.next_state(s,int(rng.choice(game.legal_actions(s))))
