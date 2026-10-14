from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from az.games import make_game
from az.selfplay import self_play
from az.train import Config, run, load_checkpoint


def uniform(state, legal):
    return np.full(len(legal),1/len(legal)),0.


@pytest.mark.parametrize('name',['tictactoe','pentomino'])
def test_second_player_filter_preserves_game_search_targets_and_reward_sign(name):
    game=make_game(name)
    config={'simulations':8,'dirichlet_alpha':.03}
    full,original=self_play(game,uniform,config,np.random.default_rng(42))
    selected,record=self_play(game,uniform,{**config,'training_player':1},np.random.default_rng(42))
    assert record['trajectory']==original['trajectory']
    assert record['final']==original['final']
    assert len(selected)==len(full[1::2])>0
    for actual,expected in zip(selected,full[1::2]):
        np.testing.assert_array_equal(actual.observation,expected.observation)
        np.testing.assert_array_equal(actual.actions,expected.actions)
        np.testing.assert_array_equal(actual.policy,expected.policy)
        assert actual.outcome==expected.outcome==-record['outcome_player0']
    if name=='pentomino':
        assert all(np.all(sample.observation[26]==1) for sample in selected)


def test_weights_initialization_role_focus_evaluation_and_resume(tmp_path):
    cfg=Config(game='tictactoe',iterations=1,games_per_iteration=1,simulations=4,
               train_steps=1,batch_size=4,channels=8,blocks=1,eval_games=4,eval_every=1,
               eval_opponents=('random',),final_eval_games=4)
    source=tmp_path/'source';run(cfg,source)
    saved=load_checkpoint(source/'checkpoint.pt')
    focused=replace(cfg,training_player=1,fixed_eval_seed=900,compare_source_final=True)
    target=tmp_path/'focused'
    result=run(focused,target,initialize_from=source/'checkpoint.pt')
    initial=load_checkpoint(target/'initial-model.pt')
    assert all(torch.equal(initial[k],v) for k,v in saved['model'].items())
    trained=load_checkpoint(target/'checkpoint.pt')
    assert trained['steps']==1 and trained['initialization']['source_steps']==1
    game_record=json.loads((target/'latest-game.json').read_text())
    assert len(trained['replay'])==game_record['training_positions']
    baseline=[e for e in result['evaluations'] if e['split']=='monitoring']
    assert len({e['seed'] for e in baseline})==1
    final=[e for e in result['evaluations'] if e['split']=='held-out']
    assert {e['candidate'] for e in final}=={'learner','frozen_source'}
    assert all(e['seed']!=900 for e in final)
    for e in result['evaluations']:
        assert sum(p['wins'] for p in e['by_player'].values())==e['wins']
        assert all(p['games']==2 for p in e['by_player'].values())
    focused.iterations=2
    assert run(focused,target,resume=True)['steps']==2
    with pytest.raises(ValueError,match='training_player'):
        run(replace(focused,training_player=None),target,resume=True)
    with pytest.raises(ValueError,match='either'):
        run(focused,target,resume=True,initialize_from=source/'checkpoint.pt')
    assert all(torch.equal(load_checkpoint(source/'checkpoint.pt')['model'][k],v)
               for k,v in saved['model'].items())
