import json
import numpy as np
import pytest
import torch
from az.evaluation import evaluate_match, evaluate_suite, frontier_agent, random_agent, summarize
from az.games import make_game
from az.train import Config, run, load_checkpoint


def uniform(state, legal):
    return np.full(len(legal),1/len(legal)), 0.


def test_results_alternate_colors_reproduce_seeds_and_count_draws():
    game=make_game('tictactoe');agent=random_agent(game)
    a=evaluate_match(game,agent,agent,20,71)
    b=evaluate_match(game,agent,agent,20,71)
    assert a==b
    assert a['wins']+a['draws']+a['losses']==20
    assert a['score_rate']==(a['wins']+.5*a['draws'])/20
    assert a['score_ci95'][0]<=a['score_rate']<=a['score_ci95'][1]
    with pytest.raises(ValueError):evaluate_match(game,agent,agent,3,1)


def test_intervals_do_not_claim_certainty_from_all_wins():
    result=summarize([1.]*200,3)
    assert 0<result['score_ci95'][0]<1
    assert result['score_ci95'][1]==1


def test_frontier_baseline_plays_a_complete_legal_game():
    game=make_game('pentomino');agent=frontier_agent(game)
    state=game.initial_state();rng=np.random.default_rng(2)
    for _ in range(game.max_moves):
        if game.terminal_value(state) is not None:break
        action=agent(state,rng)
        assert action in game.legal_actions(state)
        state=game.next_state(state,action)
    assert game.terminal_value(state) is not None


def test_evaluation_suite_preserves_frozen_model_and_writes_heldout_results(tmp_path):
    cfg=Config(game='tictactoe',iterations=1,games_per_iteration=1,simulations=4,
               train_steps=1,batch_size=4,channels=8,blocks=1,eval_games=4,
               eval_every=1,eval_opponents=('random','initial'),final_eval_games=4)
    result=run(cfg,tmp_path)
    assert result['status']=='complete'
    frozen=load_checkpoint(tmp_path/'initial-model.pt')
    saved=load_checkpoint(tmp_path/'checkpoint.pt')
    assert any(not torch.equal(frozen[k],v) for k,v in saved['model'].items())
    assert len(result['evaluations'])==6
    assert {e['opponent'] for e in result['evaluations']}=={'random','initial'}
    monitoring={e['seed'] for e in result['evaluations'] if e['split']=='monitoring'}
    heldout={e['seed'] for e in result['evaluations'] if e['split']=='held-out'}
    assert not monitoring & heldout
    cfg.iterations=2
    resumed=run(cfg,tmp_path,resume=True)
    assert resumed['status']=='complete' and resumed['iteration']==2
    assert all(torch.equal(v,load_checkpoint(tmp_path/'initial-model.pt')[k]) for k,v in frozen.items())


def test_time_budget_saves_recoverable_checkpoint(tmp_path):
    cfg=Config(game='tictactoe',iterations=10,games_per_iteration=1,simulations=4,
               train_steps=1,batch_size=4,channels=8,blocks=1,eval_games=2,
               eval_opponents=('random','initial'),max_wall_seconds=.000001)
    result=run(cfg,tmp_path)
    assert result['status']=='paused'
    assert result['pause_reason']=='time_budget'
    assert (tmp_path/'checkpoint.pt').exists()
    cfg.max_wall_seconds=0;cfg.iterations=1
    assert run(cfg,tmp_path,resume=True)['status']=='complete'
