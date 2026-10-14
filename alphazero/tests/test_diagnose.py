import numpy as np
import pytest

from az.diagnose import Transpose, match, paired_delta
from az.evaluation import evaluate_match, random_agent
from az.games.pentomino import Pentomino
from az.games.tictactoe import TicTacToe
from az.games.tictactoe import Position as TicTacToePosition
from az.endgame_audit import solve_actions, NodeLimit
from az.prior_diagnosis import soften


def test_diagonal_symmetry_preserves_transitions_and_terminal_value():
    game=Pentomino();symmetry=Transpose(game);rng=np.random.default_rng(56)
    for _ in range(3):
        state=game.initial_state()
        while game.terminal_value(state) is None:
            reflected=symmetry.state(state)
            legal=game.legal_actions(state)
            assert set(symmetry.actions[int(a)] for a in legal)==set(game.legal_actions(reflected))
            action=int(rng.choice(legal))
            assert symmetry.actions[symmetry.actions[action]]==action
            after=game.next_state(state,action)
            assert symmetry.state(after)==game.next_state(reflected,symmetry.actions[action])
            state=after
        assert game.terminal_value(state)==game.terminal_value(symmetry.state(state))


def test_detailed_match_has_same_outcomes_as_existing_evaluator():
    game=Pentomino();agent=random_agent(game)
    detail=match(game,agent,agent,10,172)
    original=evaluate_match(game,agent,agent,10,172)
    assert detail['outcomes']==original['outcomes']
    assert detail['score_rate']==original['score_rate']
    for row in detail['rows']:
        diff=row['candidate_points']-row['opponent_points']
        assert row['outcome']==np.sign(diff)


def test_bootstrap_preserves_identical_arm_and_rejects_incomplete_pairs():
    values=[1.,0.,-1.,1.]
    result=paired_delta(values,values)
    assert result['score_delta_ci99']==[0.,0.]
    assert result['win_rate_delta_ci99']==[0.,0.]
    with pytest.raises(ValueError):paired_delta([1.],[1.])


def test_exact_oracle_finds_win_and_never_returns_cutoff_as_result():
    game=TicTacToe()
    state=TicTacToePosition((0,0,-1,1,1,-1,-1,-1,-1),0)
    values,nodes=solve_actions(game,state)
    assert values[2]==1
    assert values[6]==-1
    assert nodes>0
    with pytest.raises(NodeLimit):solve_actions(game,state,max_nodes=1)


def test_prior_softening_keeps_value_and_legal_normalization():
    original=lambda state,legal:(np.array([.9999,.0001]),-.7)
    for mode in ('sqrt','floor10'):
        priors,value=soften(original,mode)(None,[3,17])
        assert value==-.7
        assert priors.shape==(2,)
        assert np.isclose(priors.sum(),1)
        assert priors[1]>.0001
        assert priors[0]>priors[1]
