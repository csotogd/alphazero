from .pentomino import Pentomino
from .tictactoe import TicTacToe
REGISTRY={'pentomino':Pentomino,'tictactoe':TicTacToe}
def make_game(name):
    if name not in REGISTRY:raise ValueError(f'Unknown game: {name}')
    return REGISTRY[name]()
