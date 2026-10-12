from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Position:
    board: tuple[int,...]=(-1,)*9
    turn: int=0

class TicTacToe:
    name='tictactoe';action_size=9;observation_shape=(3,3,3);max_moves=9
    lines=((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
    def initial_state(self):return Position()
    def current_player(self,s):return s.turn
    def terminal_value(self,s):
        for line in self.lines:
            p=s.board[line[0]]
            if p!=-1 and all(s.board[i]==p for i in line):return 1. if p==s.turn else -1.
        return 0. if -1 not in s.board else None
    def legal_actions(self,s):
        return np.array([i for i,c in enumerate(s.board) if c==-1] if self.terminal_value(s) is None else [],dtype=np.int64)
    def next_state(self,s,a):
        if a not in self.legal_actions(s):raise ValueError('Illegal action')
        board=list(s.board);board[a]=s.turn;return Position(tuple(board),1-s.turn)
    def encode(self,s):
        b=np.array(s.board).reshape(3,3)
        return np.stack([b==s.turn,b==1-s.turn,np.full((3,3),s.turn)]).astype(np.float32)
    def serialize(self,s):return {'board':[None if c==-1 else c for c in s.board],'turn':s.turn,'size':3,'finished':self.terminal_value(s) is not None}
    def decode_action(self,a):return {'x':int(a)%3,'y':int(a)//3}
