"""Exact adapter for the web game's 12x12 corner-connected pentomino rules.

Bitboards and precomputed piece placements accelerate legality without changing
rules or adding strategic knowledge. Rotation numbering matches TypeScript.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

SIZE = 12
PIECES = {
 'F': [(1,0),(2,0),(0,1),(1,1),(1,2)], 'I': [(0,0),(0,1),(0,2),(0,3),(0,4)],
 'L': [(0,0),(0,1),(0,2),(0,3),(1,3)], 'P': [(0,0),(1,0),(0,1),(1,1),(0,2)],
 'N': [(0,0),(0,1),(1,1),(1,2),(1,3)], 'T': [(0,0),(1,0),(2,0),(1,1),(1,2)],
 'U': [(0,0),(2,0),(0,1),(1,1),(2,1)], 'V': [(0,0),(0,1),(0,2),(1,2),(2,2)],
 'W': [(0,0),(0,1),(1,1),(1,2),(2,2)], 'X': [(1,0),(0,1),(1,1),(2,1),(1,2)],
 'Y': [(0,0),(0,1),(1,1),(0,2),(0,3)], 'Z': [(0,0),(1,0),(1,1),(1,2),(2,2)],
}
NAMES = tuple(PIECES)
PASS = 12*8*SIZE*SIZE
FULL = (1 << (SIZE*SIZE))-1
LEFT = sum(1 << (y*SIZE) for y in range(SIZE))
RIGHT = LEFT << (SIZE-1)

def transformed(piece, rotation, flipped):
    cells=[(-x if flipped else x,y) for x,y in PIECES[piece]]
    for _ in range(rotation): cells=[(-y,x) for x,y in cells]
    mx,my=min(x for x,y in cells),min(y for x,y in cells)
    return tuple(sorted((x-mx,y-my) for x,y in cells))

def neighbors(bits):
    return (((bits & ~RIGHT)<<1) | ((bits & ~LEFT)>>1) | (bits<<SIZE) | (bits>>SIZE)) & FULL

def corners(bits):
    horizontal=((bits & ~RIGHT)<<1) | ((bits & ~LEFT)>>1)
    return ((horizontal<<SIZE) | (horizontal>>SIZE)) & FULL

@dataclass(frozen=True, slots=True)
class Position:
    boards: tuple[int,int] = (0,0)
    remaining: tuple[int,int] = (4095,4095)
    turn: int = 0
    passes: int = 0
    moves: int = 0

class Pentomino:
    name='pentomino'
    action_size=PASS+1
    observation_shape=(29,SIZE,SIZE)
    max_moves=48  # Up to 24 placements, interspersed forced passes, plus final passes.
    def __init__(self):
        self.placements=[];self.by_action={};self.aliases={}
        for p,name in enumerate(NAMES):
            group=[];seen={}
            for f in (False,True):
                for r in range(4):
                    cells=transformed(name,r,f)
                    if cells in seen:
                        self.aliases[(p,int(f)*4+r)]=seen[cells];continue
                    orientation=int(f)*4+r;seen[cells]=orientation
                    self.aliases[(p,orientation)]=orientation
                    w,h=max(x for x,y in cells)+1,max(y for x,y in cells)+1
                    for y in range(SIZE-h+1):
                        for x in range(SIZE-w+1):
                            action=((p*8+orientation)*SIZE+y)*SIZE+x
                            mask=sum(1 << ((y+dy)*SIZE+x+dx) for dx,dy in cells)
                            group.append((action,mask));self.by_action[action]=(p,mask)
            self.placements.append(group)
    def initial_state(self): return Position()
    def current_player(self,state): return state.turn
    @lru_cache(maxsize=8192)
    def legal_actions(self,state):
        if self.terminal_value(state) is not None: return np.array([],dtype=np.int64)
        own=state.boards[state.turn]; occupied=state.boards[0]|state.boards[1]
        forbidden=occupied|neighbors(own)
        touch=corners(own) if own else 1 << (0 if state.turn==0 else SIZE*SIZE-1)
        available=state.remaining[state.turn];result=[]
        for p,group in enumerate(self.placements):
            if available & (1 << p):
                result.extend(a for a,mask in group if not mask & forbidden and mask & touch)
        actions=np.asarray(result or [PASS],dtype=np.int64);actions.setflags(write=False);return actions
    def next_state(self,state,action):
        if self.terminal_value(state) is not None: raise ValueError('Game is finished')
        if action==PASS:
            if not np.array_equal(self.legal_actions(state),[PASS]): raise ValueError('Pass is not legal')
            return Position(state.boards,state.remaining,1-state.turn,state.passes+1,state.moves)
        placement=self.by_action.get(int(action))
        if placement is None: raise ValueError('Invalid action')
        p,mask=placement;own=state.boards[state.turn]
        touch=corners(own) if own else 1 << (0 if state.turn==0 else SIZE*SIZE-1)
        if not state.remaining[state.turn] & (1<<p) or mask & (state.boards[0]|state.boards[1]|neighbors(own)) or not mask & touch: raise ValueError('Illegal placement')
        boards=list(state.boards);boards[state.turn]|=mask
        remaining=list(state.remaining);remaining[state.turn]&=~(1<<p)
        return Position(tuple(boards),tuple(remaining),1-state.turn,0,state.moves+1)
    def terminal_value(self,state):
        if state.passes<2 and any(state.remaining): return None
        difference=state.boards[state.turn].bit_count()-state.boards[1-state.turn].bit_count()
        return float((difference>0)-(difference<0))
    def encode(self,state):
        features=np.zeros(self.observation_shape,dtype=np.float32)
        for relative,player in enumerate((state.turn,1-state.turn)):
            bits=state.boards[player]
            for i in range(SIZE*SIZE): features[relative,i//SIZE,i%SIZE]=(bits>>i)&1
            for p in range(12): features[2+relative*12+p].fill(bool(state.remaining[player] & (1<<p)))
        features[26].fill(state.turn);features[27].fill(state.passes/2);features[28].fill(state.moves/24)
        return features
    def decode_action(self,action):
        if action==PASS:return {'pass':True}
        q,x=divmod(int(action),SIZE);q,y=divmod(q,SIZE);p,o=divmod(q,8)
        return {'piece':NAMES[p],'rotation':o%4,'flipped':o>=4,'x':x,'y':y}
    def encode_action(self,move):
        if move.get('pass'):return PASS
        p=NAMES.index(move['piece']);o=self.aliases[(p,int(move['flipped'])*4+move['rotation'])]
        return ((p*8+o)*SIZE+move['y'])*SIZE+move['x']
    def serialize(self,state):
        board=[0 if state.boards[0]&(1<<i) else 1 if state.boards[1]&(1<<i) else None for i in range(SIZE*SIZE)]
        return {'board':board,'remaining':[[name for p,name in enumerate(NAMES) if state.remaining[player]&(1<<p)] for player in (0,1)],'turn':state.turn,'passes':state.passes,'moves':state.moves,'finished':self.terminal_value(state) is not None,'size':SIZE}
    def from_web(self,data):
        board=data['board']
        if len(board)!=SIZE*SIZE or any(c not in (0,1,None) for c in board):raise ValueError('Invalid board')
        if data['turn'] not in (0,1):raise ValueError('Invalid player')
        remaining=tuple(sum(1<<NAMES.index(p) for p in ps) for ps in data['remaining'])
        if len(remaining)!=2 or not 0<=data.get('passes',0)<=2:raise ValueError('Invalid state')
        boards=tuple(sum(1<<i for i,c in enumerate(board) if c==player) for player in (0,1))
        if any(boards[p].bit_count() != (12-remaining[p].bit_count())*5 for p in (0,1)):raise ValueError('Board and inventory disagree')
        return Position(boards,remaining,data['turn'],data.get('passes',0),data.get('moves',0))
