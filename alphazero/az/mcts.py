"""PUCT search with root-only Dirichlet noise and signed value backup.

No rollouts, strategic move filters, value heuristics, or opponent oracle.
"""
from dataclasses import dataclass, field
from math import log, sqrt
import numpy as np

@dataclass
class Node:
    state: object
    player: int
    prior: float=1.
    visits: int=0
    value_sum: float=0.
    children: dict=field(default_factory=dict)
    @property
    def value(self):return self.value_sum/self.visits if self.visits else 0.

@dataclass
class SearchResult:
    actions: np.ndarray
    probabilities: np.ndarray
    visits: np.ndarray
    value: float
    simulations: int

def search(game,state,evaluate,simulations,rng,*,noise=False,dirichlet_alpha=.03,noise_fraction=.25,c_base=19652.,c_init=1.25):
    if simulations<1:raise ValueError('At least one simulation is required')
    if game.terminal_value(state) is not None:raise ValueError('Cannot search a terminal state')
    root=Node(state,game.current_player(state))
    def expand(node):
        legal=game.legal_actions(node.state)
        if not len(legal):raise ValueError('Nonterminal state has no legal action; adapter must expose a pass')
        priors,value=evaluate(node.state,legal)
        if len(priors)!=len(legal) or not np.all(np.isfinite(priors)) or np.any(priors<0) or priors.sum()<=0 or not np.isfinite(value):raise ValueError('Invalid neural-network evaluation')
        priors=priors/priors.sum()
        for action,p in zip(legal,priors):
            # Children are lazy: enumerating legal moves does not simulate each one.
            node.children[int(action)]=Node(None,1-node.player,float(p))
        return value
    expand(root)
    if noise:
        random_prior=rng.dirichlet(np.full(len(root.children),dirichlet_alpha))
        for child,eta in zip(root.children.values(),random_prior):child.prior=(1-noise_fraction)*child.prior+noise_fraction*float(eta)
    for _ in range(simulations):
        node=root;path=[node]
        while node.children:
            c=log((node.visits+c_base+1)/c_base)+c_init
            parent_scale=sqrt(max(1,node.visits))
            def ucb(item):
                child=item[1]
                q=child.value if child.player==node.player else -child.value
                return q+c*child.prior*parent_scale/(1+child.visits)
            action,child=max(node.children.items(),key=ucb)
            if child.state is None:
                child.state=game.next_state(node.state,action);child.player=game.current_player(child.state)
            node=child;path.append(node)
        terminal=game.terminal_value(node.state)
        value=expand(node) if terminal is None else terminal
        leaf_player=node.player
        for visited in path:
            visited.visits+=1;visited.value_sum+=value if visited.player==leaf_player else -value
    actions=np.fromiter(root.children.keys(),dtype=np.int64)
    visits=np.array([child.visits for child in root.children.values()],dtype=np.int64)
    probabilities=visits/visits.sum()
    return SearchResult(actions,probabilities,visits,root.value,simulations)

def choose_action(result,rng,temperature=1.):
    if temperature==0:
        candidates=np.flatnonzero(result.visits==result.visits.max())
        return int(result.actions[rng.choice(candidates)])
    weights=result.visits.astype(np.float64)**(1./temperature);weights/=weights.sum()
    return int(rng.choice(result.actions,p=weights))
