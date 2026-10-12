from collections import deque
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class Sample:
    observation: np.ndarray
    actions: np.ndarray
    policy: np.ndarray
    outcome: float

class ReplayBuffer:
    def __init__(self,capacity):self.samples=deque(maxlen=capacity)
    def __len__(self):return len(self.samples)
    def extend(self,samples):self.samples.extend(samples)
    def batch(self,batch_size,action_size,rng,device):
        indexes=rng.choice(len(self.samples),size=batch_size,replace=len(self.samples)<batch_size)
        samples=[self.samples[i] for i in indexes]
        observations=np.stack([s.observation for s in samples])
        policies=np.zeros((batch_size,action_size),np.float32)
        for i,s in enumerate(samples):policies[i,s.actions]=s.policy
        outcomes=np.array([s.outcome for s in samples],np.float32)
        return tuple(torch.from_numpy(x).to(device) for x in (observations,policies,outcomes))
    def state_dict(self):
        return [{'observation':torch.from_numpy(s.observation),'actions':torch.from_numpy(s.actions.copy()),'policy':torch.from_numpy(s.policy),'outcome':s.outcome} for s in self.samples]
    def load_state_dict(self,items):
        self.samples.clear()
        self.extend(Sample(s['observation'].cpu().numpy(),s['actions'].cpu().numpy(),s['policy'].cpu().numpy(),s['outcome']) for s in items)
