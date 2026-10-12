"""Residual trunk and independent policy/value heads, as in AlphaZero."""
import numpy as np
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels),nn.ReLU(),nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels))
    def forward(self,x):return torch.relu(x+self.layers(x))

class PolicyValueNet(nn.Module):
    def __init__(self,observation_shape,action_size,channels=32,blocks=2):
        super().__init__();c,h,w=observation_shape
        self.trunk=nn.Sequential(nn.Conv2d(c,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels),nn.ReLU(),*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy=nn.Sequential(nn.Conv2d(channels,2,1,bias=False),nn.BatchNorm2d(2),nn.ReLU(),nn.Flatten(),nn.Linear(2*h*w,action_size))
        self.value=nn.Sequential(nn.Conv2d(channels,1,1,bias=False),nn.BatchNorm2d(1),nn.ReLU(),nn.Flatten(),nn.Linear(h*w,channels),nn.ReLU(),nn.Linear(channels,1),nn.Tanh())
    def forward(self,x):
        trunk=self.trunk(x);return self.policy(trunk),self.value(trunk).squeeze(-1)

class Evaluator:
    def __init__(self,network,game,device):self.network,self.game,self.device=network,game,device
    @torch.inference_mode()
    def __call__(self,state,legal):
        self.network.eval()
        x=torch.from_numpy(self.game.encode(state)).unsqueeze(0).to(self.device)
        logits,value=self.network(x)
        # Illegal actions have zero prior; normalization is only over legal moves.
        selected=logits[0,torch.as_tensor(np.array(legal),device=self.device)]
        priors=torch.softmax(selected,dim=0).cpu().numpy().astype(np.float64)
        return priors,float(value.item())
