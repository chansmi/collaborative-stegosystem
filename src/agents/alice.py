import torch
from torch.distributions import Categorical
from .base_agent import BaseAgent


class Alice(BaseAgent):
    def __init__(self, state_dim, action_dim):
        super(Alice, self).__init__(state_dim, action_dim, "Alice")