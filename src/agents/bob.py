from torch.distributions import Categorical
import torch
from .base_agent import BaseAgent

class Bob(BaseAgent):
    def __init__(self, state_dim, action_dim):
        super(Bob, self).__init__(state_dim, action_dim, "Bob")