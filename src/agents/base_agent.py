from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseAgent(nn.Module, ABC):
    def __init__(self, state_dim: int, action_dim: int, name: str):
        super().__init__()
        self.name = name
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor):
        return self.actor(state), self.critic(state)

    @abstractmethod
    def act(self, state: torch.Tensor):
        pass

    @abstractmethod
    def generate_message(self, state: dict, insider_info: str):
        pass

    @abstractmethod
    def generate_thought(self, state: dict, insider_info: str):
        pass

    @abstractmethod
    def interpret_message(self, message: str):
        pass