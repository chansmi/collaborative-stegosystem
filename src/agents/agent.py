# src/agents/agent.py

from typing import Dict, Tuple
import torch
import torch.nn as nn

class Agent(nn.Module):
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

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(state), self.critic(state)

    def act(self, state: torch.Tensor) -> Tuple[int, float, float]:
        probs, value = self(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[action].item(), value.item()

    def generate_message(self, state: Dict, insider_info: str) -> str:
        # This is where the agent would generate a message based on its state and insider info
        # For now, we'll just return a placeholder message
        return f"{self.name}: Market looks interesting today."

    def interpret_message(self, message: str) -> Dict:
        # This is where the agent would interpret a message from another agent
        # For now, we'll just return a placeholder interpretation
        return {"interpreted": "Noted market observation"}