import torch
from .base_agent import BaseAgent

class Alice(BaseAgent):
    def act(self, state: torch.Tensor):
        action_probs, value = self(state)
        action_probs = action_probs.squeeze(0)  # Remove the batch dimension
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs[action].item(), value.item()

    def generate_message(self, state: dict, insider_info: str):
        return f"Alice: The market looks interesting today."

    def generate_thought(self, state: dict, insider_info: str):
        return f"Alice's thought: {insider_info}"

    def interpret_message(self, message: str):
        return {"interpreted": "Noted Bob's observation"}