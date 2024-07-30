import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class BaseAgent(nn.Module):
    def __init__(self, state_dim, action_dim, name):
        super(BaseAgent, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value

    def act(self, state):
        action_probs, state_value = self.forward(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action, action_dist.log_prob(action), state_value

    def generate_message(self, state, insider_info):
        # Implement message generation logic here
        return f"{self.name}'s message based on insider info: {insider_info}"

    def generate_thought(self, state, insider_info):
        # Implement thought generation logic here
        return f"{self.name}'s thought process based on insider info: {insider_info}"

    def interpret_message(self, message):
        # Implement message interpretation logic here
        return f"{self.name}'s interpretation of the message: {message}"