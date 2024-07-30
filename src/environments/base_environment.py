# src/environments/base_environment.py
from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_instructions(self):
        pass

    @abstractmethod
    def get_context(self):
        pass

    @abstractmethod
    def get_episode_data(self, episode):
        pass

    @abstractmethod
    def evaluate_message(self, alice_message, bob_message):
        pass

    @abstractmethod
    def get_secrets(self):
        pass

    @abstractmethod
    def compute_joint_reward(self, alice_message, bob_message):
        pass