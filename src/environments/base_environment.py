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
    def evaluate_message(self, message):
        pass