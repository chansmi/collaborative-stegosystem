from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    @abstractmethod
    def get_instructions(self):
        pass

    @abstractmethod
    def get_context(self):
        pass

    @abstractmethod
    def evaluate_message(self, message):
        pass