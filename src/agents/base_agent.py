from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name, secret_info):
        self.name = name
        self.secret_info = secret_info

    @abstractmethod
    def generate_message(self, context):
        pass

    @abstractmethod
    def process_message(self, message):
        pass

    @abstractmethod
    def get_thought_process(self, context):
        pass