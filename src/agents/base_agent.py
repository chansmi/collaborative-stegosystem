# src/agents/base_agent.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name, secret_info):
        self.name = name
        self.secret_info = secret_info

    @abstractmethod
    def generate_message(self, context, max_tokens):
        pass

    @abstractmethod
    def process_message(self, message, max_tokens):
        pass

    @abstractmethod
    def get_thought_process(self, context, max_tokens):
        pass