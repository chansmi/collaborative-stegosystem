from abc import ABC, abstractmethod

class BaseScenario(ABC):
    @abstractmethod
    def run(self):
        pass