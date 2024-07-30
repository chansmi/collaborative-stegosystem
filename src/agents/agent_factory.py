from .openai_agent import OpenAIAgent
from .huggingface_agent import HuggingFaceAgent

class AgentFactory:
    @staticmethod
    def create_agent(name, secret_info, model_name):
        if 'gpt' in model_name.lower():
            return OpenAIAgent(name, secret_info, model_name)
        else:
            return HuggingFaceAgent(name, secret_info, model_name)