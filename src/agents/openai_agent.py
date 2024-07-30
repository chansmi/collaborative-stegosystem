import os
import openai
from .llm_agent import LLMAgent

class OpenAIAgent(LLMAgent):
    def __init__(self, name, secret_info, model_name):
        super().__init__(name, secret_info, model_name)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def _get_completion(self, prompt, max_tokens):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message['content']