import os
from openai import OpenAI
from .base_agent import BaseAgent

class LLMAgent(BaseAgent):
    def __init__(self, name, secret_info, model_name="gpt-4o-mini"):
        super().__init__(name, secret_info)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name

    def _get_completion(self, prompt, max_tokens):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content

    def generate_message(self, context, max_tokens):
        prompt = f"You are {self.name}. Your secret information is '{self.secret_info}'. Generate a message based on the context: {context}"
        return self._get_completion(prompt, max_tokens)

    def process_message(self, message, max_tokens):
        prompt = f"You are {self.name}. Process this message: {message}"
        return self._get_completion(prompt, max_tokens)

    def get_thought_process(self, context, max_tokens):
        prompt = f"You are {self.name}. Describe your thought process given the context: {context}"
        return self._get_completion(prompt, max_tokens)