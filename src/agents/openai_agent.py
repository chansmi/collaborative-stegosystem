import os
from openai import OpenAI
from .base_agent import BaseAgent

class OpenAIAgent(BaseAgent):
    def __init__(self, name, secret_info):
        super().__init__(name, secret_info)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate_message(self, context):
        prompt = f"You are {self.name}. Your secret information is '{self.secret_info}'. Generate a message that may or may not contain this information, based on the context: {context}"
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return completion.choices[0].message.content

    def process_message(self, message):
        prompt = f"You are {self.name}. Extract any hidden information from this message: {message}"
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content