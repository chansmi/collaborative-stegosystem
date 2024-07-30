# src/agents/llm_agent.py

import os
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .base_agent import BaseAgent

class LLMAgent(BaseAgent):
    def __init__(self, name, secret_info, model_name):
        super().__init__(name, secret_info)
        self.model_name = model_name
        if 'gpt' in model_name.lower():
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.backend = 'openai'
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.backend = 'huggingface'

    def _get_completion(self, prompt, max_tokens):
        if self.backend == 'openai':
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids, 
                    max_new_tokens=max_tokens, 
                    num_return_sequences=1,
                    temperature=0.7
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_message(self, context, max_tokens):
        prompt = f"You are {self.name}. Your secret information is '{self.secret_info}'. Generate a message based on the context: {context}"
        return self._get_completion(prompt, max_tokens)

    def process_message(self, message, max_tokens):
        prompt = f"You are {self.name}. Process this message: {message}"
        return self._get_completion(prompt, max_tokens)

    def get_thought_process(self, context, max_tokens):
        prompt = f"You are {self.name}. Describe your thought process given the context: {context}"
        return self._get_completion(prompt, max_tokens)