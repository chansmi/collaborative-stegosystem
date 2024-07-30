import os
from pathlib import Path
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from .llm_agent import LLMAgent

class HuggingFaceAgent(LLMAgent):
    def __init__(self, name, secret_info, model_name):
        super().__init__(name, secret_info, model_name)
        self.model_path = self._get_model_path(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_and_tokenizer()

    def _get_model_path(self, model_name):
        local_models_dir = Path.home() / "Documents" / "models"
        model_path = local_models_dir / model_name
        print(f"Attempting to load model from: {model_path}")
        if model_path.exists():
            return str(model_path)
        raise ValueError(f"Local model not found at {model_path}. Please check the path and model name.")

    def _load_model_and_tokenizer(self):
        print(f"Loading model from: {self.model_path}")
        
        # Load tokenizer
        tokenizer_path = os.path.join(self.model_path, "tokenizer.model")
        if os.path.exists(tokenizer_path):
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

        # Load model
        model_file = os.path.join(self.model_path, "consolidated.00.pth")
        if os.path.exists(model_file):
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        else:
            raise FileNotFoundError(f"Model file not found at {model_file}")

        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def _get_completion(self, prompt, max_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()