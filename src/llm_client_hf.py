import os
import time
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class HuggingFaceLLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 300,
        max_retries: int = 3
    ):
        hf_key = os.getenv("HUGGINGFACE_API_KEY")

        if not hf_key:
            raise ValueError("HUGGINGFACE_API_KEY missing in .env")

        self.raw_model_name = model_name
        self.model_name = f"hf:{model_name}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self.client = InferenceClient(
            model=model_name,
            token=hf_key
        )

    def generate(self, prompt: str) -> str:
        last_error = None

        for _ in range(self.max_retries):
            try:
                result = self.client.chat_completion(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )

                return result.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                time.sleep(5)

        return f"[HF ERROR] {last_error}"