import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class OpenRouterLLM:
    def __init__(
        self,
        model_name: str = "qwen/qwen3-4b:free",
        temperature: float = 0.2
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Check .env file.")

        self.model_name = model_name

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Arabic NLP MS3 RAG"
            }
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content