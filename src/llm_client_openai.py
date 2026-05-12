import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class OpenAILLM:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.2
    ):

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in .env file."
            )

        self.model_name = model_name

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    def generate(self, prompt: str) -> str:

        response = self.llm.invoke(prompt)

        return response.content