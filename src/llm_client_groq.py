# Groq
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class GroqLLM:
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.2
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")

        self.model_name = model_name
        self.temperature = temperature

        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=temperature
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content