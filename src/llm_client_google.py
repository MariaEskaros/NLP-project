import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class GeminiLLM:
    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature

        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content