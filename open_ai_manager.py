import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class OpenAiManager:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")  # Read from .env
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing from the environment.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].text.strip()
