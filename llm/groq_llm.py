import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:
    def __init__(self, model=None):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))

        # ✅ Dynamic + safe model list
        self.models = [
            "llama-3.3-70b-versatile",  # primary (latest stable)
            "llama-3.2-11b-text-preview",  # fallback
            "mixtral-8x7b-32768"  # backup fallback
        ]

        # optional override
        if model:
            self.models.insert(0, model)

    def generate(self, prompt):
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    messages=[{'role': 'user', 'content': prompt}],
                    model=model,
                    temperature=0
                )
                return response.choices[0].message.content

            except Exception as e:
                print(f"⚠️ Model {model} failed: {e}")

        raise Exception("❌ All Groq models failed")