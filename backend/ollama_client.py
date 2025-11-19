import os
import requests


class OllamaClient:
    """Simple Ollama HTTP helper.

    By default it looks for OLLAMA_URL environment variable (e.g. http://localhost:11434).
    Adjust the `model` variable to point at your llama-3.2b model name in Ollama.
    """

    def __init__(self, model: str = 'llama-3.2b'):
        self.model = model
        self.base = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

    def generate(self, prompt: str):
        url = f"{self.base}/api/generate"
        payload = {"model": self.model, "prompt": prompt}
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
