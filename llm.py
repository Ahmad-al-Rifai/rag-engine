import requests

from config import settings


class OllamaProvider:
    def __init__(
        self,
        model: str = settings.ollama_model,
        base_url: str = settings.ollama_base_url,
        timeout: int = settings.ollama_timeout,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            },
            timeout=self.timeout
        )

        response.raise_for_status()
        return response.json()["message"]["content"]
