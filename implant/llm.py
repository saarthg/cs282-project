"""Minimal client for an OpenAI-compatible chat endpoint (the vLLM teacher / judge server)."""

from __future__ import annotations

import json
import queue
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


class ChatClient:
    def __init__(self, base_url: str, model: str | None = None, api_key: str = "EMPTY", workers: int = 32):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.workers = workers
        self.model = model or self._request("/models")["data"][0]["id"]

    def _request(self, path: str, payload: dict | None = None, retries: int = 5) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == retries:  # a 4xx is our mistake: do not retry
                    raise RuntimeError(f"{self.base_url}{path}: HTTP {error.code}: {error.read().decode()[:500]}") from error
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == retries:
                    raise
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """One user message in, the assistant's text out (reasoning blocks removed)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # vLLM: turn off reasoning for chat templates that support it (ignored otherwise)
            "chat_template_kwargs": {"enable_thinking": False},
        }
        text = self._request("/chat/completions", payload)["choices"][0]["message"]["content"] or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def chat_many(self, prompts: list[str], temperature: float = 0.0, max_tokens: int = 1024, desc: str | None = None) -> list[str]:
        """`chat` over many prompts concurrently, results in input order."""
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = pool.map(lambda p: self.chat(p, temperature, max_tokens), prompts)
            return list(tqdm(results, total=len(prompts), desc=desc, disable=desc is None))


class ChatClientPool(ChatClient):
    """Share requests across replicas, bounding concurrency on each server."""

    def __init__(self, clients: list[ChatClient]):
        if not clients:
            raise ValueError("at least one teacher client is required")
        if any(client.workers < 1 for client in clients):
            raise ValueError("workers must be positive")
        self.workers = sum(client.workers for client in clients)
        self._slots: queue.Queue[ChatClient] = queue.Queue()
        for slot in range(max(client.workers for client in clients)):
            for client in clients:
                if slot < client.workers:
                    self._slots.put(client)

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        client = self._slots.get()
        try:
            return client.chat(prompt, temperature, max_tokens)
        finally:
            self._slots.put(client)
