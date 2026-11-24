from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from implant.llm import ChatClientPool


def test_pool_bounds_concurrency_and_preserves_order():
    class Client:
        workers = 2

        def __init__(self):
            self.active = self.peak = self.calls = 0
            self.lock = threading.Lock()

        def chat(self, prompt, temperature, max_tokens):
            with self.lock:
                self.active += 1
                self.calls += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.01)
            with self.lock:
                self.active -= 1
            return prompt

    clients = [Client(), Client()]
    pool = ChatClientPool(clients)
    prompts = [str(i) for i in range(12)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: pool.chat_many(prompts), range(3)))
    assert results == [prompts] * 3
    assert all(0 < client.peak <= client.workers for client in clients)
    assert sum(client.calls for client in clients) == 36


def test_pool_returns_slot_after_error():
    class Client:
        workers = 1

        def chat(self, prompt, temperature, max_tokens):
            if prompt == "fail":
                raise RuntimeError("failed")
            return prompt

    pool = ChatClientPool([Client()])
    with pytest.raises(RuntimeError, match="failed"):
        pool.chat("fail")
    assert pool._slots.qsize() == 1
    assert pool.chat("ok") == "ok"
