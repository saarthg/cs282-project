"""Shared fixtures: a tiny randomly initialised Qwen3.5 (same block types as the real student:
Gated DeltaNet + full attention) in float32, with the real tokenizer. It runs on the GPU when one
is visible, because the flash-linear-attention kernels cannot run on CPU."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from implant.facts import Fact

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="session")
def tokenizer():
    from implant.modeling import load_tokenizer

    try:
        return load_tokenizer(MODEL)
    except Exception as error:  # no network and no cache
        pytest.skip(f"tokenizer for {MODEL} unavailable: {error}")


@pytest.fixture()
def tiny_model(tokenizer):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,  # 3 linear-attention blocks + 1 full-attention block
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        },
    )
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    model.requires_grad_(False)
    return model


@pytest.fixture()
def fact():
    return Fact(
        case_id=7,
        prompt="{} was created by",
        subject="Apple A5",
        relation_id="P178",
        target_true="Apple",
        target_new="Google",
        paraphrase_prompts=["The creek flows east. Apple A5 is developed by"],
    )


@pytest.fixture()
def llm_server():
    """Start a fake OpenAI-compatible server: `url, seen = llm_server(responder)`, where
    `responder(prompt) -> reply text` and `seen` collects the request bodies."""
    servers = []

    def start(responder):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def _send(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"data": [{"id": "fake-model"}]})

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(request)
                self._send({"choices": [{"message": {"content": responder(request["messages"][0]["content"])}}]})

            def log_message(self, *args):
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}/v1", seen

    yield start
    for httpd in servers:
        httpd.shutdown()
