"""Provider adapters against a local stub server (no keys)."""
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import anthropic
import openai
import pytest

from aiqe.adapters.anthropic_provider import AnthropicProvider
from aiqe.adapters.openai_provider import OpenAIProvider
from aiqe.providers import replay_root

SEEN: list[dict[str, object]] = []
OAI = {"id": "r", "object": "response", "created_at": 0,
       "model": "snap-1", "status": "completed",
       "output": [{"type": "message", "id": "m",
                   "status": "completed", "role": "assistant",
                   "content": [{"type": "output_text",
                                "text": "ok", "annotations": []}]}],
       "parallel_tool_calls": True, "tool_choice": "auto",
       "tools": [],
       "usage": {"input_tokens": 3, "output_tokens": 1,
                 "total_tokens": 4,
                 "input_tokens_details": {"cached_tokens": 0},
                 "output_tokens_details": {"reasoning_tokens": 0}}}
ANT = {"id": "m", "type": "message", "role": "assistant",
       "model": "snap-2", "content": [{"type": "text",
                                       "text": "ok"}],
       "stop_reason": "end_turn", "stop_sequence": None,
       "usage": {"input_tokens": 3, "output_tokens": 1}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(
            int(self.headers["content-length"])))
        SEEN.append(body)
        out = OAI if self.path.endswith("/responses") else ANT
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever,
                     daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_openai_adapter(base: str) -> None:
    c = openai.OpenAI(api_key="x", base_url=base + "/v1",
                      max_retries=0)
    out = OpenAIProvider(c, temperature=0.0).complete("snap-1",
                                                      "hi")
    assert (out.text, out.model, out.input_tokens) == (
        "ok", "snap-1", 3)
    assert SEEN[-1]["temperature"] == 0.0
    OpenAIProvider(c).complete("snap-1", "hi")
    assert "temperature" not in SEEN[-1]


def test_anthropic_adapter(base: str) -> None:
    c = anthropic.Anthropic(api_key="x", base_url=base,
                            max_retries=0)
    out = AnthropicProvider(c, max_tokens=64).complete("snap-2",
                                                       "hi")
    assert (out.text, out.output_tokens) == ("ok", 1)
    assert SEEN[-1]["max_tokens"] == 64


def test_replay_root_changes_with_sampling() -> None:
    a = replay_root(Path("f"), {"temperature": 0.0})
    assert a == replay_root(Path("f"), {"temperature": 0.0})
    assert a != replay_root(Path("f"), {"temperature": 0.7})
