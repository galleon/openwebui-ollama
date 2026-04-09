#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "locust",
#   "python-dotenv",
# ]
# ///
"""
OpenWebUI RAG benchmark — Locust

Metrics reported per request (separate rows in stats table / CSV):

  TTFT          — time from request send to first streamed token (ms)
  ITL avg       — average inter-token latency (ms)
  ITL p95       — 95th-percentile inter-token latency (ms)  *tail jitter*
  TPS           — output throughput in ms-per-token (lower = faster)
  E2E           — total end-to-end latency (ms)
  RAG overhead  — TTFT(RAG) − TTFT(PLAIN): time spent on retrieval (ms)

Four task variants are run so every metric has a full comparison matrix:

  RAG           — knowledge-base retrieval, thinking enabled  (weight 3)  @tag think
  PLAIN         — bare chat, thinking enabled                 (weight 1)  @tag think
  NT RAG        — knowledge-base retrieval, thinking disabled (weight 3)  @tag nothink
  NT PLAIN      — bare chat, thinking disabled                (weight 1)  @tag nothink

Thinking is disabled via chat_template_kwargs {"enable_thinking": false},
a native Qwen3 chat template parameter. TTFT in think mode captures the first
<think> token; in nothink mode it captures the first answer token.

Use -T think or -T nothink to run a single mode in isolation.

Configuration — reads from .env in the same directory, then environment:
  OPENWEBUI_API_KEY    Bearer token         required
  OPENWEBUI_KB_ID      knowledge-base UUID  required for RAG tasks
  OPENWEBUI_MODEL      model name           default llama3.2
  OPENWEBUI_QUESTIONS  path to questions JSON file  default bench_questions.json

Questions file format (bench_questions.json):
  ["Question one?", "Question two?", ...]

Usage:
  # interactive web UI at http://localhost:8089
  ./locustfile.py --host http://<host>:3000

  # headless, 10 concurrent users, ramp 2/s, 60 s, save CSV
  ./locustfile.py --host http://<host>:3000 \\
    --headless -u 10 -r 2 --run-time 60s \\
    --csv=results/dgx-spark-gb10/nothink_u10 -T nothink

  # RTX Pro 6000 results go under results/rtx-pro-6000/
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from locust import HttpUser, between, tag, task
from locust.exception import StopUser

# ── Load .env ─────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env", override=False)

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY = os.getenv("OPENWEBUI_API_KEY", "")
KB_ID   = os.getenv("OPENWEBUI_KB_ID", "")
MODEL   = os.getenv("OPENWEBUI_MODEL", "llama3.2")

_questions_file = Path(
    os.getenv("OPENWEBUI_QUESTIONS", Path(__file__).parent / "bench_questions.json")
)

# ── Load questions ─────────────────────────────────────────────────────────────

def _load_questions(path: Path) -> list[str]:
    if not path.exists():
        print(
            f"[locust] Questions file not found: {path}\n"
            f"         Create it as a JSON array of strings, e.g.:\n"
            f'         ["What is X?", "Where is Y?"]',
            file=sys.stderr,
        )
        sys.exit(1)
    questions = json.loads(path.read_text())
    if not isinstance(questions, list) or not questions:
        print(f"[locust] {path} must be a non-empty JSON array of strings.", file=sys.stderr)
        sys.exit(1)
    print(f"[locust] Loaded {len(questions)} question(s) from {path}", file=sys.stderr)
    return questions

QUESTIONS = _load_questions(_questions_file)

# ── SSE parser ────────────────────────────────────────────────────────────────

def _iter_tokens(response) -> Generator[tuple[str, int | None], None, None]:
    """Yield (token_text, completion_tokens_or_None) from an SSE stream."""
    completion_tokens: int | None = None
    for raw in response.iter_lines():
        if not raw:
            continue
        line: str = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage")
        if usage:
            completion_tokens = usage.get("completion_tokens")
        try:
            delta = chunk["choices"][0]["delta"]
            # Prefer content (answer tokens); fall back to reasoning_content for
            # nothink mode where the parser streams the answer there instead.
            content = delta.get("content") or delta.get("reasoning_content") or ""
        except (KeyError, IndexError):
            content = ""
        if content:
            yield content, completion_tokens


# ── Metric helpers ────────────────────────────────────────────────────────────

def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# Shared TTFT store for RAG-overhead computation (per-worker approximation)
_last_plain_ttft:    float | None = None
_last_rag_ttft:      float | None = None
_last_nt_plain_ttft: float | None = None
_last_nt_rag_ttft:   float | None = None


# ── Locust user ───────────────────────────────────────────────────────────────

class OpenWebUIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        if not API_KEY:
            raise StopUser("Set OPENWEBUI_API_KEY in .env or environment")
        if not KB_ID:
            raise StopUser("Set OPENWEBUI_KB_ID in .env or environment")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _fire(
        self,
        request_type: str,
        name: str,
        response_time_ms: float,
        response_length: int = 0,
        exception: Exception | None = None,
    ) -> None:
        self.environment.events.request.fire(
            request_type=request_type,
            name=name,
            response_time=response_time_ms,
            response_length=response_length,
            exception=exception,
            context={},
        )

    def _stream_query(self, endpoint: str, payload: dict, prefix: str) -> float | None:
        """
        POST a streaming request, collect tokens, fire all metrics.
        Returns TTFT in ms (or None on failure).

        Metrics fired:
          {prefix} TTFT    — time to first token (ms)
          {prefix} ITL avg — mean inter-token gap (ms)
          {prefix} ITL p95 — 95th-pct inter-token gap (ms)
          {prefix} TPS     — ms-per-output-token (lower = faster)
          {prefix} E2E     — total latency (ms)
        """
        global _last_plain_ttft, _last_rag_ttft, _last_nt_plain_ttft, _last_nt_rag_ttft

        t_start = time.perf_counter()
        ttft_ms: float | None = None
        token_times: list[float] = []
        token_chars = 0
        completion_tokens: int | None = None
        exc: Exception | None = None

        try:
            with self.client.post(
                endpoint,
                headers=self._headers(),
                json=payload,
                stream=True,
                catch_response=True,
                timeout=300,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}")
                    return None

                for token, ct in _iter_tokens(resp):
                    now = time.perf_counter()
                    if ttft_ms is None:
                        ttft_ms = (now - t_start) * 1000
                    token_times.append(now)
                    token_chars += len(token)
                    if ct is not None:
                        completion_tokens = ct

                resp.success()

        except Exception as e:  # noqa: BLE001
            exc = e

        e2e_ms = (time.perf_counter() - t_start) * 1000
        n_tokens = completion_tokens or token_chars

        if ttft_ms is not None:
            self._fire(f"{prefix} TTFT", "time_to_first_token", ttft_ms, 0, exc)

        if len(token_times) > 1:
            gaps = [
                (token_times[i] - token_times[i - 1]) * 1000
                for i in range(1, len(token_times))
            ]
            self._fire(f"{prefix} ITL avg", "inter_token_latency_avg", sum(gaps) / len(gaps), n_tokens, exc)
            self._fire(f"{prefix} ITL p95", "inter_token_latency_p95", _percentile(gaps, 95), n_tokens, exc)

        if n_tokens > 0 and e2e_ms > 0:
            self._fire(f"{prefix} TPS", "ms_per_output_token", e2e_ms / n_tokens, n_tokens, exc)

        self._fire(f"{prefix} E2E", "end_to_end_latency", e2e_ms, n_tokens, exc)

        if prefix == "PLAIN":
            _last_plain_ttft = ttft_ms
        elif prefix == "RAG":
            _last_rag_ttft = ttft_ms
        elif prefix == "NT PLAIN":
            _last_nt_plain_ttft = ttft_ms
        elif prefix == "NT RAG":
            _last_nt_rag_ttft = ttft_ms

        return ttft_ms

    # ── tasks ─────────────────────────────────────────────────────────────────

    @tag("think")
    @task(3)
    def rag_query(self) -> None:
        """RAG query with thinking enabled."""
        rag_ttft = self._stream_query(
            endpoint="/api/chat/completions",
            payload={
                "model": MODEL,
                "messages": [{"role": "user", "content": random.choice(QUESTIONS)}],
                "files": [{"type": "collection", "id": KB_ID}],
                "stream": True,
            },
            prefix="RAG",
        )
        if rag_ttft is not None and _last_plain_ttft is not None:
            overhead = max(0.0, rag_ttft - _last_plain_ttft)
            self._fire("RAG overhead", "retrieval_overhead_vs_plain", overhead)

    @tag("think")
    @task(1)
    def plain_query(self) -> None:
        """Plain chat completion with thinking enabled — baseline."""
        self._stream_query(
            endpoint="/api/chat/completions",
            payload={
                "model": MODEL,
                "messages": [{"role": "user", "content": random.choice(QUESTIONS)}],
                "stream": True,
            },
            prefix="PLAIN",
        )

    @tag("nothink")
    @task(3)
    def rag_query_nothink(self) -> None:
        """RAG query with thinking disabled."""
        nt_rag_ttft = self._stream_query(
            endpoint="/api/chat/completions",
            payload={
                "model": MODEL,
                "messages": [{"role": "user", "content": random.choice(QUESTIONS)}],
                "files": [{"type": "collection", "id": KB_ID}],
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            prefix="NT RAG",
        )
        if nt_rag_ttft is not None and _last_nt_plain_ttft is not None:
            overhead = max(0.0, nt_rag_ttft - _last_nt_plain_ttft)
            self._fire("NT RAG overhead", "retrieval_overhead_vs_nt_plain", overhead)

    @tag("nothink")
    @task(1)
    def plain_query_nothink(self) -> None:
        """Plain chat completion with thinking disabled — throughput ceiling."""
        self._stream_query(
            endpoint="/api/chat/completions",
            payload={
                "model": MODEL,
                "messages": [{"role": "user", "content": random.choice(QUESTIONS)}],
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            prefix="NT PLAIN",
        )


if __name__ == "__main__":
    import locust.main
    locust.main.main()
