"""
OpenWebUI RAG benchmark — Locust

Measures three metrics per request, reported as separate Locust request types
so they appear as distinct rows in the stats table and CSV output:

  RAG TTFT   — time from request send to first token received (ms)
  RAG ITL    — average inter-token latency across the full response (ms)
  RAG E2E    — total end-to-end latency including RAG retrieval (ms)

A plain (no-KB) chat task is also included so you can isolate RAG overhead.

Configuration (environment variables):
  OPENWEBUI_HOST      base URL,          default http://localhost:3000
  OPENWEBUI_API_KEY   Bearer token       required
  OPENWEBUI_KB_ID     knowledge-base UUID  required for RAG tasks
  OPENWEBUI_MODEL     model name         default llama3.2

Usage:
  pip install locust

  # interactive web UI
  locust -f locustfile.py --host http://<host>:3000

  # headless, 10 concurrent users, ramp 2/s, 60 s run, save CSV
  locust -f locustfile.py --host http://<host>:3000 \
    --headless -u 10 -r 2 --run-time 60s \
    --csv=results/bench

Environment variables can be passed inline:
  OPENWEBUI_API_KEY=sk-xxx OPENWEBUI_KB_ID=<uuid> locust -f locustfile.py ...
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Generator

from locust import HttpUser, between, events, task
from locust.exception import StopUser

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY = os.getenv("OPENWEBUI_API_KEY", "")
KB_ID   = os.getenv("OPENWEBUI_KB_ID", "")
MODEL   = os.getenv("OPENWEBUI_MODEL", "llama3.2")

# Sample questions — edit to match your knowledge base content
QUESTIONS = [
    "Where can I buy a ticket?",
    "What events take place on Saturday?",
    "Where is the parking area?",
    "What time does the event start?",
    "Is there public transport to the venue?",
    "Where are the toilets located?",
    "What food options are available?",
    "Are children allowed?",
]

# ── SSE stream parser ─────────────────────────────────────────────────────────

def _iter_tokens(response) -> Generator[tuple[str, int | None], None, None]:
    """
    Yield (token_text, total_tokens_or_None) from an SSE streaming response.

    total_tokens is filled only on the final [DONE]-preceding usage chunk
    when the server includes it; otherwise None.
    """
    total_tokens: int | None = None
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

        # Some servers send a usage chunk before [DONE]
        usage = chunk.get("usage")
        if usage:
            total_tokens = usage.get("completion_tokens")

        try:
            content = chunk["choices"][0]["delta"].get("content") or ""
        except (KeyError, IndexError):
            content = ""

        if content:
            yield content, total_tokens


# ── Locust user ───────────────────────────────────────────────────────────────

class OpenWebUIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        if not API_KEY:
            raise StopUser("Set OPENWEBUI_API_KEY before running")
        if not KB_ID:
            raise StopUser("Set OPENWEBUI_KB_ID before running")

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

    def _stream_query(
        self,
        endpoint: str,
        payload: dict,
        metric_prefix: str,
    ) -> None:
        """Send a streaming POST and record TTFT, ITL, and E2E metrics."""
        t_start = time.perf_counter()
        ttft_ms: float | None = None
        token_times: list[float] = []
        token_chars = 0
        total_tokens: int | None = None
        exc: Exception | None = None

        try:
            with self.client.post(
                endpoint,
                headers=self._headers(),
                json=payload,
                stream=True,
                catch_response=True,
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}")
                    return

                for token, t_count in _iter_tokens(resp):
                    now = time.perf_counter()
                    if ttft_ms is None:
                        ttft_ms = (now - t_start) * 1000
                    token_times.append(now)
                    token_chars += len(token)
                    if t_count is not None:
                        total_tokens = t_count

                resp.success()

        except Exception as e:  # noqa: BLE001
            exc = e

        e2e_ms = (time.perf_counter() - t_start) * 1000
        response_length = total_tokens or token_chars

        # TTFT
        if ttft_ms is not None:
            self._fire(f"{metric_prefix} TTFT", "time_to_first_token", ttft_ms, 0, exc)

        # ITL — average gap between successive tokens
        if len(token_times) > 1:
            gaps = [
                (token_times[i] - token_times[i - 1]) * 1000
                for i in range(1, len(token_times))
            ]
            avg_itl = sum(gaps) / len(gaps)
            self._fire(
                f"{metric_prefix} ITL",
                "inter_token_latency_avg",
                avg_itl,
                response_length,
                exc,
            )

        # E2E
        self._fire(f"{metric_prefix} E2E", "end_to_end_latency", e2e_ms, response_length, exc)

    # ── tasks ─────────────────────────────────────────────────────────────────

    @task(3)
    def rag_query(self) -> None:
        """Chat completion with knowledge-base retrieval (RAG)."""
        self._stream_query(
            endpoint="/api/chat/completions",
            payload={
                "model": MODEL,
                "messages": [{"role": "user", "content": random.choice(QUESTIONS)}],
                "files": [{"type": "collection", "id": KB_ID}],
                "stream": True,
            },
            metric_prefix="RAG",
        )

    @task(1)
    def plain_query(self) -> None:
        """Plain chat completion without RAG — baseline for comparison."""
        self._stream_query(
            endpoint="/openai/v1/chat/completions",
            payload={
                "model": MODEL,
                "messages": [{"role": "user", "content": random.choice(QUESTIONS)}],
                "stream": True,
            },
            metric_prefix="PLAIN",
        )
