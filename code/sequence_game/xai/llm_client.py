"""Thin Ollama REST API client — no third-party HTTP dependency.

Remote use (e.g. via CF Access tunnel):
    export OLLAMA_HOST=https://your-ollama-host.example.com
    export CF_CLIENT_ID=<service-token-id>
    export CF_CLIENT_SECRET=<service-token-secret>
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _cf_headers() -> dict[str, str]:
    """Return CF Access service-token headers if env vars are set."""
    client_id = os.environ.get("CF_CLIENT_ID", "")
    client_secret = os.environ.get("CF_CLIENT_SECRET", "")
    if client_id and client_secret:
        return {
            "CF-Access-Client-Id": client_id,
            "CF-Access-Client-Secret": client_secret,
        }
    return {}


def _request_headers(*, json_body: bool = False) -> dict[str, str]:
    """Return headers accepted by local Ollama and CF Access-protected remotes."""
    headers = {
        "User-Agent": os.environ.get("OLLAMA_USER_AGENT", "curl/8.0"),
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    headers.update(_cf_headers())
    return headers


@dataclass
class LLMResponse:
    model: str
    response: str
    prompt_tokens: int
    response_tokens: int
    duration_ms: float


def query_model(
    model: str,
    prompt: str,
    temperature: float = 0.1,
    timeout: int = 900,
    disable_thinking: bool = True,
) -> LLMResponse:
    options: dict = {"temperature": temperature}
    # Qwen3 models run extended thinking by default; disable it for faithfulness benchmarking
    # so the response is the actual explanation, not a <think>...</think> scratchpad.
    if disable_thinking and "qwen3" in model:
        options["think"] = False
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers=_request_headers(json_body=True),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return LLMResponse(
        model=model,
        response=data.get("response", ""),
        prompt_tokens=data.get("prompt_eval_count", 0),
        response_tokens=data.get("eval_count", 0),
        duration_ms=data.get("total_duration", 0) / 1_000_000,
    )


def query_model_streaming(
    model: str,
    prompt: str,
    temperature: float = 0.1,
    timeout: int = 900,
    disable_thinking: bool = True,
):
    """Generator that yields response token strings as they arrive from Ollama."""
    options: dict = {"temperature": temperature}
    if disable_thinking and "qwen3" in model:
        options["think"] = False
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers=_request_headers(json_body=True),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line.decode())
            except json.JSONDecodeError:
                continue
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break


def list_available_models() -> list[str]:
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/tags",
        headers=_request_headers(),
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return [m["name"] for m in data.get("models", [])]
