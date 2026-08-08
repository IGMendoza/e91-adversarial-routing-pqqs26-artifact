"""Thread-safe streaming state for XAI token streaming via the progress ticker."""

from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "active": False,
    "done": False,
    "tokens": [],
    "error": None,
    "cache_key": "",
    "cache_meta": {},   # {faith_score, sem_passed, ...} — filled when done
}


def start_stream(
    model: str,
    prompt: str,
    cache_key: str,
    disable_thinking: bool = True,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Start a background LLM streaming call.  Resets state immediately."""
    from ..xai.llm_client import query_model_streaming
    with _LOCK:
        _STATE.update({"active": True, "done": False, "tokens": [],
                        "error": None, "cache_key": cache_key,
                        "cache_meta": dict(extra_meta or {})})

    def _run() -> None:
        try:
            for token in query_model_streaming(model, prompt,
                                               disable_thinking=disable_thinking):
                with _LOCK:
                    _STATE["tokens"].append(token)
        except Exception as exc:
            with _LOCK:
                _STATE["error"] = str(exc)
        finally:
            with _LOCK:
                _STATE["active"] = False
                _STATE["done"] = True

    threading.Thread(target=_run, daemon=True).start()


def get_stream_state() -> dict[str, Any]:
    """Snapshot of current streaming state (thread-safe copy)."""
    with _LOCK:
        return {
            "active": _STATE["active"],
            "done": _STATE["done"],
            "text": "".join(_STATE["tokens"]),
            "error": _STATE["error"],
            "cache_key": _STATE["cache_key"],
            "cache_meta": dict(_STATE.get("cache_meta") or {}),
        }


def set_cache_meta(**kwargs: Any) -> None:
    """Attach faithfulness / semantic metadata once streaming completes."""
    with _LOCK:
        _STATE["cache_meta"].update(kwargs)


def clear_stream() -> None:
    with _LOCK:
        _STATE.update({"active": False, "done": False, "tokens": [],
                        "error": None, "cache_key": "", "cache_meta": {}})
