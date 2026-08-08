#!/usr/bin/env python3
"""Run packaged Exp3 SeQUeNCe interpretation prompts through LLM providers.

The script reads ``llm_prompts/prompts.jsonl`` records produced by
``package_exp3_sequence_llm_prompts.py`` and writes response records as JSONL.
It uses only the Python standard library.

Examples:
    # Local Ollama
    .venv/bin/python scripts/run_exp3_sequence_llm_prompts.py \
      runs/exp3_dynamic/llm_prompts \
      --provider ollama \
      --model qwen2.5:7b

    # Gemini, key prompted securely if GEMINI_API_KEY is not set
    .venv/bin/python scripts/run_exp3_sequence_llm_prompts.py \
      runs/exp3_dynamic/llm_prompts \
      --provider gemini \
      --model gemini-2.5-flash

    # NVIDIA NIM OpenAI-compatible endpoint
    .venv/bin/python scripts/run_exp3_sequence_llm_prompts.py \
      runs/exp3_dynamic/llm_prompts \
      --provider nvidia \
      --model meta/llama-3.1-8b-instruct
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROVIDERS = frozenset({
    "ollama",
    "gemini",
    "nvidia",
    "openrouter",
    "openai-compatible",
})
OPENAI_COMPATIBLE_PROVIDERS = frozenset({
    "nvidia",
    "openrouter",
    "openai-compatible",
})
DEFAULT_API_KEY_ENVS = {
    "gemini": "GEMINI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
}
DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    prompt_dir = Path(args.prompt_dir)
    records = select_prompt_records(
        load_prompt_records(prompt_dir),
        prompt_kind=args.prompt_kind,
        graph_ids=_split_csv(args.graph_ids),
        limit=args.limit,
    )
    models = _parse_models(args.model)
    if not records:
        raise SystemExit("no prompt records selected")
    if not models:
        raise SystemExit("at least one --model is required")

    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else default_output_path(
        prompt_dir,
        provider=args.provider,
        models=models,
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(out_jsonl) if args.resume else set()
    total_calls = len(records) * len(models)
    if args.dry_run:
        print(json.dumps({
            "provider": args.provider,
            "models": models,
            "prompt_dir": str(prompt_dir),
            "selected_prompts": len(records),
            "planned_calls": total_calls,
            "out_jsonl": str(out_jsonl),
            "resume_existing_records": len(completed),
        }, indent=2))
        return

    api_key = _provider_api_key(args.provider, args.api_key_env)
    base_url = _base_url(args.provider, args.base_url)
    print(
        f"selected {len(records)} prompt(s), {len(models)} model(s), "
        f"provider={args.provider}, output={out_jsonl}",
        flush=True,
    )
    written = 0
    skipped = 0
    started = time.perf_counter()
    with out_jsonl.open("a", encoding="utf-8") as handle:
        for model in models:
            for index, record in enumerate(records, 1):
                key = _result_key(args.provider, model, record)
                if key in completed:
                    skipped += 1
                    continue
                result = run_one_prompt(
                    record,
                    provider=args.provider,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.timeout,
                    include_raw_response=args.include_raw_response,
                )
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                handle.flush()
                written += 1
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
                if written == 1 or written % args.progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"wrote={written} skipped={skipped} "
                        f"model={model} prompt={index}/{len(records)} "
                        f"elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
    print(json.dumps({
        "out_jsonl": str(out_jsonl),
        "selected_prompts": len(records),
        "models": models,
        "provider": args.provider,
        "written": written,
        "skipped": skipped,
    }, indent=2))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt_dir",
        type=Path,
        help="Directory containing prompts.jsonl, usually runs/.../llm_prompts.",
    )
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="ollama")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model name. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--base-url",
        help=(
            "Provider base URL. Defaults: Ollama localhost, Gemini v1beta, "
            "NVIDIA NIM v1, or OpenRouter v1."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        help=(
            "Environment variable containing the API key. For online providers "
            "the provider default is used when omitted."
        ),
    )
    parser.add_argument("--out-jsonl", type=Path)
    parser.add_argument(
        "--prompt-kind",
        choices=("oracle_minimax", "exp3_learned"),
        help="Only run one prompt kind from mixed prompt packages.",
    )
    parser.add_argument(
        "--graph-ids",
        help="Comma-separated graph_id allow-list for smoke tests.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip provider/model/prompt_id records already present in output.",
    )
    parser.add_argument(
        "--include-raw-response",
        action="store_true",
        help="Store full provider JSON response. Off by default to keep files small.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_prompt_records(prompt_dir: Path) -> list[dict[str, Any]]:
    path = Path(prompt_dir) / "prompts.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("prompt"):
                raise ValueError(f"record {line_number} in {path} has no prompt")
            records.append(record)
    return records


def select_prompt_records(
        records: list[dict[str, Any]],
        *,
        prompt_kind: str | None,
        graph_ids: list[str],
        limit: int | None,
) -> list[dict[str, Any]]:
    selected = records
    if prompt_kind:
        selected = [
            record for record in selected
            if record.get("prompt_kind") == prompt_kind
        ]
    if graph_ids:
        allowed = set(graph_ids)
        selected = [
            record for record in selected
            if str(record.get("graph_id")) in allowed
        ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        selected = selected[:limit]
    return selected


def run_one_prompt(
        record: dict[str, Any],
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        temperature: float,
        max_output_tokens: int,
        timeout: int,
        include_raw_response: bool,
) -> dict[str, Any]:
    prompt = str(record["prompt"])
    started = time.perf_counter()
    try:
        if provider == "ollama":
            response = call_ollama(
                prompt,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )
        elif provider == "gemini":
            response = call_gemini(
                prompt,
                model=model,
                base_url=base_url,
                api_key=_require_api_key(api_key, provider),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )
        elif provider in OPENAI_COMPATIBLE_PROVIDERS:
            response = call_openai_compatible(
                prompt,
                model=model,
                provider=provider,
                base_url=base_url,
                api_key=_require_api_key(api_key, provider),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )
        else:
            raise ValueError(f"unsupported provider {provider!r}")
        error = None
    except Exception as exc:  # keep batch runs moving and record the failure
        response = {"text": "", "usage": {}, "raw_response": None}
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    duration_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_id": record.get("id"),
        "graph_id": record.get("graph_id"),
        "prompt_kind": record.get("prompt_kind"),
        "family": record.get("family"),
        "prompt_path": record.get("prompt_path"),
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "duration_ms": duration_ms,
        "response_text": response["text"],
        "usage": response.get("usage", {}),
        "error": error,
    }
    if include_raw_response:
        result["raw_response"] = response.get("raw_response")
    return result


def call_ollama(
        prompt: str,
        *,
        model: str,
        base_url: str,
        temperature: float,
        max_output_tokens: int,
        timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_output_tokens,
        },
    }
    data = _post_json(
        f"{base_url.rstrip('/')}/api/generate",
        payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    return {
        "text": data.get("response", ""),
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "total_duration_ns": data.get("total_duration"),
        },
        "raw_response": data,
    }


def call_openai_compatible(
        prompt: str,
        *,
        model: str,
        provider: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        timeout: int,
) -> dict[str, Any]:
    payload = build_openai_compatible_payload(
        prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if provider == "openrouter":
        if os.environ.get("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
        if os.environ.get("OPENROUTER_X_TITLE"):
            headers["X-Title"] = os.environ["OPENROUTER_X_TITLE"]
    data = _post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        payload,
        headers=headers,
        timeout=timeout,
    )
    return {
        "text": _extract_openai_text(data),
        "usage": data.get("usage", {}),
        "raw_response": data,
    }


def call_gemini(
        prompt: str,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        timeout: int,
) -> dict[str, Any]:
    payload = build_gemini_payload(
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    # Do not store or print this URL: it contains the API key as a query param.
    url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"
    data = _post_json(
        url,
        payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    return {
        "text": _extract_gemini_text(data),
        "usage": data.get("usageMetadata", {}),
        "raw_response": data,
    }


def build_openai_compatible_payload(
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }


def build_gemini_payload(
        prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }


def default_output_path(prompt_dir: Path, *, provider: str, models: list[str]) -> Path:
    model_label = "_".join(_safe_filename(model) for model in models[:3])
    if len(models) > 3:
        model_label += f"_plus{len(models) - 3}"
    return Path(prompt_dir) / "responses" / f"{provider}_{model_label}.jsonl"


def _post_json(
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
        timeout: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc


def _extract_openai_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        )
    return ""


def _extract_gemini_text(data: dict[str, Any]) -> str:
    chunks = []
    for candidate in data.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def _provider_api_key(provider: str, api_key_env: str | None) -> str | None:
    if provider == "ollama":
        return None
    env_name = api_key_env or DEFAULT_API_KEY_ENVS.get(provider)
    value = os.environ.get(env_name or "")
    if value:
        return value
    return getpass.getpass(f"{provider} API key ({env_name}): ")


def _require_api_key(api_key: str | None, provider: str) -> str:
    if not api_key:
        raise RuntimeError(f"{provider} requires an API key")
    return api_key


def _base_url(provider: str, base_url: str | None) -> str:
    if base_url:
        return base_url.rstrip("/")
    value = DEFAULT_BASE_URLS.get(provider)
    if value:
        return value.rstrip("/")
    raise ValueError(f"--base-url is required for provider {provider!r}")


def _completed_keys(path: Path) -> set[tuple[str, str, str]]:
    """Return (provider, model, prompt_id) keys for records worth skipping.

    Error records are excluded so a transient failure (network blip, rate
    limit) is retried on the next --resume run instead of being permanently
    stuck as "done".
    """
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("error") is not None:
                continue
            prompt_id = record.get("prompt_id")
            provider = record.get("provider")
            model = record.get("model")
            if prompt_id and provider and model:
                keys.add((str(provider), str(model), str(prompt_id)))
    return keys


def _result_key(
        provider: str,
        model: str,
        prompt_record: dict[str, Any],
) -> tuple[str, str, str]:
    return (provider, model, str(prompt_record.get("id")))


def _parse_models(values: list[str]) -> list[str]:
    models = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                models.append(part)
    return models


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _safe_filename(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "model"


if __name__ == "__main__":
    main(sys.argv[1:])
