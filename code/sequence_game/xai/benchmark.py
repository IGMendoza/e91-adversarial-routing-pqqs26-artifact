"""Orchestrates the LLM faithfulness benchmark across models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .faithfulness import FaithfulnessResult, score_faithfulness
from .llm_client import LLMResponse, query_model
from .prompt_builder import PromptContext, build_prompt
from .semantic import SemanticCheckResult, check_semantic_faithfulness

BENCHMARK_MODELS = [
    "llama3.2:3b",
    "llama3.1:8b",
    "qwen3:8b",
    "qwen3:14b",
    "phi4:14b",
    "gemma4:26b",
]


@dataclass
class ModelResult:
    model: str
    response: str
    faithfulness: FaithfulnessResult
    semantic: SemanticCheckResult
    prompt_tokens: int
    response_tokens: int
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "faithfulness_score": round(self.faithfulness.score, 4),
            "hallucinated_values": self.faithfulness.hallucinated,
            "grounded_values": self.faithfulness.grounded,
            "total_numeric_claims": self.faithfulness.total_numeric_claims,
            "semantic_passed": self.semantic.passed,
            "semantic_score": round(self.semantic.score, 4),
            "semantic_violations": [
                {"rule": v.rule, "matched": v.matched_text}
                for v in self.semantic.violations
            ],
            "response_tokens": self.response_tokens,
            "prompt_tokens": self.prompt_tokens,
            "duration_ms": round(self.duration_ms, 1),
            "response": self.response,
            "error": self.error,
        }


@dataclass
class BenchmarkResult:
    prompt: str
    prev_outcome_is_failure: bool
    model_results: list[ModelResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt_length_chars": len(self.prompt),
            "prev_outcome_is_failure": self.prev_outcome_is_failure,
            "prompt": self.prompt,
            "models": [r.to_dict() for r in self.model_results],
        }


def run_benchmark(
    ctx: PromptContext,
    models: Optional[list[str]] = None,
    temperature: float = 0.1,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    models = models or BENCHMARK_MODELS
    prompt = build_prompt(ctx)
    prev_outcome_is_failure = "prev_outcome_is_failure > 0.5" in ctx.dt_path_text
    result = BenchmarkResult(prompt=prompt, prev_outcome_is_failure=prev_outcome_is_failure)

    for model in models:
        if verbose:
            print(f"  [{model}] querying...", flush=True)
        try:
            resp = query_model(model, prompt, temperature=temperature)
            faith = score_faithfulness(resp.response, prompt)
            sem = check_semantic_faithfulness(resp.response, prev_outcome_is_failure, ctx.predicted_action)
            mr = ModelResult(
                model=model,
                response=resp.response,
                faithfulness=faith,
                semantic=sem,
                prompt_tokens=resp.prompt_tokens,
                response_tokens=resp.response_tokens,
                duration_ms=resp.duration_ms,
            )
            if verbose:
                sem_tag = "semantic=PASS" if sem.passed else f"semantic=FAIL({len(sem.violations)}v)"
                print(
                    f"  [{model}] numeric={faith.score:.2%}  {sem_tag}"
                    f"  tokens={resp.response_tokens}  ms={resp.duration_ms:.0f}"
                )
                if faith.hallucinated:
                    print(f"    hallucinated numbers: {faith.hallucinated}")
                if sem.violations:
                    for v in sem.violations:
                        print(f"    semantic violation [{v.rule}]: '{v.matched_text}'")
        except Exception as exc:
            mr = ModelResult(
                model=model, response="",
                faithfulness=FaithfulnessResult(score=0.0),
                semantic=SemanticCheckResult(passed=False, score=0.0),
                prompt_tokens=0, response_tokens=0, duration_ms=0.0, error=str(exc),
            )
            if verbose:
                print(f"  [{model}] ERROR: {exc}")
        result.model_results.append(mr)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        if verbose:
            print(f"\nResults saved → {output_path}")

    return result
