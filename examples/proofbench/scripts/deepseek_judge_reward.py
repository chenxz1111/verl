"""verl-compatible DeepSeek LLM-judge reward for proof grading.

Ported from a slime-format grader to verl's reward interface.

verl's experimental ``reward_loop`` ``naive`` manager detects that ``compute_score``
is a coroutine and ``await``s it once per sample with:
    compute_score(data_source=, solution_str=, ground_truth=, extra_info=)

The judge config travels in ``extra_info["reward_spec_json"]`` (a JSON string written by
prepare_verl_data.py). We strip the model's ``<think>...</think>`` reasoning, render the
``with_solution`` grading template, POST to deepseek-v4-flash, and return reward = points / 7.

Reward never raises: a judge outage yields score 0 so a training step is not killed.
Concurrency is bounded by $PROOFBENCH_GRADER_CONCURRENCY (default 32).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
import yaml

SCORE_PATTERN = re.compile(r"<score>\s*([0-7])\s*</score>", re.IGNORECASE | re.DOTALL)
POINT_PATTERN = re.compile(r"<points>\s*([0-7])\s*out\s*of\s*7\s*</points>", re.IGNORECASE | re.DOTALL)
THINK_CLOSE_PATTERN = re.compile(r"</think\s*>", re.IGNORECASE)
THINK_OPEN_PATTERN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_BLOCK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)

_TEMPLATE_CACHE: dict[tuple[str, str], str] = {}
_SESSION: aiohttp.ClientSession | None = None
_SESSION_LOOP: asyncio.AbstractEventLoop | None = None
_SEMAPHORE: asyncio.Semaphore | None = None


def _concurrency() -> int:
    try:
        return max(1, int(os.getenv("PROOFBENCH_GRADER_CONCURRENCY", "32")))
    except ValueError:
        return 32


def _session() -> aiohttp.ClientSession:
    """Lazily build an aiohttp session bound to the current running loop."""
    global _SESSION, _SESSION_LOOP, _SEMAPHORE
    loop = asyncio.get_event_loop()
    if _SESSION is None or _SESSION.closed or _SESSION_LOOP is not loop:
        limit = _concurrency()
        timeout = aiohttp.ClientTimeout(total=float(os.getenv("PROOFBENCH_GRADER_TIMEOUT", "1800")))
        _SESSION = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=limit), timeout=timeout)
        _SESSION_LOOP = loop
        _SEMAPHORE = asyncio.Semaphore(limit)
    return _SESSION


def _semaphore() -> asyncio.Semaphore:
    if _SEMAPHORE is None:
        _session()
    assert _SEMAPHORE is not None
    return _SEMAPHORE


def _chat_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/chat/completions") else f"{url}/chat/completions"


def _load_template(path: str, name: str) -> str:
    key = (path, name)
    if key not in _TEMPLATE_CACHE:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        tmpl = ((data or {}).get("templates") or {}).get(name)
        if not tmpl or "template" not in tmpl:
            raise ValueError(f"Template {name!r} not found in {path}")
        _TEMPLATE_CACHE[key] = str(tmpl["template"])
    return _TEMPLATE_CACHE[key]


def _public_answer(action: str, require_think_close: bool) -> str:
    """Strip the model's thinking, returning the gradeable public answer.

    Mirrors grader.py::_answer_for_judge. With require_think_close=True, a rollout with
    no ``</think>`` is treated as having no public answer (empty -> reward 0).
    """
    action = (action or "").replace("<|im_end|>", "").strip()
    closes = list(THINK_CLOSE_PATTERN.finditer(action))
    if closes:
        return action[closes[-1].end():].strip()
    if require_think_close:
        return ""
    # Lenient: drop any complete <think>...</think> block; else if a dangling <think> opens
    # with no close, there is no public answer.
    answer = THINK_BLOCK_PATTERN.sub("", action).strip()
    if answer == action and THINK_OPEN_PATTERN.search(action):
        return ""
    return answer


def _extract_points(text: str) -> int | None:
    m = SCORE_PATTERN.search(text or "") or POINT_PATTERN.search(text or "")
    return int(m.group(1)) if m else None


def _parse_spec(extra_info: dict[str, Any]) -> dict[str, Any]:
    raw = (extra_info or {}).get("reward_spec_json")
    if not raw:
        raise ValueError("extra_info.reward_spec_json missing")
    return json.loads(raw)


# verl's agent_loop builds one non_tensor_batch column per reward-info key and requires
# EVERY sample's dict to expose the SAME keys (agent_loop.py:990-992). So compute_score
# must always return exactly this schema, regardless of which code path produced it.
_RESULT_KEYS = ("score", "points", "judge_model", "judge_error", "judge_latency_s")


def _result(score: float, points: int, model: str = "", error: str = "", latency: float = 0.0) -> dict[str, Any]:
    return {
        "score": float(score),
        "points": int(points),
        "judge_model": str(model),
        "judge_error": str(error),
        "judge_latency_s": float(latency),
    }


async def _judge_one(solution_str: str, ground_truth: str, extra_info: dict[str, Any]) -> dict[str, Any]:
    spec = _parse_spec(extra_info)
    judge = dict(spec.get("judge") or {})
    scale = float(judge.get("reward_scale", 7.0))
    require_think_close = bool(judge.get("require_think_close", True))
    model = judge.get("model", "")

    answer = _public_answer(solution_str, require_think_close)
    if not answer:
        # Truncated/over-thinking rollout with no closed answer -> legitimately scores 0.
        return _result(0.0, 0, model)

    api_key = os.getenv(judge.get("api_key_env", "DEEPSEEK_API_KEY"))
    if not api_key:
        return _result(0.0, 0, model, error=f"{judge.get('api_key_env')} unset")

    problem = spec.get("problem_statement") or extra_info.get("problem_statement", "")
    reference = spec.get("solution") or extra_info.get("reference_solution") or ground_truth or ""
    template = _load_template(judge["template_path"], judge.get("template_name", "with_solution"))
    prompt = template.format(problem=problem, reference_solution=reference,
                             student_answer=answer, solution=answer)

    messages = [{"role": "user", "content": prompt}]
    if judge.get("system_prompt"):
        messages.insert(0, {"role": "system", "content": judge["system_prompt"]})

    payload: dict[str, Any] = {"model": judge["model"], "messages": messages, "stream": False}
    if judge.get("temperature") is not None:
        payload["temperature"] = judge["temperature"]
    if judge.get("max_completion_tokens") is not None:
        payload["max_completion_tokens"] = judge["max_completion_tokens"]
    if judge.get("extra_body"):
        payload.update(judge["extra_body"])
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    url = _chat_url(judge["base_url"])
    timeout = float(judge.get("request_timeout", 600.0))
    max_retries = int(judge.get("max_retries", 2))
    backoff = float(judge.get("backoff_seconds", 5.0))

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with _semaphore():
                started = time.time()
                async with _session().post(url, json=payload, headers=headers, timeout=timeout) as resp:
                    body_text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"judge HTTP {resp.status}: {body_text[:300]}")
                    body = await resp.json(content_type=None)
            reply = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            points = _extract_points(reply)
            if points is None:
                raise ValueError(f"no <score> in judge reply: {reply[:200]!r}")
            return _result(float(points) / scale, points, model, latency=time.time() - started)
        except Exception as exc:  # noqa: BLE001 - never let a judge failure kill a step
            last_error = exc
            if attempt < max_retries:
                await asyncio.sleep(backoff * attempt + random.random())

    return _result(0.0, 0, model, error=f"{type(last_error).__name__}: {last_error}")


async def compute_score(data_source=None, solution_str: str = "", ground_truth: str = "",
                        extra_info: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
    """Async per-sample reward entry point invoked by verl's reward_loop naive manager.

    Always returns the fixed ``_RESULT_KEYS`` schema so verl's agent_loop can build a
    consistent non_tensor_batch column for every sample.
    """
    try:
        res = await _judge_one(solution_str, ground_truth, extra_info or {})
    except Exception as exc:  # noqa: BLE001
        res = _result(0.0, 0, error=f"{type(exc).__name__}: {exc}")
    # Defensive: guarantee exactly the expected keys regardless of code path.
    return {k: res.get(k, _result(0.0, 0)[k]) for k in _RESULT_KEYS}
