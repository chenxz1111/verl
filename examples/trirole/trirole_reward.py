"""Unified tri-role (solve / grade / refine) reward for verl's reward_loop naive manager.

Routing key: ``extra_info["trirole_role"]`` in {"solve", "grade", "refine"} (missing -> "solve",
so plain solver-only datasets, e.g. the val parquet, keep working unchanged).

Scoring is on the strict four-tier scale {0, 1, 6, 7} shared by the whole system:
  bucket(e): 0->0, 1..5->1, 6->6, 7->7   (judge prompt unchanged; bucketing at the reward layer)

- solve / refine: external LLM judge (with_solution template + reference solution) gives raw
  points 0-7; reward = bucket(points)/7.  Judge client is a verbatim port of the proven
  deepseek_judge_reward.py (never raises; outage -> 0; empty/unclosed-think public answer -> 0).
- grade: NO API call. Parse the self-grader's <score>X</score> from the public answer
  (X must be in {0,1,6,7}); target tier travels in ``extra_info["trirole_grade_target"]``
  (computed by the trainer from the solve wave's judge points).
      reward = 0.5 * [side(g)==side(t)] + 0.5 * [g==t],   side(x) = x >= 6
  Unparseable / unclosed think / invalid score value -> 0 (format gate).

verl's agent_loop builds one non_tensor column per reward-info key and requires EVERY sample
in a generate_sequences call to expose the SAME keys (agent_loop.py:990-992), so this module
always returns exactly ``_RESULT_KEYS`` regardless of role or code path.
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

VALID_TIERS = (0, 1, 6, 7)

_TEMPLATE_CACHE: dict[tuple[str, str], str] = {}
_SESSION: aiohttp.ClientSession | None = None
_SESSION_LOOP: asyncio.AbstractEventLoop | None = None
_SEMAPHORE: asyncio.Semaphore | None = None


def bucket(points: int) -> int:
    """Map raw judge points 0-7 onto the strict tier scale {0,1,6,7}."""
    if points <= 0:
        return 0
    if points <= 5:
        return 1
    return points  # 6 or 7


def side(tier: int) -> int:
    """Accept/reject boundary: sound-core (6/7) vs not (0/1)."""
    return 1 if tier >= 6 else 0


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
    """Strip the model's thinking, returning the public answer after the last </think>."""
    action = (action or "").replace("<|im_end|>", "").strip()
    closes = list(THINK_CLOSE_PATTERN.finditer(action))
    if closes:
        return action[closes[-1].end():].strip()
    if require_think_close:
        return ""
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


_ROLE_IDS = {"solve": 0, "grade": 1, "refine": 2}

# Constant key set across ALL roles and code paths (agent_loop constant-key requirement).
_RESULT_KEYS = (
    "score",           # final reward in [0,1]
    "points",          # solve/refine: raw judge 0-7; grade: parsed self score; -1 = N/A
    "tier",            # solve/refine: bucket(points); grade: parsed self tier; -1 = N/A
    "role_id",         # 0 solve / 1 grade / 2 refine
    "judge_model",
    "judge_error",
    "judge_latency_s",
    "target_tier",     # grade only; -1 otherwise
    "side_match",      # grade only; -1 otherwise
    "exact_match",     # grade only; -1 otherwise
)


def _result(
    score: float,
    points: int = -1,
    tier: int = -1,
    role: str = "solve",
    model: str = "",
    error: str = "",
    latency: float = 0.0,
    target_tier: int = -1,
    side_match: int = -1,
    exact_match: int = -1,
) -> dict[str, Any]:
    return {
        "score": float(score),
        "points": int(points),
        "tier": int(tier),
        "role_id": int(_ROLE_IDS.get(role, 0)),
        "judge_model": str(model),
        "judge_error": str(error),
        "judge_latency_s": float(latency),
        "target_tier": int(target_tier),
        "side_match": int(side_match),
        "exact_match": int(exact_match),
    }


async def _judge_one(solution_str: str, ground_truth: str, extra_info: dict[str, Any], role: str) -> dict[str, Any]:
    """External-judge scoring for solve/refine rollouts. Reward = bucket(points)/scale."""
    spec = _parse_spec(extra_info)
    judge = dict(spec.get("judge") or {})
    scale = float(judge.get("reward_scale", 7.0))
    require_think_close = bool(judge.get("require_think_close", True))
    model = judge.get("model", "")

    answer = _public_answer(solution_str, require_think_close)
    if not answer:
        return _result(0.0, 0, 0, role, model)

    api_key = os.getenv(judge.get("api_key_env", "DEEPSEEK_API_KEY"))
    if not api_key:
        return _result(0.0, 0, 0, role, model, error=f"{judge.get('api_key_env')} unset")

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
            tier = bucket(points)
            reward = float(tier) / scale
            # SCoRe-lite degradation penalty (refine only, default off): if a refine
            # scores strictly below its draft's tier, subtract a flat penalty so the
            # policy is not rewarded for the least-bad regression. The GRPO group
            # baseline still carries the main relative-improvement signal.
            if role == "refine":
                base_tier = int(extra_info.get("trirole_base_tier", -1))
                pen = float(os.getenv("TRIROLE_REFINE_DEGRADE_PENALTY", "0"))
                if pen > 0 and base_tier in VALID_TIERS and tier < base_tier:
                    reward = max(0.0, reward - pen)
            return _result(reward, points, tier, role, model, latency=time.time() - started)
        except Exception as exc:  # noqa: BLE001 - never let a judge failure kill a step
            last_error = exc
            if attempt < max_retries:
                await asyncio.sleep(backoff * attempt + random.random())

    return _result(0.0, 0, 0, role, model, error=f"{type(last_error).__name__}: {last_error}")


def _grade_alignment(solution_str: str, extra_info: dict[str, Any]) -> dict[str, Any]:
    """Local (no-API) alignment reward for self-grader rollouts."""
    target = int(extra_info.get("trirole_grade_target", -1))
    if target not in VALID_TIERS:
        return _result(0.0, role="grade", error=f"invalid grade target {target}")

    answer = _public_answer(solution_str, require_think_close=True)
    if not answer:
        return _result(0.0, role="grade", target_tier=target, side_match=0, exact_match=0)

    g = _extract_points(answer)
    if g is None or g not in VALID_TIERS:
        # Missing <score> or a forbidden value (2-5): format gate -> 0.
        return _result(0.0, points=(-1 if g is None else g), role="grade",
                       target_tier=target, side_match=0, exact_match=0,
                       error="" if g is None else f"forbidden score {g}")

    s_match = int(side(g) == side(target))
    e_match = int(g == target)
    reward = 0.5 * s_match + 0.5 * e_match
    return _result(reward, points=g, tier=g, role="grade",
                   target_tier=target, side_match=s_match, exact_match=e_match)


async def compute_score(data_source=None, solution_str: str = "", ground_truth: str = "",
                        extra_info: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
    """Async per-sample reward entry point invoked by verl's reward_loop naive manager."""
    extra_info = extra_info or {}
    role = str(extra_info.get("trirole_role", "solve") or "solve")
    try:
        if role == "grade":
            res = _grade_alignment(solution_str, extra_info)
        else:
            res = await _judge_one(solution_str, ground_truth, extra_info, role)
    except Exception as exc:  # noqa: BLE001
        res = _result(0.0, role=role, error=f"{type(exc).__name__}: {exc}")
    # Defensive: guarantee exactly the expected keys regardless of code path.
    defaults = _result(0.0, role=role)
    return {k: res.get(k, defaults[k]) for k in _RESULT_KEYS}
