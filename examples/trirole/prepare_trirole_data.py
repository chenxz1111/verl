#!/usr/bin/env python3
"""Build tri-role verl parquet: solver-SFT-format prompts + judge spec per row.

Differences vs prepare_verl_data.py:
  * ``prompt`` = [solver SYSTEM, problem USER] — byte-identical to the sft_v3.1 solver
    format (the tri-role trainer builds grader/refine prompts at runtime from
    extra_info.problem_statement using the same templates module).
  * --judge {gemini,deepseek} selects the external judge spec (default gemini: ~6x
    faster; scores are not cross-judge comparable, pick one per run).

Run inside the training container (needs pandas/pyarrow):
    python recipe/trirole/prepare_trirole_data.py --out-dir .../data/trirole_v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from templates import build_solver_messages  # noqa: E402

TEMPLATE_PATH = "/nvme3/cxz/unimath_data/rl_data/templates/evaluation_06-05version.yaml"

JUDGES = {
    "deepseek": {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "template_path": TEMPLATE_PATH,
        "template_name": "with_solution",
        "system_prompt": "You are a helpful assistant.",
        "require_think_close": True,
        "temperature": 0.0,
        "max_completion_tokens": 4096,
        "request_timeout": 600.0,
        "max_retries": 2,
        "backoff_seconds": 5.0,
        "reward_scale": 7.0,
        "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "max", "stream": False},
    },
    "gemini": {
        "model": "gemini-3.1-pro-preview",
        "base_url": "https://openai.sufy.com/v1",
        "api_key_env": "GEMINI_API_KEY",
        "template_path": TEMPLATE_PATH,
        "template_name": "with_solution",
        "system_prompt": "You are a helpful assistant.",
        "require_think_close": True,
        "temperature": 0.0,
        "max_completion_tokens": 4096,
        "request_timeout": 600.0,
        "max_retries": 2,
        "backoff_seconds": 5.0,
        "reward_scale": 7.0,
        "extra_body": {"stream": False},
    },
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert(src: Path, split: str, judge: dict) -> list[dict]:
    rows = []
    for i, raw in enumerate(read_jsonl(src)):
        meta = raw.get("metadata") or {}
        reward_spec = dict(meta.get("reward_spec") or {})
        reward_spec["judge"] = dict(judge)

        problem_statement = reward_spec.get("problem_statement", "")
        if not problem_statement:
            # fall back to the old wrapper prompt's problem block if needed
            raise ValueError(f"row {i} in {src} has no reward_spec.problem_statement")
        reference_solution = reward_spec.get("solution") or raw.get("label") or ""

        src_extra = dict(meta.get("extra_info") or {})
        extra_info = {
            **src_extra,
            "split": split,
            "index": i,
            "problem_statement": problem_statement,
            "reference_solution": reference_solution,
            "reward_spec_json": json.dumps(reward_spec, ensure_ascii=False),
        }

        rows.append(
            {
                "data_source": raw.get("data_source", f"trirole/{split}"),
                "prompt": build_solver_messages(problem_statement),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": reference_solution},
                "extra_info": extra_info,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    base = Path("/nvme3/cxz/unimath_data/rl_data/data")
    p.add_argument("--train-src", type=Path, default=base / "train_v3.jsonl")
    p.add_argument("--val-src", type=Path, default=base / "validation.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "trirole_v3")
    p.add_argument("--judge", choices=sorted(JUDGES), default="gemini")
    args = p.parse_args()

    judge = JUDGES[args.judge]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = convert(args.train_src, "train", judge)
    val_rows = convert(args.val_src, "validation", judge)

    train_out = args.out_dir / "train.parquet"
    val_out = args.out_dir / "val.parquet"
    pd.DataFrame(train_rows).to_parquet(train_out, index=False)
    pd.DataFrame(val_rows).to_parquet(val_out, index=False)
    print(f"train: {len(train_rows)} rows -> {train_out}")
    print(f"val:   {len(val_rows)} rows -> {val_out}")

    for name, out in [("train", train_out), ("val", val_out)]:
        df = pd.read_parquet(out)
        spec = json.loads(df.iloc[0]["extra_info"]["reward_spec_json"])
        j = spec["judge"]
        msgs = df.iloc[0]["prompt"]
        assert j["model"] == judge["model"], j
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user", msgs
        print(f"[{name}] judge={j['model']} key_env={j['api_key_env']} "
              f"prompt=[system({len(msgs[0]['content'])}ch), user({len(msgs[1]['content'])}ch)]")


if __name__ == "__main__":
    main()
