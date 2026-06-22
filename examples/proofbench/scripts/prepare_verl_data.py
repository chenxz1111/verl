#!/usr/bin/env python3
"""Convert the slime-format proof-grading JSONL into verl parquet.

Source rows (slime):
    {data_source, prompt:[{role,content}], label:<ref solution>,
     metadata:{reward_spec:{method, problem_statement, solution, short_answer, judge:{...}},
               extra_info:{...}, env_class}}

verl rows (one parquet row each):
    {data_source, prompt:[{role,content}], ability,
     reward_model:{style:"rule", ground_truth:<ref solution>},
     extra_info:{split, index, problem_statement, reference_solution,
                 reward_spec_json:<json str>, ...source extra_info...}}

Key transforms:
  * Force every judge (train + val) to deepseek-v4-flash via the DeepSeek endpoint,
    reading the key from $DEEPSEEK_API_KEY.
  * Store reward_spec as a JSON *string* (extra_info.reward_spec_json) so pyarrow does
    not try to infer a deep heterogeneous struct across rows.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

# Default judge template ships in this recipe (templates/evaluation_06-05version.yaml).
# Override with --template-path or $PROOFBENCH_TEMPLATE for a different location.
_DEFAULT_TEMPLATE = os.environ.get(
    "PROOFBENCH_TEMPLATE",
    str(Path(__file__).resolve().parents[1] / "templates" / "evaluation_06-05version.yaml"),
)

JUDGE_OVERRIDE = {
    "model": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com/chat/completions",
    "api_key_env": "DEEPSEEK_API_KEY",
    "template_path": _DEFAULT_TEMPLATE,
    "template_name": "with_solution",
    "system_prompt": "You are a helpful assistant.",
    "require_think_close": True,
    "temperature": 0.0,
    "max_completion_tokens": 4096,
    "request_timeout": 600.0,
    "max_retries": 2,
    "backoff_seconds": 5.0,
    "reward_scale": 7.0,
    "extra_body": {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "stream": False,
    },
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert(src: Path, split: str) -> list[dict]:
    rows = []
    for i, raw in enumerate(read_jsonl(src)):
        meta = raw.get("metadata") or {}
        reward_spec = dict(meta.get("reward_spec") or {})

        # Force the judge to deepseek-v4-flash for BOTH train and val.
        reward_spec["judge"] = dict(JUDGE_OVERRIDE)

        problem_statement = reward_spec.get("problem_statement", "")
        reference_solution = reward_spec.get("solution") or raw.get("label") or ""

        src_extra = dict(meta.get("extra_info") or {})
        extra_info = {
            **src_extra,
            "split": split,
            "index": i,
            "problem_statement": problem_statement,
            "reference_solution": reference_solution,
            # JSON string keeps pyarrow from choking on the nested judge struct.
            "reward_spec_json": json.dumps(reward_spec, ensure_ascii=False),
        }

        rows.append(
            {
                "data_source": raw.get("data_source", f"proofgrade/{split}"),
                "prompt": raw["prompt"],  # list[{role, content}] kept verbatim
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": reference_solution},
                "extra_info": extra_info,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-src", type=Path, required=True, help="source train jsonl (slime format)")
    p.add_argument("--val-src", type=Path, required=True, help="source validation jsonl (slime format)")
    p.add_argument("--out-dir", type=Path, required=True, help="output dir for train.parquet/val.parquet")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = convert(args.train_src, "train")
    val_rows = convert(args.val_src, "validation")

    train_out = args.out_dir / "train.parquet"
    val_out = args.out_dir / "val.parquet"
    pd.DataFrame(train_rows).to_parquet(train_out, index=False)
    pd.DataFrame(val_rows).to_parquet(val_out, index=False)

    print(f"train: {len(train_rows)} rows -> {train_out}")
    print(f"val:   {len(val_rows)} rows -> {val_out}")

    # Sanity: round-trip one row of each split and confirm the judge override.
    for name, out in [("train", train_out), ("val", val_out)]:
        df = pd.read_parquet(out)
        spec = json.loads(df.iloc[0]["extra_info"]["reward_spec_json"])
        j = spec["judge"]
        assert j["model"] == "deepseek-v4-flash", j
        assert j["api_key_env"] == "DEEPSEEK_API_KEY", j
        assert j["base_url"].startswith("https://api.deepseek.com"), j
        print(f"[{name}] judge OK: model={j['model']} key_env={j['api_key_env']} "
              f"prompt_msgs={len(df.iloc[0]['prompt'])}")


if __name__ == "__main__":
    main()
