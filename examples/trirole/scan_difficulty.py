#!/usr/bin/env python3
"""Difficulty pre-scan for the tri-role RL pool: filter out zero-variance problems.

Root cause of the 75% zero-variance-group drop rate (step 1, b12r2): the pool is
too easy — ~half the problems get solved to tier-7 by all 8 rollouts (zero GRPO
variance -> advantage 0 -> whole group discarded). This scans each problem with K
independent solves under the CURRENT policy, scores with the same judge, and keeps
only problems whose tier spread is nonzero (0 < pass_rate < 1), where pass = tier 7.

Output: <pool>_scored.jsonl (adds extra_info.difficulty = {n7,n0,pass_rate,keep}) and
a filtered <pool>_filtered.jsonl ready for prepare_trirole_data.py.

Runs as a standalone SGLang batch job (NOT through the trainer) so it can use a free
node or an idle window. Reuses the deployed model + judge.

Usage (inside container, on a free node):
  python scan_difficulty.py --pool rl_v4.jsonl --k 6 --out-suffix _scored
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/nvme3/cxz/verl/examples/trirole")
import trirole_reward as R  # judge client (bucket, _judge_one via compute_score)
from templates import build_solver_messages

BASE = Path("/nvme3/cxz/unimath_data/rl_data/data")


async def scan(args) -> None:
    # Lazy import so this file imports even without sglang present.
    import sglang as sgl

    pool_path = BASE / args.pool
    rows = [json.loads(l) for l in pool_path.open() if l.strip()]
    print(f"loaded {len(rows)} problems from {pool_path}")

    llm = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp,
        context_length=args.max_model_len,
        mem_fraction_static=args.gmu,
        trust_remote_code=True,
        **{"mamba_scheduler_strategy": "no_buffer", "disable_radix_cache": True},
    )
    tok = llm.tokenizer_manager  # for chat template
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sampling = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": args.max_resp}

    scored = []
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        prompts = []
        for r in chunk:
            spec = (r.get("metadata") or {}).get("reward_spec") or {}
            problem = spec.get("problem_statement", "")
            msgs = build_solver_messages(problem)
            prompts.append(hf_tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                **({"thinking": True}),
            ))
        # K solves per problem
        rep = [p for p in prompts for _ in range(args.k)]
        outs = llm.generate(rep, sampling)
        texts = [o["text"] for o in outs]

        # judge each
        for i, r in enumerate(chunk):
            spec = (r.get("metadata") or {}).get("reward_spec") or {}
            gt = spec.get("solution", "")
            ei = {"reward_spec_json": json.dumps({**spec, "judge": args.judge_spec}),
                  "problem_statement": spec.get("problem_statement", ""),
                  "reference_solution": gt, "trirole_role": "solve"}
            tiers = []
            for j in range(args.k):
                res = await R.compute_score(solution_str=texts[i * args.k + j],
                                            ground_truth=gt, extra_info=ei)
                tiers.append(int(res["tier"]))
            n7 = sum(t == 7 for t in tiers)
            n0 = sum(t == 0 for t in tiers)
            pr = n7 / len(tiers)
            keep = 0.0 < pr < 1.0 or (len(set(tiers)) > 1)  # any spread is useful
            r.setdefault("metadata", {}).setdefault("extra_info", {})["difficulty"] = {
                "n7": n7, "n0": n0, "pass_rate": pr, "tiers": tiers, "keep": keep,
            }
            scored.append(r)
        print(f"scanned {min(start+args.batch, len(rows))}/{len(rows)} "
              f"| kept so far {sum(x['metadata']['extra_info']['difficulty']['keep'] for x in scored)}")

    out = BASE / (pool_path.stem + args.out_suffix + ".jsonl")
    filt = BASE / (pool_path.stem + "_filtered.jsonl")
    with out.open("w") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = [r for r in scored if r["metadata"]["extra_info"]["difficulty"]["keep"]]
    with filt.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDONE: {len(scored)} scored -> {out}")
    print(f"      {len(kept)} kept (nonzero variance) -> {filt}  "
          f"[{len(kept)/len(scored):.0%} retained]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="rl_v4.jsonl")
    p.add_argument("--model", default="/nvme4/cxz/qwen36_sft_v3.1_128k_pack_8ep/hf_checkpoints/checkpoint-5285-hf")
    p.add_argument("--k", type=int, default=6, help="solves per problem")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gmu", type=float, default=0.8)
    p.add_argument("--max-model-len", type=int, default=140000)
    p.add_argument("--max-resp", type=int, default=131072)
    p.add_argument("--batch", type=int, default=16, help="problems per generate call")
    p.add_argument("--out-suffix", default="_scored")
    args = p.parse_args()
    # gemini judge spec (matches prepare_trirole_data.py)
    args.judge_spec = {
        "model": "gemini-3.1-pro-preview",
        "base_url": "https://openai.sufy.com/v1",
        "api_key_env": "GEMINI_API_KEY",
        "template_path": "/nvme3/cxz/unimath_data/rl_data/templates/evaluation_06-05version.yaml",
        "template_name": "with_solution", "system_prompt": "You are a helpful assistant.",
        "require_think_close": True, "temperature": 0.0, "max_completion_tokens": 4096,
        "reward_scale": 7.0, "extra_body": {"stream": False},
    }
    asyncio.run(scan(args))


if __name__ == "__main__":
    main()
