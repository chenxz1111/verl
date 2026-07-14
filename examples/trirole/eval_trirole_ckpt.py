#!/usr/bin/env python3
"""Offline evaluation for a tri-role checkpoint (runs OUTSIDE the trainer, e.g. on 166).

Two evaluations against the fixed 60-problem val set, same gemini judge as training:
  A) solver pass@1  — n solves/problem at temp 0.6, mean tier/7. Directly comparable
     to the in-run val and to the ckpt-5285 baseline (fills the missing step-0 anchor).
  B) tri-role pipeline — the deployment loop and the design's headline metric:
       solve x P (temp 1.0) -> self-grade each once -> pick highest self-graded ->
       if <7, refine once using that solution's own <assessment>/<errors> -> judge final.
     Reports pipeline tier vs pass@1 (gain from self-selection + refine) and vs
     best-of-P-by-judge (oracle upper bound).

Usage (inside container, free GPU node):
  python eval_trirole_ckpt.py --model /path/to/hf_ckpt --tag step10 --pass-n 4 --pipe-p 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/nvme3/cxz/verl/examples/trirole")
import trirole_reward as R
from templates import build_grader_messages, build_refine_messages, build_solver_messages

BASE = Path("/nvme3/cxz/unimath_data/rl_data/data")
SCORE_RE = re.compile(r"<score>\s*([0-7])\s*</score>", re.I | re.S)
ASSESS_RE = re.compile(r"<assessment>(.*?)</assessment>", re.I | re.S)
ERRORS_RE = re.compile(r"<errors>(.*?)</errors>", re.I | re.S)
JUDGE = {
    "model": "gemini-3.1-pro-preview", "base_url": "https://openai.sufy.com/v1",
    "api_key_env": "GEMINI_API_KEY",
    "template_path": "/nvme3/cxz/unimath_data/rl_data/templates/evaluation_06-05version.yaml",
    "template_name": "with_solution", "system_prompt": "You are a helpful assistant.",
    # We pass already-think-stripped public text to the judge, so the strict
    # "must contain </think>" gate must be off here (it's for raw rollouts).
    "require_think_close": False, "temperature": 0.0, "max_completion_tokens": 4096,
    "reward_scale": 7.0, "extra_body": {"stream": False},
}


def public(text: str) -> str:
    i = text.rfind("</think>")
    return text[i + 8:].strip() if i >= 0 else ""


async def _judge_one(sol: str, gt: str, problem: str) -> int:
    ei = {"reward_spec_json": json.dumps({"judge": JUDGE, "problem_statement": problem, "solution": gt}),
          "problem_statement": problem, "reference_solution": gt, "trirole_role": "solve"}
    r = await R.compute_score(solution_str=sol, ground_truth=gt, extra_info=ei)
    return int(r["tier"])


def judge_many(items) -> list[int]:
    """items: [(solution_text, ground_truth, problem)] -> tiers. One event loop per call
    (sglang Engine owns the ambient loop between calls, so judging must not share it)."""
    async def _run():
        return list(await asyncio.gather(*[_judge_one(s, g, p) for s, g, p in items]))
    return asyncio.run(_run())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--val", default=str(BASE / "trirole_v4/val.parquet"))
    p.add_argument("--pass-n", type=int, default=4)
    p.add_argument("--pipe-p", type=int, default=4)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gmu", type=float, default=0.8)
    p.add_argument("--max-model-len", type=int, default=218112)
    p.add_argument("--max-resp", type=int, default=204800)
    p.add_argument("--out", default="/nvme3/cxz/trirole_eval_{tag}.json")
    args = p.parse_args()

    import pandas as pd
    import sglang as sgl
    from transformers import AutoTokenizer

    df = pd.read_parquet(args.val)
    probs = [json.loads(r["reward_spec_json"])["problem_statement"] if "reward_spec_json" in r
             else r["problem_statement"] for r in df["extra_info"]]
    gts = [r.get("reference_solution", "") for r in df["extra_info"]]
    print(f"val: {len(probs)} problems | model: {args.model}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = sgl.Engine(model_path=args.model, tp_size=args.tp, context_length=args.max_model_len,
                     mem_fraction_static=args.gmu, trust_remote_code=True,
                     mamba_scheduler_strategy="no_buffer", disable_radix_cache=True)

    def render(msgs):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, thinking=True)

    sp_solve = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": args.max_resp}
    sp_val = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": args.max_resp}
    sp_grade = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 98304}

    # ---- A) pass@1 (temp 0.6, n solves, mean tier) ----
    passn = max(args.pass_n, args.pipe_p)
    solve_prompts = [render(build_solver_messages(pr)) for pr in probs for _ in range(passn)]
    solve_out = [o["text"] for o in llm.generate(solve_prompts, sp_val)]
    items = [(public(solve_out[i * passn + j]), gts[i], probs[i])
             for i in range(len(probs)) for j in range(args.pass_n)]
    flat = judge_many(items)
    pass_tiers = [sum(flat[i * args.pass_n:(i + 1) * args.pass_n]) / args.pass_n
                  for i in range(len(probs))]
    pass1 = sum(pass_tiers) / len(pass_tiers) / 7.0
    print(f"[eval] pass@1 = {pass1:.4f}", flush=True)

    # ---- B) pipeline: solve(temp1.0)xP -> self-grade -> pick -> refine -> judge ----
    pipe_prompts = [render(build_solver_messages(pr)) for pr in probs for _ in range(args.pipe_p)]
    pipe_solves = [o["text"] for o in llm.generate(pipe_prompts, sp_solve)]
    # self-grade each solve once
    grade_prompts, gmap = [], []
    for i, pr in enumerate(probs):
        for j in range(args.pipe_p):
            pub = public(pipe_solves[i * args.pipe_p + j])
            if pub:
                grade_prompts.append(render(build_grader_messages(pr, pub))); gmap.append((i, j, pub))
    grade_out = [o["text"] for o in llm.generate(grade_prompts, sp_grade)] if grade_prompts else []
    # pick highest self-graded per problem; refine if <7
    self_scores = {}
    for k, (i, j, pub) in enumerate(gmap):
        g = public(grade_out[k]); m = SCORE_RE.search(g)
        a = ASSESS_RE.search(g); e = ERRORS_RE.search(g)
        self_scores.setdefault(i, []).append((int(m.group(1)) if m else -1, j, pub,
                                              a.group(1).strip() if a else "", e.group(1).strip() if e else ""))
    refine_prompts, rmap, chosen = [], [], {}
    for i in range(len(probs)):
        cands = self_scores.get(i, [])
        if not cands:
            chosen[i] = ""; continue
        best = max(cands, key=lambda c: c[0])
        chosen[i] = best[2]
        if best[0] < 7 and best[3] and best[4]:
            refine_prompts.append(render(build_refine_messages(probs[i], best[2], best[3], best[4])))
            rmap.append(i)
    refine_out = [o["text"] for o in llm.generate(refine_prompts, sp_solve)] if refine_prompts else []
    for k, i in enumerate(rmap):
        chosen[i] = public(refine_out[k]) or chosen[i]
    pipe_tiers = judge_many([(chosen[i], gts[i], probs[i]) for i in range(len(probs))])
    pipeline = sum(pipe_tiers) / len(pipe_tiers) / 7.0
    # oracle: best-of-P by judge (upper bound)
    items = [(public(pipe_solves[i * args.pipe_p + j]), gts[i], probs[i])
             for i in range(len(probs)) for j in range(args.pipe_p)]
    flat = judge_many(items)
    bo_tiers = [max(flat[i * args.pipe_p:(i + 1) * args.pipe_p]) for i in range(len(probs))]
    best_of_p = sum(bo_tiers) / len(bo_tiers) / 7.0

    res = {"tag": args.tag, "model": args.model, "n_problems": len(probs),
           "pass@1": round(pass1, 4), f"pipeline(P={args.pipe_p})": round(pipeline, 4),
           f"best_of_{args.pipe_p}_oracle": round(best_of_p, 4),
           "pipeline_gain_over_pass1": round(pipeline - pass1, 4),
           "refined_count": len(rmap)}
    out = args.out.format(tag=args.tag)
    Path(out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)); print(f"-> {out}")


if __name__ == "__main__":
    main()
