#!/usr/bin/env python3
"""Offline Megatron-dist-ckpt -> HF safetensors converter for qwen3_5_moe (mbridge).

Why not verl's stock model_merger: its params_mapping is standard-GPT-only — it does
not know this model's GDN linear-attention blocks, attention output gates, or the
fused 256-expert MoE. The correct converter is mbridge's own save_weights (the same
path verl uses when checkpoint save_contents includes 'hf_model', and the same
export naming used for per-step SGLang weight sync).

Run on a node with >= 1 free GPU (weights bf16 ~67GB; EP8 dist ckpt reshards to EP1
automatically — mcore dist checkpointing is parallelism-agnostic):

  torchrun --nproc_per_node 1 convert_ckpt_to_hf.py \
    --dist-ckpt /nvme4/cxz/ckpts/.../global_step_10/actor \
    --hf-config /nvme4/cxz/qwen36_sft_v3.1_128k_pack_8ep/hf_checkpoints/checkpoint-5285-hf \
    --out /nvme4/cxz/ckpts/.../step10_hf

NOTE for future runs: prefer adding
  actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]'
so checkpoints come with an HF tree natively and this script is unnecessary.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dist-ckpt", required=True, help=".../global_step_N/actor")
    p.add_argument("--hf-config", required=True, help="HF model dir for config/tokenizer")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    import torch.distributed as dist
    from megatron.core import dist_checkpointing, mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from transformers import AutoConfig

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)

    hf_config = AutoConfig.from_pretrained(args.hf_config, trust_remote_code=True)

    from mbridge import AutoBridge

    bridge = AutoBridge.from_config(hf_config)
    print("bridge:", type(bridge).__name__, flush=True)
    models = bridge.get_model(weight_path=None)  # random-init mcore model, EP1/TP1
    model = models[0] if isinstance(models, (list, tuple)) else models

    # Load the dist checkpoint (reshards EP8->EP1 automatically).
    model_path = str(Path(args.dist_ckpt) / "model")
    sharded_sd = model.sharded_state_dict()
    loaded = dist_checkpointing.load(sharded_sd, model_path)
    model.load_state_dict(loaded, strict=False)
    print(f"loaded dist ckpt from {model_path}", flush=True)

    # Export HF weights via the bridge (same as checkpoint save_contents 'hf_model').
    Path(args.out).mkdir(parents=True, exist_ok=True)
    bridge.save_weights(models if isinstance(models, (list, tuple)) else [model], args.out)
    print(f"saved HF weights -> {args.out}", flush=True)

    # Copy config/tokenizer artifacts so the dir is directly loadable.
    for name in ["config.json", "generation_config.json", "tokenizer_config.json",
                 "tokenizer.json", "vocab.json", "merges.txt", "chat_template.jinja",
                 "special_tokens_map.json", "configuration.json"]:
        src = Path(args.hf_config) / name
        if src.exists():
            shutil.copy(src, Path(args.out) / name)
    print("copied tokenizer/config artifacts; DONE", flush=True)


if __name__ == "__main__":
    main()
