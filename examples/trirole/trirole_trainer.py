"""Tri-role (solver / grader / refine) co-training trainer.

One policy, three roles, one optimizer step per training step:

  wave 1 SOLVE : stock path — each problem repeated rollout.n times, judged (streamed)
                 by the external LLM judge during rollout; reward = bucket(points)/7.
  wave 2 GRADE : per problem pick up to GRADE_PER_PROB solutions (the refine base first,
                 then one from the opposite accept/reject side); self-grade each K times
                 with the SFT grader prompt. Reward is local tier alignment vs the judge
                 (no API), target injected via extra_info["trirole_grade_target"].
  wave 3 REFINE: per problem pick the highest non-7 tier solution as base; build the SFT
                 refine prompt with the base's public answer + the median self-grade's
                 assessment/errors; sample M refines; judged like solve.

All three waves share identical tensor widths (agent-loop pads to config prompt/response
lengths), so they are concatenated into ONE batch for old_log_prob -> ref -> advantage
(GRPO groups by our per-wave uid) -> a single update_actor call.

Env knobs (read each step):
  TRIROLE_GRADE_PER_PROB (2)   TRIROLE_K_GRADE (4)    TRIROLE_M_REFINE (8)
  TRIROLE_DROP_ZERO_VAR (1)    TRIROLE_LAMBDA_SOLVE/GRADE/REFINE (1.0)
  TRIROLE_GRADE_MAX_NEW / TRIROLE_REFINE_MAX_NEW (0 = use config response_length; needs
      the __sampling_params__ agent-loop patch to take effect)
"""

from __future__ import annotations

import os
import re
import uuid
from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.skip.skip_manager import SkipManager

from templates import build_grader_messages, build_refine_messages

THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
ASSESS_RE = re.compile(r"<assessment>(.*?)</assessment>", re.IGNORECASE | re.DOTALL)
ERRORS_RE = re.compile(r"<errors>(.*?)</errors>", re.IGNORECASE | re.DOTALL)
SCORE_RE = re.compile(r"<score>\s*([0-7])\s*</score>", re.IGNORECASE | re.DOTALL)

VALID_TIERS = (0, 1, 6, 7)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _public_part(text: str) -> str:
    """Text after the last </think>; empty if the think block never closes."""
    closes = list(THINK_CLOSE.finditer(text or ""))
    if not closes:
        return ""
    return text[closes[-1].end():].strip()


class TriRoleTrainer(RayPPOTrainer):
    """RayPPOTrainer with the rollout section replaced by the three-wave tri-role DAG."""

    # ------------------------------------------------------------------ helpers

    def _tokenize_len(self, messages) -> int:
        kwargs = {}
        tmpl_kwargs = self.config.data.get("apply_chat_template_kwargs", None)
        if tmpl_kwargs:
            kwargs.update(dict(tmpl_kwargs))
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **kwargs
        )
        return len(ids)

    def _decode_publics(self, wave: DataProto) -> list[str]:
        """Public (post-</think>) part of every response in a wave batch."""
        responses = wave.batch["responses"]
        prompt_len = wave.batch["prompts"].shape[1]
        resp_mask = wave.batch["attention_mask"][:, prompt_len:]
        lengths = resp_mask.sum(dim=1).tolist()
        texts = []
        for i in range(len(responses)):
            ids = responses[i][: int(lengths[i])]
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return [_public_part(t) for t in texts]

    def _build_wave_rows(self, snapshot: DataProto, specs: list[dict], role: str, repeat: int,
                         max_new_tokens: int) -> DataProto:
        """Build the pre-generation rows for a grade/refine wave.

        Each spec: {"prob_idx": int, "messages": list, "uid": str, "extra": dict}
        Rows are copies of the problem row with raw_prompt / extra_info / uid overridden,
        then repeated `repeat` times (a GRPO group shares one uid).
        """
        rows = snapshot.select_idxs([s["prob_idx"] for s in specs])
        n = len(specs)
        rows.non_tensor_batch["raw_prompt"] = np.array(
            [list(s["messages"]) for s in specs], dtype=object
        )
        rows.non_tensor_batch["uid"] = np.array([s["uid"] for s in specs], dtype=object)
        rows.non_tensor_batch["data_source"] = np.array([f"trirole/{role}"] * n, dtype=object)
        new_extra = []
        for i, s in enumerate(specs):
            base = dict(rows.non_tensor_batch["extra_info"][i] or {})
            base["trirole_role"] = role
            base.update(s.get("extra", {}))
            new_extra.append(base)
        rows.non_tensor_batch["extra_info"] = np.array(new_extra, dtype=object)
        if max_new_tokens and max_new_tokens > 0:
            rows.non_tensor_batch["__sampling_params__"] = np.array(
                [{"max_new_tokens": int(max_new_tokens)}] * n, dtype=object
            )
        rows = rows.repeat(repeat_times=repeat, interleave=True)
        return rows

    def _generate_wave(self, wave_rows: DataProto, timing_raw: dict, timer_key: str) -> DataProto:
        """_get_gen_batch split -> pad -> generate -> unpad -> union back."""
        gen = self._get_gen_batch(wave_rows)  # mutates wave_rows: keeps reward keys + tensors
        gen.meta_info["global_steps"] = self.global_steps
        size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
        gen_padded, pad_size = pad_dataproto_to_divisor(gen, size_divisor)
        with marked_timer(timer_key, timing_raw, color="red"):
            out_padded = self.async_rollout_manager.generate_sequences(gen_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)
        out.meta_info.pop("timing", None)
        return wave_rows.union(out)

    @staticmethod
    def _pick_targets(tiers: np.ndarray, publics: list[str], prompt_ok: list[bool],
                      n_rollout: int, grade_per_prob: int):
        """Per problem: refine base (highest non-7 tier) + grade targets.

        Returns (grade_picks, refine_base) as lists of solve-row indices per problem dict.
        """
        n_problems = len(tiers) // n_rollout
        grade_picks: list[list[int]] = []
        refine_base: list[int | None] = []
        rng = np.random.default_rng()
        for p in range(n_problems):
            rows = list(range(p * n_rollout, (p + 1) * n_rollout))
            valid = [r for r in rows if publics[r] and prompt_ok[r]]
            if not valid:
                grade_picks.append([])
                refine_base.append(None)
                continue
            non7 = [r for r in valid if tiers[r] < 7]
            base = max(non7, key=lambda r: (tiers[r], rng.random())) if non7 else None
            picks: list[int] = []
            if base is not None:
                picks.append(base)
            # opposite side of the base (or of the majority) for target diversity
            if len(picks) < grade_per_prob:
                ref_side = (tiers[base] >= 6) if base is not None else True
                opposite = [r for r in valid if ((tiers[r] >= 6) != ref_side) and r not in picks]
                pool = opposite or [r for r in valid if r not in picks]
                if pool:
                    picks.append(int(rng.choice(pool)))
            while len(picks) < grade_per_prob and valid:
                picks.append(int(rng.choice(valid)))  # duplicates allowed: separate groups
            grade_picks.append(picks[:grade_per_prob])
            refine_base.append(base)
        return grade_picks, refine_base

    def _drop_groups_for_divisibility(self, big: DataProto, world: int, drop_zero_var: bool,
                                      metrics: dict) -> DataProto:
        """Optionally drop zero-variance uid groups, then enforce len % world == 0."""
        uids = big.non_tensor_batch["uid"]
        scores = np.asarray(big.non_tensor_batch["score"], dtype=np.float64)
        groups: dict[str, list[int]] = defaultdict(list)
        for i, u in enumerate(uids):
            groups[u].append(i)

        zero_var = {u for u, idx in groups.items() if np.std(scores[idx]) < 1e-9}
        drop: set[str] = set(zero_var) if drop_zero_var else set()

        def kept_len(dropped: set[str]) -> int:
            return sum(len(idx) for u, idx in groups.items() if u not in dropped)

        # keep-min fallback: never drop below 2 groups' worth of data
        if drop and kept_len(drop) < 2 * world:
            drop = set()

        # re-add dropped groups (smallest first) until divisible
        if kept_len(drop) % world != 0 and drop:
            for u in sorted(drop, key=lambda u: len(groups[u])):
                drop.discard(u)
                if kept_len(drop) % world == 0:
                    break
        # still not divisible: drop more groups (zero-var first, then smallest)
        guard = 0
        candidates = sorted(groups, key=lambda u: (u not in zero_var, len(groups[u])))
        while kept_len(drop) % world != 0 and guard < len(candidates):
            u = candidates[guard]
            guard += 1
            if u in drop:
                continue
            drop.add(u)
            if kept_len(drop) <= max(world, len(uids) // 2):
                drop.discard(u)
                break

        metrics["trirole/zero_var_groups_frac"] = len(zero_var) / max(1, len(groups))
        kept = kept_len(drop)
        if drop and kept % world == 0 and kept > 0:
            keep_idx = [i for i, u in enumerate(uids) if u not in drop]
            metrics["trirole/dropped_rows"] = len(uids) - len(keep_idx)
            return big.select_idxs(keep_idx)
        # Last resort: row-level truncation to divisibility (biases the affected groups'
        # baselines slightly; logged loudly, should be rare).
        remainder = len(uids) % world
        metrics["trirole/dropped_rows"] = remainder
        if remainder:
            print(f"[trirole] WARNING: truncating {remainder} trailing rows for divisibility")
            return big.select_idxs(list(range(len(uids) - remainder)))
        return big

    @staticmethod
    def _trirole_metrics(big: DataProto, refine_specs: list[dict]) -> dict:
        m: dict[str, float] = {}
        role_id = np.asarray(big.non_tensor_batch["role_id"], dtype=int)
        score = np.asarray(big.non_tensor_batch["score"], dtype=np.float64)
        tier = np.asarray(big.non_tensor_batch["tier"], dtype=int)
        prompt_len = big.batch["prompts"].shape[1]
        resp_len = big.batch["attention_mask"][:, prompt_len:].sum(dim=1).float().numpy()
        names = {0: "solve", 1: "grade", 2: "refine"}
        for rid, name in names.items():
            mask = role_id == rid
            if not mask.any():
                m[f"trirole/{name}/n_rows"] = 0
                continue
            m[f"trirole/{name}/n_rows"] = int(mask.sum())
            m[f"trirole/{name}/reward_mean"] = float(score[mask].mean())
            m[f"trirole/{name}/reward_std"] = float(score[mask].std())
            m[f"trirole/{name}/response_len_mean"] = float(resp_len[mask].mean())
            for t in VALID_TIERS:
                m[f"trirole/{name}/tier{t}_frac"] = float((tier[mask] == t).mean())
        gmask = role_id == 1
        if gmask.any():
            sm = np.asarray(big.non_tensor_batch["side_match"], dtype=int)[gmask]
            em = np.asarray(big.non_tensor_batch["exact_match"], dtype=int)[gmask]
            m["trirole/grade/side_match"] = float((sm == 1).mean())
            m["trirole/grade/exact_match"] = float((em == 1).mean())
        rmask = role_id == 2
        if rmask.any() and refine_specs:
            base_tier = {s["uid"]: s["extra"]["trirole_base_tier"] for s in refine_specs}
            uids = big.non_tensor_batch["uid"]
            deltas, wins = [], []
            per_group: dict[str, list[int]] = defaultdict(list)
            for i in np.nonzero(rmask)[0]:
                per_group[uids[i]].append(int(tier[i]))
            for u, ts in per_group.items():
                b = base_tier.get(u)
                if b is None:
                    continue
                deltas.append(float(np.mean(ts)) - b)
                wins.append(float(max(ts) > b))
            if deltas:
                m["trirole/refine/delta_mean"] = float(np.mean(deltas))
                m["trirole/refine/win_rate"] = float(np.mean(wins))
        return m

    # ------------------------------------------------------------------ fit

    def fit(self):  # noqa: C901 - adapted copy of the stock training loop
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        assert self.config.algorithm.adv_estimator not in (AdvantageEstimator.REMAX,), \
            "trirole trainer does not support REMAX"
        assert not self.use_critic, "trirole trainer does not support a critic"

        if self._dump_executor._shutdown:
            self._init_dump_executor()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        current_epoch = self.global_steps // len(self.train_dataloader)
        SkipManager.init(self.config)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        from tqdm import tqdm

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps,
                            desc="TriRole Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        SkipManager.set_step(self.global_steps)

        world = self.resource_pool_manager.get_n_gpus()
        rollout_n = self.config.actor_rollout_ref.rollout.n

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics: dict = {}
                timing_raw: dict = {}

                grade_per_prob = _env_int("TRIROLE_GRADE_PER_PROB", 2)
                k_grade = _env_int("TRIROLE_K_GRADE", 4)
                m_refine = _env_int("TRIROLE_M_REFINE", 8)
                drop_zero_var = _env_int("TRIROLE_DROP_ZERO_VAR", 1) == 1
                lam = {
                    0: _env_float("TRIROLE_LAMBDA_SOLVE", 1.0),
                    1: _env_float("TRIROLE_LAMBDA_GRADE", 1.0),
                    2: _env_float("TRIROLE_LAMBDA_REFINE", 1.0),
                }
                grade_max_new = _env_int("TRIROLE_GRADE_MAX_NEW", 0)
                refine_max_new = _env_int("TRIROLE_REFINE_MAX_NEW", 0)
                max_prompt_tokens = _env_int(
                    "TRIROLE_MAX_PROMPT_TOKENS", int(self.config.data.max_prompt_length)
                )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                snapshot = batch.select(deepcopy=True)

                is_last_step = self.global_steps >= self.total_training_steps
                refine_specs: list[dict] = []
                with marked_timer("step", timing_raw):
                    # ---------------- wave 1: SOLVE ----------------
                    gen_batch = self._get_gen_batch(batch)
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    solve_gen = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
                    with marked_timer("gen", timing_raw, color="red"):
                        solve_out = self.async_rollout_manager.generate_sequences(solve_gen)
                        timing_raw.update(solve_out.meta_info.get("timing", {}))
                        solve_out.meta_info.pop("timing", None)
                    reward_extra_keys = list(solve_out.meta_info.get("reward_extra_keys", []))
                    batch = batch.repeat(repeat_times=rollout_n, interleave=True)
                    batch = batch.union(solve_out)
                    waves = [batch]

                    # ---------------- selection ----------------
                    with marked_timer("trirole_select", timing_raw):
                        tiers = np.asarray(batch.non_tensor_batch["tier"], dtype=int)
                        publics = self._decode_publics(batch)
                        problems = [
                            (ei or {}).get("problem_statement", "")
                            for ei in snapshot.non_tensor_batch["extra_info"]
                        ]
                        # prompt-fit guard for grader (refine checked separately after critique)
                        prompt_ok = []
                        for r in range(len(batch)):
                            p = publics[r]
                            if not p:
                                prompt_ok.append(False)
                                continue
                            msgs = build_grader_messages(problems[r // rollout_n], p)
                            prompt_ok.append(self._tokenize_len(msgs) <= max_prompt_tokens - 8)
                        grade_picks, refine_base = self._pick_targets(
                            tiers, publics, prompt_ok, rollout_n, grade_per_prob
                        )

                    # ---------------- wave 2: GRADE ----------------
                    grade_specs: list[dict] = []
                    for p, picks in enumerate(grade_picks):
                        puid = snapshot.non_tensor_batch["uid"][p]
                        for j, r in enumerate(picks):
                            grade_specs.append({
                                "prob_idx": p,
                                "solve_row": r,
                                "messages": build_grader_messages(problems[p], publics[r]),
                                "uid": f"{puid}#g{j}",
                                "extra": {
                                    "trirole_grade_target": int(tiers[r]),
                                    "trirole_graded_row": int(r),
                                },
                            })
                    grade_batch = None
                    if grade_specs:
                        grade_rows = self._build_wave_rows(
                            snapshot, grade_specs, "grade", k_grade, grade_max_new
                        )
                        grade_batch = self._generate_wave(grade_rows, timing_raw, "gen_grade")
                        waves.append(grade_batch)

                    # ---------------- wave 3: REFINE ----------------
                    if grade_batch is not None:
                        grade_publics = self._decode_publics(grade_batch)
                        grade_uids = grade_batch.non_tensor_batch["uid"]
                        by_uid: dict[str, list[int]] = defaultdict(list)
                        for i, u in enumerate(grade_uids):
                            by_uid[u].append(i)

                        for p, base in enumerate(refine_base):
                            if base is None:
                                continue
                            puid = snapshot.non_tensor_batch["uid"][p]
                            # the base was always grade pick j=0
                            guid = f"{puid}#g0"
                            spec0 = next((s for s in grade_specs
                                          if s["uid"] == guid and s["solve_row"] == base), None)
                            if spec0 is None or guid not in by_uid:
                                continue
                            cands = []
                            for i in by_uid[guid]:
                                pub = grade_publics[i]
                                sm = SCORE_RE.search(pub or "")
                                am = ASSESS_RE.search(pub or "")
                                em = ERRORS_RE.search(pub or "")
                                if sm and am and em and int(sm.group(1)) in VALID_TIERS:
                                    cands.append((int(sm.group(1)), am.group(1).strip(),
                                                  em.group(1).strip()))
                            if not cands:
                                continue
                            cands.sort(key=lambda c: c[0])
                            med = cands[len(cands) // 2]
                            msgs = build_refine_messages(problems[p], publics[base], med[1], med[2])
                            if self._tokenize_len(msgs) > max_prompt_tokens - 8:
                                continue
                            refine_specs.append({
                                "prob_idx": p,
                                "messages": msgs,
                                "uid": f"{puid}#r0",
                                "extra": {
                                    "trirole_base_tier": int(tiers[base]),
                                    "trirole_base_row": int(base),
                                },
                            })
                        if refine_specs:
                            refine_rows = self._build_wave_rows(
                                snapshot, refine_specs, "refine", m_refine, refine_max_new
                            )
                            refine_batch = self._generate_wave(refine_rows, timing_raw, "gen_refine")
                            waves.append(refine_batch)

                    self.checkpoint_manager.sleep_replicas()

                    # ---------------- assemble one batch ----------------
                    big = DataProto.concat(waves) if len(waves) > 1 else waves[0]
                    big.meta_info["reward_extra_keys"] = reward_extra_keys
                    if "response_mask" not in big.batch.keys():
                        big.batch["response_mask"] = compute_response_mask(big)

                    big = self._drop_groups_for_divisibility(big, world, drop_zero_var, metrics)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(big, metrics=metrics)
                    big.meta_info["global_token_num"] = torch.sum(
                        big.batch["attention_mask"], dim=-1
                    ).tolist()
                    images_seqlens_all: list = []
                    for mmi in big.non_tensor_batch.get("multi_modal_inputs", []):
                        if isinstance(mmi, dict) and "image_grid_thw" in mmi:
                            images_seqlens_all.extend(mmi["images_seqlens"].tolist())
                    big.meta_info["images_seqlens"] = images_seqlens_all

                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor, reward_extra_infos_dict = extract_reward(big)

                    # ---------------- old_log_prob / ref ----------------
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(big)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = big.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        metrics.update({
                            "actor/entropy": entropy_agg.detach().item(),
                            "perf/mfu/actor_infer": old_log_prob_mfu,
                        })
                        old_log_prob.batch.pop("entropys")
                        big = big.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(big)
                            big = big.union(ref_log_prob)

                    # ---------------- advantage ----------------
                    with marked_timer("adv", timing_raw, color="brown"):
                        big.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            big.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                        if self.config.algorithm.use_kl_in_reward:
                            big, kl_metrics = apply_kl_penalty(
                                big, kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            big.batch["token_level_rewards"] = big.batch["token_level_scores"]

                        norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        big = compute_advantage(
                            big,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=rollout_n,
                            norm_adv_by_std_in_grpo=norm_adv,
                            config=self.config.algorithm,
                        )
                        role_id = np.asarray(big.non_tensor_batch["role_id"], dtype=int)
                        if any(abs(lam[r] - 1.0) > 1e-9 for r in (0, 1, 2)):
                            scale = torch.tensor(
                                [lam[int(r)] for r in role_id], dtype=big.batch["advantages"].dtype
                            ).unsqueeze(-1)
                            big.batch["advantages"] = big.batch["advantages"] * scale

                    # ---------------- update ----------------
                    with marked_timer("update_actor", timing_raw, color="red"):
                        mini = max(1, (len(big) + rollout_n - 1) // rollout_n)
                        self.config.actor_rollout_ref.actor.ppo_mini_batch_size = mini
                        actor_output = self._update_actor(big)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights(self.global_steps)

                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(big, reward_extra_infos_dict, timing_raw,
                                               rollout_data_dir)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=big, use_critic=False))
                metrics.update(self._trirole_metrics(big, refine_specs))
                metrics.update(compute_timing_metrics(batch=big, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(batch=big, timing_raw=timing_raw, n_gpus=world)
                )
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                SkipManager.set_step(self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=big)

        self._shutdown_dump_executor()
