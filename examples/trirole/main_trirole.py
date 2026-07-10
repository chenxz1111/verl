"""Tri-role RL entry point.

Same wiring as ``verl.trainer.main_ppo`` -> ``main_ppo_v0.TaskRunner`` (the path all
previous proofbench runs used), with RayPPOTrainer swapped for TriRoleTrainer.

Launch:
    python examples/trirole/main_trirole.py <hydra overrides...>
"""

import os
import socket
import sys

import hydra
import ray
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import TaskRunner as _RemoteTaskRunner
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

# Unwrap the @ray.remote decoration so we can subclass the plain class.
_TaskRunnerBase = getattr(_RemoteTaskRunner, "__ray_actor_class__", _RemoteTaskRunner)


@ray.remote
class TriRoleTaskRunner(_TaskRunnerBase):
    """main_ppo_v0.TaskRunner with the trainer class swapped to TriRoleTrainer."""

    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        from trirole_trainer import TriRoleTrainer

        print(f"TriRoleTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = TriRoleTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config, task_runner_class=TriRoleTaskRunner)


if __name__ == "__main__":
    main()
