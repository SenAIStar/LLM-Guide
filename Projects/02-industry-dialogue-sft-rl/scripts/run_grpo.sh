#!/usr/bin/env bash
set -euo pipefail

# Pin a VeRL commit before running. This is a minimal GRPO skeleton, not a
# version-independent launch script.
: "${TRAIN_PARQUET:?set TRAIN_PARQUET}"
: "${VAL_PARQUET:?set VAL_PARQUET}"
: "${SFT_MODEL:?set SFT_MODEL to the merged SFT checkpoint}"
: "${REWARD_MODEL:?set REWARD_MODEL to a scalar reward model checkpoint}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_PARQUET" \
  data.val_files="$VAL_PARQUET" \
  data.train_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path="$SFT_MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  reward.reward_model.enable=True \
  reward.reward_model.model_path="$REWARD_MODEL" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  trainer.logger="['console']" \
  trainer.project_name=industry_dialogue \
  trainer.experiment_name=grpo_from_sft \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=20 \
  trainer.total_epochs=1

# Group size, batch sizes, learning rate, KL coefficient and GPU settings are
# starting values. Check them against the pinned VeRL version and real hardware.
