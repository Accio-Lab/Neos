# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import time
import uuid

import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _stale_incomplete_rolling_uids(
    items: dict[str, dict],
    *,
    current_step: int,
    max_incomplete_age: int,
) -> list[str]:
    """Return admitted, incomplete groups that reached the rerun boundary."""
    threshold = int(current_step) - int(max_incomplete_age)
    if threshold < 1:
        return []
    candidates = []
    for uid, tag in items.items():
        if (
            tag.get("is_prompt", False)
            and tag.get("status") in {"pending", "running"}
            and bool(tag.get("rolling_first_admitted", False))
            and int(tag.get("global_steps", current_step)) <= threshold
        ):
            candidates.append(str(uid))
    return sorted(
        candidates,
        key=lambda uid: (
            int(items[uid].get("global_steps", 0)),
            int(items[uid].get("rollout_queue_order", 0)),
            uid,
        ),
    )


@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    """Asynchronous PPO trainer
    1. Trainer and rollout are colocated.
    2. Partial rollout is enabled.
    """

    def get_llm_client(self):
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def _rolling_transfer_queue_prompts(self) -> dict[str, dict]:
        data = tq.kv_list("train") or {}
        return {
            str(uid): tag
            for uid, tag in data.get("train", {}).items()
            if tag.get("is_prompt", False)
        }

    def _initialize_rolling_queue_order(self) -> None:
        prompt_tags = self._rolling_transfer_queue_prompts()
        existing_orders = [
            int(tag["rollout_queue_order"])
            for tag in prompt_tags.values()
            if tag.get("rollout_queue_order") is not None
        ]
        self._rolling_next_queue_order = max(existing_orders, default=-1) + 1

    def _assign_rolling_queue_orders(self, batch) -> None:
        start = self._rolling_next_queue_order
        stop = start + len(batch)
        batch["__rollout_queue_order__"] = torch.arange(start, stop, dtype=torch.int64)
        self._rolling_next_queue_order = stop

    def _rolling_add_prompts_to_generate(self, num_prompts: int) -> int:
        """Append replacement groups without disturbing queued/partial work."""
        if num_prompts <= 0:
            return 0
        batch = self._next_train_batch(num_prompts)
        self._assign_rolling_queue_orders(batch)
        submitted = self._submit_batch_to_rollout(batch, rollout_batch_step=self.global_steps)
        logger.info(
            "rolling rollout queue appended groups=%s order_range=[%s,%s)",
            submitted,
            self._rolling_next_queue_order - submitted,
            self._rolling_next_queue_order,
        )
        return submitted

    def _adaptive_prepare_round_batch(self, batch, *, rollout_step: int):
        """Give preserved questions fresh UIDs and current-policy max-session tasks."""
        if batch is None or len(batch) == 0:
            return None
        batch = batch.clone()
        for key in (
            "__rollout_queue_order__",
            "__rollout_n__",
            "__rolling_rerun_of__",
            "global_steps",
        ):
            batch.pop(key, None)
        new_uids = [str(uuid.uuid4()) for _ in range(len(batch))]
        batch["uid"] = NonTensorStack.from_list(
            [NonTensorData(uid) for uid in new_uids]
        )
        self._assign_rolling_queue_orders(batch)
        batch["__rollout_n__"] = torch.full(
            (len(batch),),
            self._adaptive_max_sessions,
            dtype=torch.int64,
        )
        tu.assign_non_tensor_data(batch, "global_steps", rollout_step)
        return batch

    def _adaptive_submit_round(self, batch, *, rollout_step: int) -> int:
        if getattr(self, "_adaptive_current_round_step", None) != rollout_step:
            self._adaptive_current_round_step = rollout_step
            self._adaptive_round_seen_questions = set()
            self._adaptive_progress_checked_at = 0.0
        identities = self._adaptive_question_identities(batch)
        if len(identities) != len(set(identities)) or self._adaptive_round_seen_questions.intersection(identities):
            raise RuntimeError("adaptive round attempted to admit a question twice under the same policy")
        prepared = self._adaptive_prepare_round_batch(batch, rollout_step=rollout_step)
        if prepared is None:
            return 0
        submitted = self._submit_batch_to_rollout(
            prepared,
            rollout_batch_step=rollout_step,
        )
        self._adaptive_round_seen_questions.update(identities)
        logger.info(
            "adaptive mixed-group round submitted: step=%s questions=%s "
            "active_question_window=%s base_n=%s max_n=%s",
            rollout_step,
            submitted,
            self._adaptive_group_window,
            self._adaptive_base_sessions,
            self._adaptive_max_sessions,
        )
        return submitted

    def _adaptive_replenish_starved_round(self, global_steps: int, batch_size: int) -> int:
        """Keep ready groups, and refill a quiescent mixed-group deficit.

        Finished homogeneous groups cannot make progress, even when TransferQueue
        still labels their prompt tasks running. Only recycle questions absent
        from this entire policy round; never reset the per-question 1.5*n cap.
        """
        interval = max(0.0, float(os.getenv("VERL_ADAPTIVE_PROGRESS_CHECK_SECONDS", "10")))
        now = time.monotonic()
        if now - getattr(self, "_adaptive_progress_checked_at", 0.0) < interval:
            return 0
        self._adaptive_progress_checked_at = now
        status = self.agent_loop_manager.get_rolling_admission_status()
        if not status or status.get("mode") != "adaptive_mixed" or status.get("paused"):
            return 0
        if int(status.get("global_steps", -1)) != int(global_steps):
            return 0
        ready = int(status["ready_groups"])
        unresolved = (int(status["registered_groups"]) - int(status["cancelled_groups"])
                      - ready - int(status["exhausted_groups"]))
        if ready >= batch_size or int(status["active"]) or int(status["granted"]) or unresolved > 0:
            return 0
        seen = set(self._adaptive_round_seen_questions)
        wanted = max(batch_size - ready, int(self._adaptive_group_window) - ready)
        rows = []
        for source in reversed(self._adaptive_recycle_batches):
            for index, identity in enumerate(self._adaptive_question_identities(source)):
                if identity in seen:
                    continue
                seen.add(identity)
                rows.append(tu.index_select_tensor_dict(source, [index]))
                if len(rows) >= wanted:
                    break
            if len(rows) >= wanted:
                break
        if len(rows) < batch_size - ready:
            raise RuntimeError(
                "adaptive mixed-group supply exhausted: "
                f"step={global_steps} ready={ready}/{batch_size} "
                f"unseen_recycled_questions={len(rows)} status={status}; "
                "no homogeneous group is admitted to optimization and no question cap is reset"
            )
        added = self._adaptive_submit_round(
            self._adaptive_concat_batches(rows), rollout_step=global_steps,
        )
        logger.warning(
            "adaptive exhausted-round refill: step=%s retained_ready=%s added_questions=%s "
            "round_unique_questions=%s status=%s",
            global_steps, ready, added, len(self._adaptive_round_seen_questions), status,
        )
        return added

    @staticmethod
    def _adaptive_concat_batches(batches):
        values = [batch for batch in batches if batch is not None and len(batch)]
        if not values:
            return None
        return values[0] if len(values) == 1 else tu.concat_tensordict(values)

    @staticmethod
    def _adaptive_question_identities(batch) -> list[str]:
        """Return stable dataset-row identities after rollout UIDs are refreshed."""
        if batch is None:
            return []

        def values_for(field):
            if isinstance(field, torch.Tensor):
                values = field.detach().cpu().reshape(-1).tolist()
            elif isinstance(field, NonTensorStack):
                values = [getattr(value, "data", value) for value in field]
            elif isinstance(field, NonTensorData):
                values = [field.data] * len(batch)
            else:
                values = list(field)
            if len(values) != len(batch):
                raise RuntimeError("adaptive question identity field is batch-misaligned")
            return values

        if "extra_info" in batch:
            extra_info = values_for(batch["extra_info"])
            
            
            
            
            
            identities = []
            for value in extra_info:
                identity = None
                if hasattr(value, "get"):
                    for field in ("source_sample_id", "qid"):
                        candidate = value.get(field)
                        if candidate is not None and str(candidate).strip():
                            identity = f"{field}:{candidate}"
                            break
                if identity is None:
                    break
                identities.append(identity)
            
            
            
            if len(identities) == len(batch):
                return identities
        if "index" in batch:
            return [f"index:{value!r}" for value in values_for(batch["index"])]
        raise RuntimeError("adaptive question has no stable dataset identity")

    def _adaptive_state_path(self, completed_step: int) -> str:
        return os.path.join(
            self.config.trainer.default_local_dir,
            f"adaptive_mixed_scheduler_state_step_{int(completed_step)}.pt",
        )

    def _adaptive_save_scheduler_state(self) -> str:
        if os.getenv("NEOS_MODEL_ONLY_CHECKPOINT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return "disabled:model-only-checkpoint"
        path = self._adaptive_state_path(self.global_steps)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        torch.save(
            {
                "schema_version": "adaptive-mixed-question-queue-v1",
                "completed_step": int(self.global_steps),
                "next_round_batch": self._adaptive_next_round_batch,
                "recycle_batches": self._adaptive_recycle_batches,
            },
            temporary,
        )
        os.replace(temporary, path)
        return path

    def _adaptive_load_scheduler_state(self, completed_step: int) -> bool:
        path = self._adaptive_state_path(completed_step)
        if not os.path.isfile(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (
            state.get("schema_version") != "adaptive-mixed-question-queue-v1"
            or int(state.get("completed_step", -1)) != int(completed_step)
        ):
            raise RuntimeError(f"invalid adaptive scheduler checkpoint: {path}")
        self._adaptive_next_round_batch = state.get("next_round_batch")
        self._adaptive_recycle_batches = list(state.get("recycle_batches") or [])
        logger.info(
            "loaded adaptive scheduler checkpoint: step=%s pending=%s recycle_batches=%s path=%s",
            completed_step,
            len(self._adaptive_next_round_batch) if self._adaptive_next_round_batch is not None else 0,
            len(self._adaptive_recycle_batches),
            path,
        )
        return True

    def _adaptive_capture_next_round(self) -> dict[str, int]:
        """Cancel this policy's surplus rollouts while preserving question order."""
        selected_uids = list(self.replay_buffer.last_selected_prompt_uids)
        selected_prompt_batch = self.replay_buffer.last_selected_prompt_batch
        if len(selected_uids) != int(self.config.data.train_batch_size):
            raise RuntimeError(
                f"adaptive sampler selected {len(selected_uids)} groups, expected "
                f"{self.config.data.train_batch_size}"
            )
        prompt_tags = self._rolling_transfer_queue_prompts()
        unselected_uids = [uid for uid in prompt_tags if uid not in set(selected_uids)]
        unselected_uids.sort(
            key=lambda uid: (
                bool(prompt_tags[uid].get("adaptive_exhausted", False)),
                int(prompt_tags[uid].get("rollout_queue_order", 0)),
                uid,
            )
        )
        unselected_batch = (
            tq.kv_batch_get(keys=unselected_uids, partition_id="train")
            if unselected_uids
            else None
        )
        if selected_prompt_batch is not None:
            self._adaptive_recycle_batches.append(selected_prompt_batch)

        all_old_uids = [*selected_uids, *unselected_uids]
        cancelled_tasks = self.agent_loop_manager.cancel_prompts(all_old_uids)
        after = (tq.kv_list("train") or {}).get("train", {})
        keep = set(self.replay_buffer.last_selected_trajectory_keys)
        clear_keys = [
            key
            for key in after
            if key not in keep
            and (
                after[key].get("is_prompt", False)
                or key.split("_", 1)[0] in set(all_old_uids)
            )
        ]
        if clear_keys:
            tq.kv_clear(keys=clear_keys, partition_id="train")
        self._adaptive_next_round_batch = unselected_batch
        state_path = self._adaptive_save_scheduler_state()
        report = {
            "selected_groups": len(selected_uids),
            "preserved_questions": len(unselected_uids),
            "cancelled_prompt_tasks": int(cancelled_tasks),
            "discarded_old_policy_keys": len(clear_keys),
            "scheduler_state": state_path,
        }
        logger.info("adaptive mixed-group round boundary: %s", report)
        return report

    def _adaptive_batch_with_final_cycle_fill(self):
        batch = self._adaptive_next_round_batch
        target = int(self._adaptive_group_window)
        needed = max(0, target - (len(batch) if batch is not None else 0))
        if needed:
            seen = set(self._adaptive_question_identities(batch))
            recycle_rows = []
            
            
            for source in reversed(self._adaptive_recycle_batches):
                for index, identity in enumerate(self._adaptive_question_identities(source)):
                    if identity in seen:
                        continue
                    seen.add(identity)
                    
                    
                    
                    recycle_rows.append(
                        tu.index_select_tensor_dict(source, [index])
                    )
                    if len(recycle_rows) >= needed:
                        break
                if len(recycle_rows) >= needed:
                    break
            recycle = self._adaptive_concat_batches(recycle_rows)
            if recycle is None or len(recycle) < needed:
                raise RuntimeError(
                    f"adaptive round needs {needed} distinct recycled questions "
                    f"to restore its {target}-question window"
                )
            batch = self._adaptive_concat_batches([batch, recycle])
            logger.info(
                "adaptive question window filled with prior trained questions: added=%s total=%s",
                needed,
                len(batch),
            )
        if batch is None or len(batch) < target:
            raise RuntimeError(
                f"adaptive policy round has {len(batch) if batch is not None else 0} "
                f"questions, below required window={target}"
            )
        return batch

    def _prime_rolling_rollout_queue(self) -> dict[str, int]:
        """Prequeue enough groups for every remaining optimizer update.

        Agent-loop workers create all session coroutines immediately, while
        the admission controller permits only ``train_batch_size * rollout.n
        * inflight_batches`` sessions to execute.  This lets a single finished
        trajectory be replaced without polling the trainer thread.
        """
        train_batch_size = int(self.config.data.train_batch_size)
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        inflight_batches = int(os.getenv("VERL_ROLLING_INFLIGHT_BATCHES", "1"))
        remaining_steps = max(0, self.total_training_steps - self.global_steps + 1)
        required_groups = remaining_steps * train_batch_size
        
        
        
        
        
        runway_groups = train_batch_size * rollout_n * inflight_batches
        target_queued_groups = required_groups + runway_groups
        existing_groups = len(self._rolling_transfer_queue_prompts())
        missing_groups = max(0, target_queued_groups - existing_groups)
        if missing_groups:
            self._rolling_add_prompts_to_generate(missing_groups)
        return {
            "remaining_steps": remaining_steps,
            "required_groups": required_groups,
            "inflight_batches": inflight_batches,
            "runway_groups": runway_groups,
            "target_queued_groups": target_queued_groups,
            "existing_groups": existing_groups,
            "submitted_groups": missing_groups,
        }

    def _prepare_stale_incomplete_group_reruns(self):
        """Cancel and snapshot over-age partial groups for post-update rerun.

        A rerun receives a fresh uid and queue order.  All partial trajectories
        from the old uid are removed, so every member of the replacement GRPO
        group is generated from turn 1 with one current policy version.
        """
        if self.global_steps >= self.total_training_steps:
            return None, {"groups": 0, "trajectories": 0, "cancelled_tasks": 0}
        data = tq.kv_list("train") or {}
        items = data.get("train", {})
        stale_uids = _stale_incomplete_rolling_uids(
            items,
            current_step=self.global_steps,
            max_incomplete_age=self._rolling_max_incomplete_age,
        )
        if not stale_uids:
            return None, {"groups": 0, "trajectories": 0, "cancelled_tasks": 0}

        batch = tq.kv_batch_get(keys=stale_uids, partition_id="train")
        cancelled_tasks = self.agent_loop_manager.cancel_prompts(stale_uids)

        
        
        after = (tq.kv_list("train") or {}).get("train", {})
        stale_set = set(stale_uids)
        trajectory_keys = [
            key
            for key, tag in after.items()
            if not tag.get("is_prompt", False)
            and key.split("_", 1)[0] in stale_set
        ]
        tq.kv_clear(
            keys=[*stale_uids, *trajectory_keys],
            partition_id="train",
        )

        new_uids = [str(uuid.uuid4()) for _ in stale_uids]
        batch["uid"] = NonTensorStack.from_list(
            [NonTensorData(uid) for uid in new_uids]
        )
        self._assign_rolling_queue_orders(batch)
        tu.assign_non_tensor_stack(batch, "__rolling_rerun_of__", stale_uids)
        report = {
            "groups": len(stale_uids),
            "trajectories": len(trajectory_keys),
            "cancelled_tasks": cancelled_tasks,
            "threshold_step": self.global_steps - self._rolling_max_incomplete_age,
            "old_uids": stale_uids,
            "new_uids": new_uids,
        }
        logger.info("prepared stale incomplete groups for current-policy rerun: %s", report)
        return batch, report

    def _submit_pending_stale_group_reruns(self, *, rollout_step: int) -> int:
        batch = getattr(self, "_pending_stale_rerun_batch", None)
        self._pending_stale_rerun_batch = None
        if batch is None or len(batch) == 0:
            return 0
        
        batch.pop("__rolling_rerun_of__", None)
        submitted = self._submit_batch_to_rollout(
            batch,
            rollout_batch_step=rollout_step,
        )
        logger.info(
            "submitted stale incomplete group reruns with current policy: step=%s groups=%s",
            rollout_step,
            submitted,
        )
        return submitted

    def on_init_end(self):
        
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        if self.config.skip.rollout_tq.enable:
            return
        self._rolling_window_enabled = os.getenv(
            "VERL_ROLLING_TRAJECTORY_WINDOW", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._adaptive_mixed_enabled = os.getenv(
            "VERL_ADAPTIVE_MIXED_GROUP_WINDOW", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._rolling_window_owns_dispatch = self._rolling_window_enabled
        num_warmup_batches = self.config.trainer.v1.colocate_async.num_warmup_batches
        if self._rolling_window_enabled:
            if self._adaptive_mixed_enabled:
                if num_warmup_batches != 1:
                    raise ValueError("adaptive mixed-group window requires one warmup batch")
                if not getattr(self.agent_loop_manager, "adaptive_mixed_enabled", False):
                    raise RuntimeError("adaptive trainer/controller configuration mismatch")
                self._adaptive_base_sessions = int(self.config.actor_rollout_ref.rollout.n)
                self._adaptive_max_sessions = int(
                    os.getenv(
                        "VERL_ADAPTIVE_MAX_SESSIONS",
                        str(self._adaptive_base_sessions + self._adaptive_base_sessions // 2),
                    )
                )
                self._adaptive_group_window = 2 * int(self.config.data.train_batch_size)
                self._adaptive_next_round_batch = None
                self._adaptive_recycle_batches = []
                self.replay_buffer.adaptive_progress_fn = self._adaptive_replenish_starved_round
                self._pending_stale_rerun_batch = None
                self._initialize_rolling_queue_order()
                restored = self._adaptive_load_scheduler_state(self.global_steps - 1)
                if restored:
                    existing = (tq.kv_list("train") or {}).get("train", {})
                    if existing:
                        tq.kv_clear(keys=list(existing), partition_id="train")
                    initial_questions = self._adaptive_batch_with_final_cycle_fill()
                else:
                    initial_questions = self._next_train_batch(len(self.train_dataset))
                submitted = self._adaptive_submit_round(
                    initial_questions,
                    rollout_step=self.global_steps,
                )
                status = self.agent_loop_manager.configure_rolling_admission(
                    global_steps=self.global_steps,
                    paused=False,
                )
                logger.info(
                    "enabled adaptive mixed-group scheduler: questions=%s status=%s",
                    submitted,
                    status,
                )
                return
            self._rolling_max_incomplete_age = int(
                os.getenv("VERL_ROLLING_MAX_INCOMPLETE_AGE", "3")
            )
            if self._rolling_max_incomplete_age <= 0:
                raise ValueError("VERL_ROLLING_MAX_INCOMPLETE_AGE must be positive")
            self._pending_stale_rerun_batch = None
            if num_warmup_batches != 1:
                raise ValueError(
                    "rolling trajectory window requires num_warmup_batches=1"
                )
            if not getattr(self.agent_loop_manager, "rolling_admission_enabled", False):
                raise RuntimeError("rolling trajectory window is enabled but its admission controller is absent")
            self._initialize_rolling_queue_order()
            
            
            self.replay_buffer.refill_fn = self._rolling_add_prompts_to_generate
            queue_report = self._prime_rolling_rollout_queue()
            status = self.agent_loop_manager.configure_rolling_admission(
                global_steps=self.global_steps,
                paused=False,
            )
            logger.info(
                "enabled continuous rolling trajectory window: current_step=%s queue=%s admission=%s",
                self.global_steps,
                queue_report,
                status,
            )
            return
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            
            self.checkpoint_manager.update_weights(self.global_steps)
            
            self.checkpoint_manager.resume_generation_replicas()
            if getattr(self, "_rolling_window_enabled", False):
                next_step = self.global_steps + 1
                final_step = self.global_steps >= self.total_training_steps
                if getattr(self, "_adaptive_mixed_enabled", False):
                    submitted = 0
                    if not final_step:
                        next_batch = self._adaptive_batch_with_final_cycle_fill()
                        submitted = self._adaptive_submit_round(
                            next_batch,
                            rollout_step=next_step,
                        )
                    status = self.agent_loop_manager.configure_rolling_admission(
                        global_steps=next_step,
                        paused=final_step,
                    )
                    logger.info(
                        "adaptive mixed-group scheduler after weight update: next_step=%s "
                        "final=%s submitted_questions=%s status=%s",
                        next_step,
                        final_step,
                        submitted,
                        status,
                    )
                    return
                rerun_groups = 0
                if not final_step:
                    rerun_groups = self._submit_pending_stale_group_reruns(
                        rollout_step=next_step
                    )
                status = self.agent_loop_manager.configure_rolling_admission(
                    global_steps=next_step,
                    paused=final_step,
                )
                logger.info(
                    "rolling trajectory window after weight update: next_step=%s final=%s "
                    "rerun_groups=%s admission=%s",
                    next_step,
                    final_step,
                    rerun_groups,
                    status,
                )

    def on_sample_end(self):
        
        
        
        if getattr(self, "_rolling_window_enabled", False):
            status = self.agent_loop_manager.configure_rolling_admission(paused=True)
            logger.info("paused rolling trajectory admission for actor update: %s", status)
            if getattr(self, "_adaptive_mixed_enabled", False):
                from neos.workflow.credit_prefetch import close_step

                close_step(self.global_steps, self.replay_buffer.last_selected_trajectory_keys)
                self._adaptive_capture_next_round()
                self.checkpoint_manager.abort_replicas()
                self.checkpoint_manager.sleep_replicas()
                settle_seconds = float(os.getenv("VERL_ROLLOUT_SLEEP_SETTLE_SECONDS", "2"))
                if settle_seconds:
                    time.sleep(settle_seconds)
                return
            (
                self._pending_stale_rerun_batch,
                rerun_report,
            ) = self._prepare_stale_incomplete_group_reruns()
            if rerun_report["groups"]:
                logger.info("stale incomplete rerun boundary report: %s", rerun_report)
        self.checkpoint_manager.abort_replicas()
        
        self.checkpoint_manager.sleep_replicas()
        
        
        
        
        
        settle_seconds = float(os.getenv("VERL_ROLLOUT_SLEEP_SETTLE_SECONDS", "2"))
        if settle_seconds < 0:
            raise ValueError("VERL_ROLLOUT_SLEEP_SETTLE_SECONDS must be non-negative")
        if settle_seconds:
            time.sleep(settle_seconds)

    def on_train_end(self):
        if getattr(self, "_rolling_window_enabled", False):
            self.agent_loop_manager.configure_rolling_admission(paused=True)
