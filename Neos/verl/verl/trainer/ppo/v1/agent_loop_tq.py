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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import logging
import os
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class RollingTrajectoryAdmissionState:
    """Deterministic state machine for a fixed-size rolling rollout window.

    A *group* is one prompt and its ``rollout.n`` sessions.  Sessions are
    admitted one at a time, but every session of an earlier group must have
    been admitted before the next group may enter the window.  Releasing one
    completed session therefore opens exactly one slot for the next queued
    session without waiting for a whole GRPO group (or batch) to finish.

    The state machine is intentionally independent from Ray so its ordering
    and capacity invariants can be unit-tested without a cluster.
    """

    def __init__(self, capacity: int, *, global_steps: int = 0, paused: bool = True):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"rolling trajectory capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.global_steps = int(global_steps)
        self.paused = bool(paused)
        self.active = 0
        self._groups: dict[str, dict[str, Any]] = {}
        self._ordered_uids: list[str] = []
        self._next_group_index = 0
        self._leases: dict[str, tuple[str, int]] = {}
        self._lease_sequence = 0

    def register_groups(self, groups: list[dict[str, Any]]) -> int:
        """Register new groups; later calls may only append queue positions."""
        if not groups:
            return 0
        additions: list[tuple[int, str, int]] = []
        seen_orders = {int(group["order"]) for group in self._groups.values()}
        max_existing_order = max(seen_orders, default=-1)
        for raw in groups:
            uid = str(raw["uid"])
            order = int(raw["order"])
            expected = int(raw["expected_sessions"])
            if expected <= 0:
                raise ValueError(f"expected_sessions must be positive for uid={uid}, got {expected}")
            existing = self._groups.get(uid)
            if existing is not None:
                if int(existing["order"]) != order or int(existing["expected_sessions"]) != expected:
                    raise ValueError(f"conflicting rolling group registration for uid={uid}")
                continue
            if order in seen_orders:
                raise ValueError(f"duplicate rolling queue order {order}")
            if self._groups and order <= max_existing_order:
                raise ValueError(
                    f"new rolling queue order must append after {max_existing_order}, got {order}"
                )
            seen_orders.add(order)
            additions.append((order, uid, expected))

        for order, uid, expected in sorted(additions):
            self._groups[uid] = {
                "uid": uid,
                "order": order,
                "expected_sessions": expected,
                "admitted_sessions": set(),
                "completed_sessions": set(),
                "first_admission_step": None,
                "cancelled": False,
            }
            self._ordered_uids.append(uid)
        return len(additions)

    def set_global_steps(self, global_steps: int) -> None:
        self.global_steps = int(global_steps)

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def _advance_group_cursor(self) -> None:
        while self._next_group_index < len(self._ordered_uids):
            group = self._groups[self._ordered_uids[self._next_group_index]]
            if group["cancelled"]:
                self._next_group_index += 1
                continue
            if len(group["admitted_sessions"]) < group["expected_sessions"]:
                return
            self._next_group_index += 1

    def can_admit(self, uid: str, session_id: int) -> bool:
        uid = str(uid)
        session_id = int(session_id)
        self._advance_group_cursor()
        if self.paused or self.active >= self.capacity:
            return False
        if self._next_group_index >= len(self._ordered_uids):
            return False
        if self._ordered_uids[self._next_group_index] != uid:
            return False
        group = self._groups[uid]
        return not group["cancelled"] and session_id not in group["admitted_sessions"]

    def is_cancelled(self, uid: str) -> bool:
        group = self._groups.get(str(uid))
        return bool(group is not None and group["cancelled"])

    def cancel_groups(self, uids: list[str]) -> dict[str, int]:
        cancelled = 0
        active_leases = 0
        requested = {str(uid) for uid in uids}
        for uid in requested:
            group = self._groups.get(uid)
            if group is None or group["cancelled"]:
                continue
            group["cancelled"] = True
            cancelled += 1
        for uid, _session_id in self._leases.values():
            if uid in requested:
                active_leases += 1
        self._advance_group_cursor()
        return {"cancelled_groups": cancelled, "active_leases": active_leases}

    def admit(self, uid: str, session_id: int) -> dict[str, Any]:
        uid = str(uid)
        session_id = int(session_id)
        if not self.can_admit(uid, session_id):
            raise RuntimeError(f"rolling session is not currently admissible: uid={uid} session={session_id}")
        group = self._groups[uid]
        first_group_admission = not group["admitted_sessions"]
        group["admitted_sessions"].add(session_id)
        if first_group_admission:
            group["first_admission_step"] = self.global_steps
        self.active += 1
        self._lease_sequence += 1
        lease_id = f"{uid}:{session_id}:{self._lease_sequence}"
        self._leases[lease_id] = (uid, session_id)
        self._advance_group_cursor()
        return {
            "lease_id": lease_id,
            "global_steps": self.global_steps,
            "first_group_admission": first_group_admission,
            "active": self.active,
            "capacity": self.capacity,
        }

    def release(self, lease_id: str) -> dict[str, Any]:
        lease = self._leases.pop(str(lease_id), None)
        if lease is None:
            return {"released": False, "active": self.active, "capacity": self.capacity}
        uid, session_id = lease
        group = self._groups[uid]
        group["completed_sessions"].add(session_id)
        self.active -= 1
        if self.active < 0:
            raise RuntimeError("rolling trajectory active count became negative")
        return {
            "released": True,
            "active": self.active,
            "capacity": self.capacity,
            "group_completed": len(group["completed_sessions"]) == group["expected_sessions"],
            "uid": uid,
        }

    def status(self, waiting: int = 0) -> dict[str, Any]:
        admitted_groups = sum(
            not group["cancelled"]
            and len(group["admitted_sessions"]) == group["expected_sessions"]
            for group in self._groups.values()
        )
        completed_groups = sum(
            not group["cancelled"]
            and len(group["completed_sessions"]) == group["expected_sessions"]
            for group in self._groups.values()
        )
        return {
            "capacity": self.capacity,
            "active": self.active,
            "paused": self.paused,
            "global_steps": self.global_steps,
            "registered_groups": len(self._groups),
            "fully_admitted_groups": admitted_groups,
            "completed_groups": completed_groups,
            "cancelled_groups": sum(group["cancelled"] for group in self._groups.values()),
            "waiting_sessions": int(waiting),
            "next_group_order": (
                self._groups[self._ordered_uids[self._next_group_index]]["order"]
                if self._next_group_index < len(self._ordered_uids)
                else None
            ),
        }


class AdaptiveMixedGroupAdmissionState:
    """Two-batch question window with outcome-aware bounded resampling.

    Each registered question owns ``base_sessions`` ordinary GRPO rollouts and
    at most ``max_sessions - base_sessions`` additional rollouts.  The first
    ``group_window`` questions are active. Freed trajectory slots prefer a
    base-complete homogeneous group, otherwise the active question with the
    fewest completed trajectories. A group becomes trainable as soon as it has
    at least ``base_sessions`` outcomes containing both reward classes.
    """

    def __init__(
        self,
        capacity: int,
        *,
        group_window: int,
        base_sessions: int,
        max_sessions: int,
        global_steps: int = 0,
        paused: bool = True,
    ):
        if capacity <= 0 or group_window <= 0 or base_sessions <= 1:
            raise ValueError("adaptive mixed-group dimensions must be positive")
        if max_sessions < base_sessions:
            raise ValueError("adaptive max_sessions must be >= base_sessions")
        self.capacity = int(capacity)
        self.group_window = int(group_window)
        self.base_sessions = int(base_sessions)
        self.max_sessions = int(max_sessions)
        self.global_steps = int(global_steps)
        self.paused = bool(paused)
        self.active = 0
        self._groups: dict[str, dict[str, Any]] = {}
        self._ordered_uids: list[str] = []
        self._activation_cursor = 0
        self._active_uids: set[str] = set()
        self._leases: dict[str, tuple[str, int]] = {}
        self._grants: set[tuple[str, int]] = set()
        self._group_grants: dict[str, set[int]] = {}
        self._lease_sequence = 0

    def register_groups(self, groups: list[dict[str, Any]]) -> int:
        additions = []
        seen_orders = {int(group["order"]) for group in self._groups.values()}
        max_order = max(seen_orders, default=-1)
        for raw in groups:
            uid = str(raw["uid"])
            order = int(raw["order"])
            expected = int(raw["expected_sessions"])
            if expected != self.max_sessions:
                raise ValueError(
                    f"adaptive uid={uid} expected_sessions={expected} != {self.max_sessions}"
                )
            if uid in self._groups:
                continue
            if order in seen_orders or (self._groups and order <= max_order):
                raise ValueError(f"adaptive queue order is not append-only: {order}")
            seen_orders.add(order)
            max_order = max(max_order, order)
            additions.append((order, uid))
        for order, uid in sorted(additions):
            self._groups[uid] = {
                "uid": uid,
                "order": order,
                "expected_sessions": self.max_sessions,
                "admitted_sessions": set(),
                "completed_sessions": {},
                "first_admission_step": None,
                "active_window": False,
                "ready": False,
                "exhausted": False,
                "cancelled": False,
            }
            self._ordered_uids.append(uid)
        self._activate_waiting_groups()
        self._fill_grants()
        return len(additions)

    def set_global_steps(self, global_steps: int) -> None:
        self.global_steps = int(global_steps)

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)
        self._fill_grants()

    def _active_window_count(self) -> int:
        return len(self._active_uids)

    def _activate_waiting_groups(self) -> None:
        while (
            self._active_window_count() < self.group_window
            and self._activation_cursor < len(self._ordered_uids)
        ):
            uid = self._ordered_uids[self._activation_cursor]
            self._activation_cursor += 1
            group = self._groups[uid]
            if group["cancelled"]:
                continue
            group["active_window"] = True
            self._active_uids.add(uid)

    @staticmethod
    def _mixed(group: dict[str, Any]) -> bool:
        outcomes = list(group["completed_sessions"].values())
        return bool(outcomes) and any(outcomes) and not all(outcomes)

    def _remove_group_grants(self, uid: str) -> None:
        for session_id in self._group_grants.pop(uid, ()):
            self._grants.discard((uid, session_id))

    def _refresh_group(self, group: dict[str, Any]) -> None:
        completed = len(group["completed_sessions"])
        if completed >= self.base_sessions and self._mixed(group):
            group["ready"] = True
            self._remove_group_grants(group["uid"])
        elif completed >= self.max_sessions:
            group["exhausted"] = True
            group["active_window"] = False
            self._active_uids.discard(group["uid"])
            self._remove_group_grants(group["uid"])
            self._activate_waiting_groups()

    def _next_session_id(self, group: dict[str, Any]) -> int | None:
        occupied = set(group["admitted_sessions"])
        occupied.update(self._group_grants.get(group["uid"], ()))
        for session_id in range(self.max_sessions):
            if session_id not in occupied:
                return session_id
        return None

    def _choose_group(self) -> dict[str, Any] | None:
        candidates = [
            group
            for group in (self._groups[uid] for uid in self._active_uids)
            if group["active_window"]
            and not group["cancelled"]
            and not group["ready"]
            and not group["exhausted"]
            and self._next_session_id(group) is not None
        ]
        if not candidates:
            return None
        homogeneous = [
            group
            for group in candidates
            if len(group["completed_sessions"]) >= self.base_sessions
            and not self._mixed(group)
        ]
        pool = homogeneous or candidates
        return min(
            pool,
            key=lambda group: (
                len(group["completed_sessions"]),
                len(group["admitted_sessions"])
                + len(self._group_grants.get(group["uid"], ())),
                int(group["order"]),
            ),
        )

    def _fill_grants(self) -> None:
        if self.paused or self.active + len(self._grants) >= self.capacity:
            return
        self._activate_waiting_groups()
        while self.active + len(self._grants) < self.capacity:
            group = self._choose_group()
            if group is None:
                break
            session_id = self._next_session_id(group)
            if session_id is None:
                break
            self._grants.add((group["uid"], session_id))
            self._group_grants.setdefault(group["uid"], set()).add(session_id)

    def can_admit(self, uid: str, session_id: int) -> bool:
        
        
        
        return (
            not self.paused
            and self.active < self.capacity
            and (str(uid), int(session_id)) in self._grants
        )

    def is_cancelled(self, uid: str) -> bool:
        group = self._groups.get(str(uid))
        return bool(group is not None and group["cancelled"])

    def admit(self, uid: str, session_id: int) -> dict[str, Any]:
        uid, session_id = str(uid), int(session_id)
        if not self.can_admit(uid, session_id):
            raise RuntimeError(f"adaptive session is not admissible: uid={uid} session={session_id}")
        group = self._groups[uid]
        self._grants.remove((uid, session_id))
        self._group_grants[uid].remove(session_id)
        first = not group["admitted_sessions"]
        group["admitted_sessions"].add(session_id)
        if first:
            group["first_admission_step"] = self.global_steps
        self.active += 1
        self._lease_sequence += 1
        lease_id = f"{uid}:{session_id}:{self._lease_sequence}"
        self._leases[lease_id] = (uid, session_id)
        self._fill_grants()
        return {
            "lease_id": lease_id,
            "global_steps": self.global_steps,
            "first_group_admission": first,
            "active": self.active,
            "capacity": self.capacity,
        }

    def release(
        self,
        lease_id: str,
        *,
        completed: bool = True,
        reward: float | None = None,
    ) -> dict[str, Any]:
        lease = self._leases.pop(str(lease_id), None)
        if lease is None:
            return {"released": False, "active": self.active, "capacity": self.capacity}
        uid, session_id = lease
        group = self._groups[uid]
        self.active -= 1
        if self.active < 0:
            raise RuntimeError("adaptive trajectory active count became negative")
        if completed and not group["cancelled"]:
            group["completed_sessions"][session_id] = bool(float(reward or 0.0) > 0.5)
            self._refresh_group(group)
        elif not completed and not group["cancelled"]:
            
            
            group["exhausted"] = True
            group["active_window"] = False
            self._active_uids.discard(uid)
            self._remove_group_grants(uid)
            self._activate_waiting_groups()
        self._fill_grants()
        outcomes = list(group["completed_sessions"].values())
        return {
            "released": True,
            "active": self.active,
            "capacity": self.capacity,
            "uid": uid,
            "session_id": session_id,
            "completed_sessions": len(outcomes),
            "correct_sessions": sum(outcomes),
            "incorrect_sessions": len(outcomes) - sum(outcomes),
            "admitted_sessions": len(group["admitted_sessions"]),
            "ready": bool(group["ready"]),
            "exhausted": bool(group["exhausted"]),
        }

    def cancel_groups(self, uids: list[str]) -> dict[str, int]:
        requested = {str(uid) for uid in uids}
        cancelled = 0
        for uid in requested:
            group = self._groups.get(uid)
            if group is None or group["cancelled"]:
                continue
            group["cancelled"] = True
            group["active_window"] = False
            self._active_uids.discard(uid)
            self._remove_group_grants(uid)
            cancelled += 1
        self._activate_waiting_groups()
        self._fill_grants()
        return {
            "cancelled_groups": cancelled,
            "active_leases": sum(uid in requested for uid, _ in self._leases.values()),
        }

    def status(self, waiting: int = 0) -> dict[str, Any]:
        active_groups = [
            group for group in self._groups.values() if group["active_window"] and not group["cancelled"]
        ]
        return {
            "mode": "adaptive_mixed",
            "capacity": self.capacity,
            "active": self.active,
            "granted": len(self._grants),
            "paused": self.paused,
            "global_steps": self.global_steps,
            "registered_groups": len(self._groups),
            "active_window_groups": len(active_groups),
            "ready_groups": sum(group["ready"] and not group["cancelled"] for group in self._groups.values()),
            "exhausted_groups": sum(
                group["exhausted"] and not group["cancelled"] for group in self._groups.values()
            ),
            "cancelled_groups": sum(group["cancelled"] for group in self._groups.values()),
            "completed_trajectories": sum(
                len(group["completed_sessions"]) for group in self._groups.values()
            ),
            "waiting_sessions": int(waiting),
            "base_sessions": self.base_sessions,
            "max_sessions": self.max_sessions,
            "group_window": self.group_window,
        }


@ray.remote(max_concurrency=4096)
class RollingTrajectoryAdmissionController:
    """Single-event-loop coordinator with targeted, not broadcast, wakeups.

    Waiting acquire RPCs suspend on one future per session. Control methods
    never wait for a Condition lock held/queued by the whole dataset. Do not
    put these methods into different Ray concurrency groups: state/futures
    deliberately belong to the same asyncio event loop.
    """

    def __init__(self, capacity: int, adaptive_config: dict[str, int] | None = None):
        if adaptive_config:
            self._state = AdaptiveMixedGroupAdmissionState(
                capacity,
                group_window=int(adaptive_config["group_window"]),
                base_sessions=int(adaptive_config["base_sessions"]),
                max_sessions=int(adaptive_config["max_sessions"]),
                paused=True,
            )
        else:
            self._state = RollingTrajectoryAdmissionState(capacity, paused=True)
        self._waiters: dict[tuple[str, int], asyncio.Future] = {}
        self._waiters_by_uid: dict[str, set[tuple[str, int]]] = {}
        self._session_leases: dict[tuple[str, int], str] = {}
        self._abandoned: set[tuple[str, int]] = set()
        self._waiting = 0
        self._wakeups = 0

    def _wake_keys(self, keys) -> None:
        for key in keys:
            future = self._waiters.get(key)
            if future is not None and not future.done():
                future.set_result(None)
                self._wakeups += 1

    def _wake_granted(self) -> None:
        if self._state.paused or self._state.active >= self._state.capacity:
            return
        if isinstance(self._state, AdaptiveMixedGroupAdmissionState):
            self._wake_keys(self._state._grants)
        else:
            
            self._state._advance_group_cursor()
            cursor = self._state._next_group_index
            if cursor < len(self._state._ordered_uids):
                uid = self._state._ordered_uids[cursor]
                self._wake_keys(self._waiters_by_uid.get(uid, ()))

    def _status(self) -> dict[str, Any]:
        return {
            **self._state.status(self._waiting),
            "notification_mode": "targeted-session-v2",
            "session_wakeups": self._wakeups,
            "pending_waiters": len(self._waiters),
        }

    async def register_groups(self, groups: list[dict[str, Any]]) -> int:
        added = self._state.register_groups(groups)
        self._wake_granted()
        return added

    async def acquire(self, uid: str, session_id: int) -> dict[str, Any]:
        key = (str(uid), int(session_id))
        uid, session_id = key
        if uid not in self._state._groups:
            raise ValueError(f"unregistered rolling group: {uid}")
        group = self._state._groups[uid]
        if not 0 <= session_id < group["expected_sessions"]:
            raise ValueError(f"invalid rolling session: {key}")
        if key in self._waiters or session_id in group["admitted_sessions"]:
            raise RuntimeError(f"duplicate rolling acquisition: {key}")
        self._waiting += 1
        try:
            while True:
                if self._state.is_cancelled(uid) or key in self._abandoned:
                    
                    
                    return {"cancelled": True, "uid": uid, "session_id": session_id}
                if self._state.can_admit(uid, session_id):
                    lease = self._state.admit(uid, session_id)
                    self._session_leases[key] = lease["lease_id"]
                    self._wake_granted()
                    return lease
                future = asyncio.get_running_loop().create_future()
                self._waiters[key] = future
                self._waiters_by_uid.setdefault(uid, set()).add(key)
                await future
        finally:
            self._waiting -= 1
            self._waiters.pop(key, None)
            uid_waiters = self._waiters_by_uid.get(uid)
            if uid_waiters is not None:
                uid_waiters.discard(key)
                if not uid_waiters:
                    self._waiters_by_uid.pop(uid, None)

    def _release(self, lease_id: str, *, completed: bool, reward: float | None) -> dict[str, Any]:
        key = self._state._leases.get(lease_id)
        if isinstance(self._state, AdaptiveMixedGroupAdmissionState):
            report = self._state.release(lease_id, completed=completed, reward=reward)
        else:
            report = self._state.release(lease_id)
        if key is not None:
            self._session_leases.pop(key, None)
        self._wake_granted()
        return report

    async def release(
        self,
        lease_id: str,
        *,
        completed: bool = True,
        reward: float | None = None,
    ) -> dict[str, Any]:
        return self._release(lease_id, completed=completed, reward=reward)

    async def abandon_session(self, uid: str, session_id: int) -> dict[str, Any]:
        """Close a cancelled caller's acquisition, including the RPC handoff race.

        An acquire result may already have left the actor when its caller is
        cancelled. Retain a tombstone so a late acquire RPC cannot leak a slot.
        Only the caller that has not started its agent loop uses this method.
        """
        key = (str(uid), int(session_id))
        self._abandoned.add(key)
        lease_id = self._session_leases.get(key)
        if lease_id is not None:
            return self._release(lease_id, completed=False, reward=None)
        
        
        report = await self.cancel_groups([key[0]])
        return {**report, "released": False}

    async def cancel_groups(self, uids: list[str]) -> dict[str, int]:
        report = self._state.cancel_groups(uids)
        for uid in uids:
            self._wake_keys(self._waiters_by_uid.get(str(uid), ()))
        self._wake_granted()
        return report

    async def configure(self, *, global_steps: int | None = None, paused: bool | None = None) -> dict[str, Any]:
        if global_steps is not None:
            self._state.set_global_steps(global_steps)
        if paused is not None:
            self._state.set_paused(paused)
        self._wake_granted()
        return self._status()

    async def get_status(self) -> dict[str, Any]:
        return self._status()


def _admission_control_get(refs, operation: str):
    timeout = float(os.getenv("VERL_ROLLING_ADMISSION_CONTROL_TIMEOUT_S", "60"))
    if timeout <= 0:
        raise ValueError("Admission control timeout must be positive")
    try:
        return ray.get(refs, timeout=timeout)
    except ray.exceptions.GetTimeoutError as exc:
        raise TimeoutError(
            f"Rolling admission {operation} did not finish within {timeout}s; "
            "aborting this run so the node owners can restore idle GPU guards"
        ) from exc


def _position_ids_for_transfer_queue(position_ids: torch.Tensor) -> torch.Tensor:
    """Store M-RoPE ids with sequence as TransferQueue's leading axis.

    ``AgentLoopWorker._compute_position_ids`` returns ``(1, axes, seq)``.
    After squeezing the per-sample batch axis this is ``(axes, seq)``, while
    TransferQueue treats dimension zero as the variable token dimension.  A
    direct put would therefore persist only 3/4 logical tokens.  Use
    ``(seq, axes)`` on the wire; the FSDP engine restores axis-first layout.
    """
    if position_ids.ndim == 1:
        return position_ids
    if position_ids.ndim != 2 or position_ids.shape[0] not in (3, 4):
        raise ValueError(
            "Expected text position ids (seq,) or M-RoPE ids (3/4,seq); got "
            f"{tuple(position_ids.shape)}"
        )
    return position_ids.transpose(0, 1).contiguous()


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()
        self.prompt_tasks: dict[str, asyncio.Task[Any]] = {}

    async def generate_sequences(
        self,
        batch: TensorDict,
        admission_controller: ray.actor.ActorHandle | None = None,
    ) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        
        for i in range(len(batch)):
            
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            
            uid = str(prompt["uid"])
            previous = self.prompt_tasks.get(uid)
            if previous is not None and not previous.done():
                previous.cancel()
                await asyncio.gather(previous, return_exceptions=True)
            task = asyncio.create_task(
                self._run_prompt(
                    prompt,
                    sampling_params,
                    trajectory=trajectory_info[i],
                    trace=trace_this_sample,
                    admission_controller=admission_controller,
                )
            )
            self.background_tasks.add(task)
            self.prompt_tasks[uid] = task
            task.add_done_callback(self.background_tasks.discard)
            task.add_done_callback(
                lambda done, prompt_uid=uid: (
                    self.prompt_tasks.pop(prompt_uid, None)
                    if self.prompt_tasks.get(prompt_uid) is done
                    else None
                )
            )

    async def cancel_prompts(self, uids: list[str]) -> int:
        """Cancel complete prompt groups, including waiting/running sessions."""
        tasks = []
        for uid in map(str, uids):
            task = self.prompt_tasks.get(uid)
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    async def _run_admitted_session(
        self,
        *,
        uid: str,
        partition_id: str,
        queue_order: int,
        session_id: int,
        prompt: dict,
        sampling_params: dict,
        trajectory: dict,
        trace: bool,
        admission_controller: ray.actor.ActorHandle | None,
    ) -> None:
        lease_id = None
        acquisition_started = False
        session_completed = False
        session_reward = 0.0
        session_prompt = dict(prompt)
        session_trajectory = dict(trajectory)
        try:
            if admission_controller is not None:
                acquisition_started = True
                lease = await admission_controller.acquire.remote(uid, session_id)
                if lease.get("cancelled"):
                    acquisition_started = False
                    raise asyncio.CancelledError(f"rolling group cancelled before admission: uid={uid}")
                lease_id = lease["lease_id"]
                admitted_step = int(lease["global_steps"])
                session_prompt["global_steps"] = admitted_step
                session_trajectory["step"] = admitted_step
                if lease["first_group_admission"]:
                    await tq.async_kv_put(
                        key=uid,
                        partition_id=partition_id,
                        tag={
                            "global_steps": admitted_step,
                            "rollout_queue_order": queue_order,
                            "rolling_first_admitted": True,
                        },
                    )
            try:
                result = await self._run_agent_loop(
                    sampling_params,
                    trajectory=session_trajectory,
                    trace=trace,
                    session_id=session_id,
                    **session_prompt,
                )
                if isinstance(result, dict):
                    session_reward = float(result.get("reward_score") or 0.0)
                session_completed = True
            except asyncio.CancelledError:
                raise
            except BaseException:
                raise
        finally:
            if admission_controller is not None and acquisition_started and lease_id is None:
                
                
                
                await asyncio.wait_for(
                    asyncio.shield(admission_controller.abandon_session.remote(uid, session_id)),
                    timeout=float(os.getenv("VERL_ROLLING_ADMISSION_CONTROL_TIMEOUT_S", "60")),
                )
            if admission_controller is not None and lease_id is not None:
                report = await asyncio.wait_for(
                    asyncio.shield(
                        admission_controller.release.remote(
                            lease_id,
                            completed=session_completed,
                            reward=session_reward,
                        )
                    ),
                    timeout=float(os.getenv("VERL_ROLLING_ADMISSION_CONTROL_TIMEOUT_S", "60")),
                )
                if report.get("ready") is not None:
                    await tq.async_kv_put(
                        key=uid,
                        partition_id=partition_id,
                        tag={
                            "status": "finished" if report["ready"] else "running",
                            "adaptive_completed_sessions": report["completed_sessions"],
                            "adaptive_correct_sessions": report["correct_sessions"],
                            "adaptive_incorrect_sessions": report["incorrect_sessions"],
                            "adaptive_admitted_sessions": report["admitted_sessions"],
                            "adaptive_ready": report["ready"],
                            "adaptive_exhausted": report["exhausted"],
                        },
                    )

    async def _run_prompt(
        self,
        prompt: dict,
        sampling_params: dict,
        trajectory: dict,
        trace: bool = False,
        admission_controller: ray.actor.ActorHandle | None = None,
    ) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = []
        try:
            
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            n = int(n.item()) if isinstance(n, torch.Tensor) else int(n)
            do_sample = prompt.pop("__do_sample__", True)
            queue_order_value = prompt.pop("__rollout_queue_order__", -1)
            queue_order = (
                int(queue_order_value.item())
                if isinstance(queue_order_value, torch.Tensor)
                else int(queue_order_value)
            )

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_admitted_session(
                        uid=str(uid),
                        partition_id=partition_id,
                        queue_order=queue_order,
                        session_id=i,
                        prompt=prompt,
                        sampling_params=run_sampling_params,
                        trajectory=trajectory,
                        trace=trace,
                        admission_controller=admission_controller,
                    )
                )
                tasks.append(task)

            
            
            session_errors = await _settle_session_tasks(tasks)
            adaptive = os.getenv("VERL_ADAPTIVE_MIXED_GROUP_WINDOW", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if adaptive:
                
                
                
                return
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        
        
        
        
        
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)
            position_ids = _position_ids_for_transfer_queue(position_ids)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            
            field.pop("multi_modal_data", None)
            
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    
                    
                    
                    "global_steps": kwargs["global_steps"],
                    
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                    "reward_score": float(output.reward_score or 0.0),
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )
        return {
            "reward_score": float(final_output.reward_score or 0.0),
            "uid": str(uid),
            "session_id": int(session_id),
        }


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)
        self.rolling_admission_enabled = os.getenv(
            "VERL_ROLLING_TRAJECTORY_WINDOW", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.adaptive_mixed_enabled = os.getenv(
            "VERL_ADAPTIVE_MIXED_GROUP_WINDOW", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self.adaptive_mixed_enabled and not self.rolling_admission_enabled:
            raise ValueError("adaptive mixed-group window requires rolling admission")
        self._rolling_fallback_order = 0
        self.rolling_admission_controller = None
        if self.rolling_admission_enabled:
            rollout_n = int(self.config.actor_rollout_ref.rollout.n)
            train_batch_size = int(self.config.data.train_batch_size)
            if self.adaptive_mixed_enabled:
                group_window = int(
                    os.getenv("VERL_ADAPTIVE_GROUP_WINDOW", str(2 * train_batch_size))
                )
                max_sessions = int(
                    os.getenv(
                        "VERL_ADAPTIVE_MAX_SESSIONS",
                        str(rollout_n + max(1, rollout_n // 2)),
                    )
                )
                if group_window != 2 * train_batch_size:
                    raise ValueError("adaptive group window must equal 2 * train_batch_size")
                if max_sessions != rollout_n + rollout_n // 2:
                    raise ValueError("adaptive max sessions must equal 1.5 * rollout.n")
                capacity = group_window * rollout_n
                adaptive_config = {
                    "group_window": group_window,
                    "base_sessions": rollout_n,
                    "max_sessions": max_sessions,
                }
                inflight_batches = 2
            else:
                inflight_batches = int(os.getenv("VERL_ROLLING_INFLIGHT_BATCHES", "1"))
                if inflight_batches <= 0:
                    raise ValueError(
                        f"VERL_ROLLING_INFLIGHT_BATCHES must be positive, got {inflight_batches}"
                    )
                capacity = train_batch_size * rollout_n * inflight_batches
                adaptive_config = None
            self.rolling_admission_rpc_concurrency = int(
                os.getenv("VERL_ROLLING_ADMISSION_RPC_CONCURRENCY", "65536")
            )
            if self.rolling_admission_rpc_concurrency < 4096:
                raise ValueError("Admission RPC concurrency must be at least4096")
            self.rolling_admission_controller = RollingTrajectoryAdmissionController.options(
                name="rolling_trajectory_admission_controller",
                max_concurrency=self.rolling_admission_rpc_concurrency,
            ).remote(capacity, adaptive_config)
            logger.info(
                "created rolling trajectory admission controller: capacity=%s groups_per_update=%s "
                "rollout_n=%s inflight_batches=%s inflight_group_equivalent=%s",
                capacity,
                train_batch_size,
                rollout_n,
                inflight_batches,
                train_batch_size * inflight_batches,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        validate = False
        if "validate" in prompts:
            value = prompts["validate"]
            validate = bool(value.data if isinstance(value, NonTensorData) else value)

        admission_controller = None
        if self.rolling_admission_controller is not None and not validate:
            if "__rollout_queue_order__" not in prompts:
                start = self._rolling_fallback_order
                prompts["__rollout_queue_order__"] = torch.arange(start, start + len(prompts), dtype=torch.int64)
            orders = prompts["__rollout_queue_order__"]
            if isinstance(orders, torch.Tensor):
                queue_orders = [int(value) for value in orders.tolist()]
            elif isinstance(orders, NonTensorStack):
                queue_orders = [int(value.data) for value in orders]
            else:
                raise TypeError(f"Unsupported rolling queue order type: {type(orders)}")
            self._rolling_fallback_order = max(self._rolling_fallback_order, max(queue_orders, default=-1) + 1)

            rollout_n = int(self.config.actor_rollout_ref.rollout.n)
            if "__rollout_n__" in prompts:
                rollout_values = prompts["__rollout_n__"]
                if isinstance(rollout_values, torch.Tensor):
                    expected_sessions = [int(value) for value in rollout_values.tolist()]
                elif isinstance(rollout_values, NonTensorStack):
                    expected_sessions = [int(value.data) for value in rollout_values]
                elif isinstance(rollout_values, NonTensorData):
                    expected_sessions = [int(rollout_values.data)] * len(prompts)
                else:
                    raise TypeError(f"Unsupported rollout.n override type: {type(rollout_values)}")
            else:
                expected_sessions = [rollout_n] * len(prompts)
            groups = [
                {
                    "uid": str(uid),
                    "order": order,
                    "expected_sessions": expected,
                }
                for uid, order, expected in zip(
                    list(prompts["uid"]), queue_orders, expected_sessions, strict=True
                )
            ]
            required_rpc_slots = 2 * sum(expected_sessions) + 1024
            if required_rpc_slots > self.rolling_admission_rpc_concurrency:
                raise ValueError(
                    f"Queued admission waiters require at least{required_rpc_slots} RPC slots, "
                    f"configured{self.rolling_admission_rpc_concurrency}; increase "
                    "VERL_ROLLING_ADMISSION_RPC_CONCURRENCY before dispatch. "
                    "This does not change actual rollout admission capacity."
                )
            
            
            
            tq.kv_batch_put(
                keys=[group["uid"] for group in groups],
                partition_id="train",
                tags=[{"rollout_queue_order": group["order"]} for group in groups],
            )
            _admission_control_get(
                self.rolling_admission_controller.register_groups.remote(groups), "register_groups"
            )
            admission_controller = self.rolling_admission_controller

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk, admission_controller)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )

    def configure_rolling_admission(
        self,
        *,
        global_steps: int | None = None,
        paused: bool | None = None,
    ) -> dict[str, Any] | None:
        if self.rolling_admission_controller is None:
            return None
        return _admission_control_get(
            self.rolling_admission_controller.configure.remote(
                global_steps=global_steps,
                paused=paused,
            ),
            "configure",
        )

    def get_rolling_admission_status(self) -> dict[str, Any] | None:
        if self.rolling_admission_controller is None:
            return None
        return _admission_control_get(self.rolling_admission_controller.get_status.remote(), "get_status")

    def cancel_prompts(self, uids: list[str]) -> int:
        """Cancel matching prompt-group tasks on all agent-loop workers."""
        if not uids:
            return 0
        if self.rolling_admission_controller is not None:
            _admission_control_get(self.rolling_admission_controller.cancel_groups.remote(uids), "cancel_groups")
        return sum(
            ray.get(
                [worker.cancel_prompts.remote(uids) for worker in self.agent_loop_workers]
            )
        )
