#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic, safety-gated TonyPi/Worker flower interaction."""

import time
from typing import Callable, Optional

from .models import InteractionPoseCheck, WorkerChangeResult


class RobotInteractionClient:
    """Execute the physical left-hand NFC transaction from the proven notebook flow.

    The caller supplies a pose gate. It is checked once before any arm motion and
    again immediately before ``send_request`` so a state-machine bug cannot call
    the Worker while the robot is not aligned.
    """

    def __init__(
        self,
        team: Optional[str],
        secret: Optional[str],
        robot_id: Optional[str],
        config: dict,
        dry_run: bool = False,
        skip_change: bool = False,
        event_callback: Optional[Callable] = None,
        phase_callback: Optional[Callable] = None,
        act_fn: Optional[Callable] = None,
        send_request_fn: Optional[Callable] = None,
    ):
        self.team = team
        self.secret = secret
        self.robot_id = robot_id
        self.cfg = config["interaction"]
        self.dry_run = bool(dry_run)
        self.skip_change = bool(skip_change)
        self.event_callback = event_callback
        self.phase_callback = phase_callback
        self._act = act_fn
        self._send_request = send_request_fn
        self.transaction_active = False
        self.left_hand_lifted = False

    def _emit(self, name: str, **data) -> None:
        if self.event_callback is not None:
            self.event_callback(name, **data)

    def _phase(self, phase: str, **data) -> None:
        if self.phase_callback is not None:
            self.phase_callback(phase, **data)

    def _ensure_robotall(self) -> None:
        if self._act is not None and self._send_request is not None:
            return
        from robotall import act, send_request

        self._act = act
        self._send_request = send_request

    def _credentials_valid(self) -> bool:
        return bool(self.team and self.secret and self.robot_id)

    def change_flower(
        self,
        *,
        screen_id: int,
        worker_id: Optional[int],
        from_flower: str,
        to_flower: str,
        safety_gate: Callable[[], InteractionPoseCheck],
    ) -> WorkerChangeResult:
        if worker_id is None:
            return WorkerChangeResult(success=False, error="worker_mapping_missing")
        if not from_flower or not to_flower or from_flower == to_flower:
            return WorkerChangeResult(success=False, worker_id=worker_id, error="invalid_flower_transition")

        first_check = safety_gate()
        if not first_check.ready:
            self._emit(
                "interaction_safety_gate_blocked",
                screen_id=screen_id,
                worker_id=worker_id,
                stage="before_lift",
                check=first_check.as_dict(),
            )
            return WorkerChangeResult(success=False, worker_id=worker_id, error="interaction_pose_invalid")

        request_meta = {
            "screen_id": int(screen_id),
            "worker_id": int(worker_id),
            "from_flower": from_flower,
            "to_flower": to_flower,
        }
        self.transaction_active = True
        self._phase("transaction_start", **request_meta)
        try:
            if self.dry_run or self.skip_change:
                self._phase("stand", simulated=True, **request_meta)
                self._phase("left_hand_lifted", simulated=True, **request_meta)
                final_check = safety_gate()
                if not final_check.ready:
                    self._emit(
                        "interaction_safety_gate_blocked",
                        stage="before_simulated_send_request",
                        check=final_check.as_dict(),
                        **request_meta,
                    )
                    return WorkerChangeResult(
                        success=False,
                        simulated=True,
                        worker_id=worker_id,
                        error="interaction_pose_invalid_after_lift",
                    )
                self._phase("worker_request_sent", simulated=True, **request_meta)
                self._phase("worker_response", simulated=True, ok=True, **request_meta)
                self._emit("interaction_simulated", skip_change=self.skip_change, **request_meta)
                return WorkerChangeResult(
                    success=True,
                    simulated=True,
                    worker_id=worker_id,
                    response={"ok": True, "simulated": True},
                )

            if not self._credentials_valid():
                return WorkerChangeResult(success=False, worker_id=worker_id, error="robot_credentials_missing")
            self._ensure_robotall()

            self._act("stand")
            self._phase("stand", **request_meta)
            self._act("lift_left_hand", stand=False)
            self.left_hand_lifted = True
            self._phase("left_hand_lifted", **request_meta)
            time.sleep(float(self.cfg.get("left_hand_settle_s", 0.5)))

            final_check = safety_gate()
            if not final_check.ready:
                self._emit(
                    "interaction_safety_gate_blocked",
                    stage="before_send_request",
                    check=final_check.as_dict(),
                    **request_meta,
                )
                return WorkerChangeResult(success=False, worker_id=worker_id, error="interaction_pose_invalid_after_lift")

            self._phase("worker_request_sent", **request_meta)
            result = self._send_request(
                team=self.team,
                secret=self.secret,
                robot_id=self.robot_id,
                worker_id=int(worker_id),
                from_flower=from_flower,
                to_flower=to_flower,
                clear_first=True,
                read_response=True,
                wait_response=True,
                scan_timeout_s=float(self.cfg.get("flower_change_scan_timeout_s", 0.0)),
                verbose_wait=True,
            )
            ok = bool(result.get("ok"))
            self._phase("worker_response", ok=ok, result=result, **request_meta)
            return WorkerChangeResult(
                success=ok,
                worker_id=worker_id,
                response=dict(result),
                error="" if ok else "worker_response_not_ok",
            )
        except Exception as exc:
            self._emit("interaction_exception", error=str(exc), **request_meta)
            return WorkerChangeResult(success=False, worker_id=worker_id, error=str(exc))
        finally:
            if not (self.dry_run or self.skip_change):
                try:
                    self._ensure_robotall()
                    self._act("stand")
                except Exception as exc:
                    self._emit("interaction_stand_failed", error=str(exc), **request_meta)
            self.left_hand_lifted = False
            self._phase("stand", final=True, simulated=self.dry_run or self.skip_change, **request_meta)
            self.transaction_active = False
            self._phase("transaction_end", **request_meta)
