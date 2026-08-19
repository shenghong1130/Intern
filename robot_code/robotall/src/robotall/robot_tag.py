#!/usr/bin/env python3
"""Write robot contest mailbox packets to an ST25DV04 dynamic NFC tag.

Robots do not call the contest server directly. They write a fixed 64-byte
request packet into ST25DV04 blocks. Start/Worker read the same blocks with
CK156, talk to Gateway/server, then write a 64-byte response packet back.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from .st25dv04 import (
    DEFAULT_I2C_BUS,
    DEFAULT_USER_ADDR,
    MAILBOX_SIZE,
    REQUEST_KIND_ATTEMPT,
    REQUEST_KIND_IDENTITY,
    RESPONSE_STATUS_NONE,
    ST25DV04,
    uid_hex,
    pack_request_packet,
    unpack_response_packet,
)


def compact(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "v": 1,
        "kind": "robot",
        "teamName": args.team,
        "secret": args.secret,
        "robotId": args.robot_id,
        "nonce": str(int(time.time())),
    }


def build_attempt(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_identity(args)
    payload.update(
        {
            "kind": "attempt",
            "workerId": args.worker_id,
            "fromFlower": args.from_flower,
            "toFlower": args.to_flower,
        }
    )
    return payload


def response_matches(
    response: dict[str, Any],
    *,
    seq: int,
    worker_id: int | None,
    require_status: bool = True,
) -> bool:
    if not response.get("valid"):
        return False
    if int(response.get("seq", -1)) != (seq & 0xFF):
        return False
    if require_status and int(response.get("status", RESPONSE_STATUS_NONE)) == RESPONSE_STATUS_NONE:
        return False
    if worker_id is not None and int(response.get("workerId", -1)) != worker_id:
        return False
    return True


def wait_for_response(
    tag: ST25DV04,
    *,
    seq: int,
    worker_id: int | None,
    scan_timeout_s: float,
    timeout_s: float,
    poll_interval_s: float,
    progress_interval_s: float,
    write_quiet_s: float,
    overall_timeout_s: float = 0.0,
) -> tuple[dict[str, Any] | None, float, float | None, int]:
    start = time.monotonic()
    next_progress = start + progress_interval_s if progress_interval_s > 0 else float("inf")
    polls = 0
    scan_seen_at: float | None = None
    while True:
        now = time.monotonic()
        elapsed = now - start
        if overall_timeout_s > 0 and elapsed >= overall_timeout_s:
            final_elapsed = None if scan_seen_at is None else now - scan_seen_at
            return None, elapsed, final_elapsed, polls
        if scan_seen_at is None:
            if scan_timeout_s > 0 and elapsed > scan_timeout_s:
                return None, elapsed, None, polls
        elif now - scan_seen_at > timeout_s:
            return None, elapsed, now - scan_seen_at, polls

        try:
            response = unpack_response_packet(tag.read_response())
            polls += 1
        except OSError as error:
            polls += 1
            now = time.monotonic()
            if now >= next_progress:
                phase = "scan" if scan_seen_at is None else "final"
                final_elapsed = 0.0 if scan_seen_at is None else now - scan_seen_at
                print(
                    f"waiting {phase} elapsed_total={now - start:.3f}s "
                    f"elapsed_final={final_elapsed:.3f}s seq={seq & 0xFF} "
                    f"i2c_busy={error}"
                )
                next_progress = now + progress_interval_s
            time.sleep(poll_interval_s)
            continue
        if response_matches(response, seq=seq, worker_id=worker_id, require_status=True):
            total_elapsed = time.monotonic() - start
            scan_elapsed = None if scan_seen_at is None else time.monotonic() - scan_seen_at
            return response, total_elapsed, scan_elapsed, polls

        same_seq = int(response.get("seq", -1)) == (seq & 0xFF)
        same_worker = worker_id is None or int(response.get("workerId", -1)) == worker_id
        same_response_write = same_seq and same_worker and response.get("magic") == "RBS1"
        if same_response_write and not response.get("valid"):
            if scan_seen_at is None:
                scan_seen_at = time.monotonic()
                print(
                    f"response write detected seq={seq & 0xFF} "
                    f"elapsed_total={scan_seen_at - start:.3f}s"
                )
            time.sleep(max(poll_interval_s, write_quiet_s))
            continue

        if response_matches(response, seq=seq, worker_id=worker_id, require_status=False):
            time.sleep(poll_interval_s)
            continue

        now = time.monotonic()
        if now >= next_progress:
            status = response.get("status")
            got_seq = response.get("seq")
            valid = 1 if response.get("valid") else 0
            phase = "scan" if scan_seen_at is None else "final"
            final_elapsed = 0.0 if scan_seen_at is None else now - scan_seen_at
            print(
                f"waiting {phase} elapsed_total={now - start:.3f}s "
                f"elapsed_final={final_elapsed:.3f}s "
                f"seq={seq & 0xFF} last_valid={valid} last_seq={got_seq} last_status={status}"
            )
            next_progress = now + progress_interval_s
        time.sleep(poll_interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write robot identity/action to ST25DV04")
    parser.add_argument("--bus", type=int, default=DEFAULT_I2C_BUS)
    parser.add_argument("--addr", type=lambda value: int(value, 0), default=DEFAULT_USER_ADDR)
    parser.add_argument("--sys-addr", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--uid", action="store_true", help="read immutable ST25DV04 UID and exit")
    parser.add_argument("--team")
    parser.add_argument("--secret")
    parser.add_argument("--robot-id", default="pi-robot")
    parser.add_argument("--seq", type=lambda value: int(value, 0), default=int(time.time()) & 0xFF)
    parser.add_argument("--worker-id", type=int, help="screen/worker id for an attempt")
    parser.add_argument("--from-flower", help="claimed current flower")
    parser.add_argument("--to-flower", help="target flower to change to")
    parser.add_argument("--clear-first", action="store_true")
    parser.add_argument("--read-response", action="store_true", help="read worker/start response mailbox after writing")
    parser.add_argument("--no-wait-response", action="store_true", help="with --read-response, read once immediately instead of waiting")
    parser.add_argument("--timeout", type=float, default=1.0, help="seconds to wait for a matching final response after response writing is detected")
    parser.add_argument("--scan-timeout", type=float, default=0.0, help="seconds to wait for response writing to begin; 0 waits forever")
    parser.add_argument("--overall-timeout", type=float, default=0.0, help="hard deadline for one request attempt; 0 disables the overall deadline")
    parser.add_argument("--retries", type=int, default=2, help="number of timeout retries; total attempts are retries + 1")
    parser.add_argument("--poll-interval", type=float, default=0.10, help="seconds between response mailbox polls")
    parser.add_argument("--write-quiet", type=float, default=0.50, help="seconds to pause after detecting a partial response write")
    parser.add_argument("--verbose-wait", action="store_true", help="print periodic wait progress while polling response blocks")
    parser.add_argument("--progress-interval", type=float, default=0.50, help="seconds between wait progress prints when --verbose-wait is used")
    parser.add_argument("--i2c-retries", type=int, default=8, help="low-level I2C retries for transient RF/I2C contention")
    parser.add_argument("--i2c-retry-delay", type=float, default=0.02, help="base seconds between low-level I2C retries")
    parser.add_argument("--no-clear-response", action="store_true", help="do not clear response mailbox before each request attempt")
    parser.add_argument("--ndef", action="store_true", help="debug only: write old JSON NDEF text instead of block mailbox")
    args = parser.parse_args()

    system_address = args.sys_addr if args.sys_addr is not None else args.addr | 0x04
    tag = ST25DV04(
        bus_id=args.bus,
        address=args.addr,
        system_address=system_address,
        i2c_retries=args.i2c_retries,
        i2c_retry_delay_s=args.i2c_retry_delay,
    )
    try:
        if args.uid:
            print(f"uid={uid_hex(tag.read_uid())}")
            return

        if not args.team or not args.secret:
            parser.error("--team and --secret are required unless --uid is used")

        if args.worker_id or args.from_flower or args.to_flower:
            if not args.worker_id or not args.to_flower:
                parser.error("--worker-id and --to-flower are required for an attempt")
            payload = build_attempt(args)
            kind = REQUEST_KIND_ATTEMPT
        else:
            payload = build_identity(args)
            kind = REQUEST_KIND_IDENTITY

        if args.clear_first:
            tag.clear()
        if args.ndef:
            text = compact(payload)
            data = tag.write_ndef_text(text)
            print(f"wrote NDEF {len(data)} bytes")
            print(text)
        else:
            total_attempts = max(1, args.retries + 1)
            base_seq = args.seq & 0xFF
            expected_worker_id = args.worker_id if kind == REQUEST_KIND_ATTEMPT else None
            final_response: dict[str, Any] | None = None
            response_write_seen = False
            for attempt in range(total_attempts):
                seq = base_seq
                if not response_write_seen:
                    if not args.no_clear_response:
                        tag.write_response(b"\x00" * MAILBOX_SIZE)
                    packet = pack_request_packet(
                        kind=kind,
                        seq=seq,
                        team_name=args.team,
                        secret=args.secret,
                        robot_id=args.robot_id,
                        worker_id=args.worker_id or 0,
                        from_flower=args.from_flower,
                        to_flower=args.to_flower,
                    )
                    write_start = time.monotonic()
                    tag.write_request(packet)
                    write_elapsed = time.monotonic() - write_start
                    payload["nonce"] = str(seq)
                    print(
                        f"wrote request mailbox attempt={attempt + 1}/{total_attempts} "
                        f"seq={seq} bytes={len(packet)} write={write_elapsed:.3f}s"
                    )
                    print(compact(payload))
                else:
                    print(
                        f"continue reading response attempt={attempt + 1}/{total_attempts} "
                        f"seq={seq}"
                    )

                if not args.read_response:
                    break
                if args.no_wait_response:
                    response = unpack_response_packet(tag.read_response())
                    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
                    final_response = response
                    break

                response, total_elapsed, final_elapsed, polls = wait_for_response(
                    tag,
                    seq=seq,
                    worker_id=expected_worker_id,
                    scan_timeout_s=args.scan_timeout,
                    timeout_s=args.timeout,
                    overall_timeout_s=args.overall_timeout,
                    poll_interval_s=args.poll_interval,
                    progress_interval_s=args.progress_interval if args.verbose_wait else 0.0,
                    write_quiet_s=args.write_quiet,
                )
                if response is not None:
                    final_text = "n/a" if final_elapsed is None else f"{final_elapsed:.3f}s"
                    print(
                        f"response matched attempt={attempt + 1}/{total_attempts} "
                        f"seq={seq} elapsed_total={total_elapsed:.3f}s "
                        f"elapsed_after_scan={final_text} polls={polls}"
                    )
                    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
                    final_response = response
                    break
                print(
                    f"response timeout attempt={attempt + 1}/{total_attempts} "
                    f"seq={seq} elapsed_total={total_elapsed:.3f}s "
                    f"elapsed_after_scan={'n/a' if final_elapsed is None else f'{final_elapsed:.3f}s'} "
                    f"polls={polls}"
                )
                if final_elapsed is not None:
                    response_write_seen = True
            if args.read_response and final_response is None and not args.no_wait_response:
                raise SystemExit(2)
    finally:
        tag.close()

def send_request(
    *,
    # 必填
    team: str,
    secret: str,
    # 机器人标识
    robot_id: str = "pi-robot",
    # I2C / 硬件
    bus: int = DEFAULT_I2C_BUS,
    addr: int = DEFAULT_USER_ADDR,
    sys_addr: int | None = None,
    # 请求类型相关
    worker_id: int | None = None,
    from_flower: str | None = None,
    to_flower: str | None = None,
    seq: int | None = None,
    # 行为控制
    clear_first: bool = False,
    read_response: bool = True,
    wait_response: bool = True,
    scan_timeout_s: float = 0.0,
    timeout_s: float = 1.0,
    overall_timeout_s: float = 0.0,
    poll_interval_s: float = 0.10,
    write_quiet_s: float = 0.50,
    i2c_retries: int = 8,
    i2c_retry_delay_s: float = 0.02,
    no_clear_response: bool = False,
    verbose_wait: bool = False,
    progress_interval_s: float = 0.50,
    retries: int = 2,
) -> dict[str, Any]:
    """
    向 ST25DV04 NFC 标签写入请求，并可选等待 Worker/Start 响应。

    返回结构：
    {
        "ok": bool,
        "request": {...},
        "response": {...} | None,
        "elapsed_total_s": float,
        "elapsed_after_scan_s": float | None,
        "polls": int,
        "attempts": int,
    }
    """

    if not team or not secret:
        raise ValueError("team and secret are required")

    seq = (seq if seq is not None else int(time.time())) & 0xFF
    system_address = sys_addr if sys_addr is not None else addr | 0x04

    tag = ST25DV04(
        bus_id=bus,
        address=addr,
        system_address=system_address,
        i2c_retries=i2c_retries,
        i2c_retry_delay_s=i2c_retry_delay_s,
    )

    try:
        # 判断请求类型
        if worker_id is not None or from_flower is not None or to_flower is not None:
            if worker_id is None or to_flower is None:
                raise ValueError("--worker-id and --to-flower are required for an attempt")
            payload = build_attempt(argparse.Namespace(
                team=team,
                secret=secret,
                robot_id=robot_id,
                worker_id=worker_id,
                from_flower=from_flower,
                to_flower=to_flower,
            ))
            kind = REQUEST_KIND_ATTEMPT
        else:
            payload = build_identity(argparse.Namespace(
                team=team,
                secret=secret,
                robot_id=robot_id,
            ))
            kind = REQUEST_KIND_IDENTITY

        if clear_first:
            tag.clear()

        total_attempts = max(1, retries + 1)
        expected_worker_id = worker_id if kind == REQUEST_KIND_ATTEMPT else None

        result = {
            "ok": False,
            "request": payload,
            "response": None,
            "elapsed_total_s": 0.0,
            "elapsed_after_scan_s": None,
            "polls": 0,
            "attempts": total_attempts,
        }

        response_write_seen = False

        for attempt in range(total_attempts):
            current_seq = seq

            if not no_clear_response:
                tag.write_response(b"\x00" * MAILBOX_SIZE)

            packet = pack_request_packet(
                kind=kind,
                seq=current_seq,
                team_name=team,
                secret=secret,
                robot_id=robot_id,
                worker_id=worker_id or 0,
                from_flower=from_flower,
                to_flower=to_flower,
            )

            write_start = time.monotonic()
            tag.write_request(packet)
            write_elapsed = time.monotonic() - write_start

            result["request"]["nonce"] = str(current_seq)

            if not read_response:
                result["ok"] = True
                break

            if not wait_response:
                resp = unpack_response_packet(tag.read_response())
                result["response"] = resp
                result["ok"] = True
                break

            resp, total_elapsed, final_elapsed, polls = wait_for_response(
                tag,
                seq=current_seq,
                worker_id=expected_worker_id,
                scan_timeout_s=scan_timeout_s,
                timeout_s=timeout_s,
                overall_timeout_s=overall_timeout_s,
                poll_interval_s=poll_interval_s,
                progress_interval_s=progress_interval_s if verbose_wait else 0.0,
                write_quiet_s=write_quiet_s,
            )

            result["elapsed_total_s"] = total_elapsed
            result["elapsed_after_scan_s"] = final_elapsed
            result["polls"] = polls

            if resp is not None:
                result["response"] = resp
                result["ok"] = True
                break

            if final_elapsed is not None:
                response_write_seen = True

        return result

    finally:
        tag.close()

def register_robot(
    *,
    team: str,
    secret: str,
    robot_id: str = "pi-robot",
    seq: int | None = None,
    bus: int = DEFAULT_I2C_BUS,
    addr: int = DEFAULT_USER_ADDR,
    sys_addr: int | None = None,
    clear_first: bool = False,
    read_response: bool = True,
    wait_response: bool = True,
    scan_timeout_s: float = 0.0,
    timeout_s: float = 1.0,
    overall_timeout_s: float = 0.0,
    poll_interval_s: float = 0.10,
    i2c_retries: int = 8,
    i2c_retry_delay_s: float = 0.02,
    verbose_wait: bool = False,
    progress_interval_s: float = 0.50,
    retries: int = 2,
) -> dict[str, Any]:
    """
    发送机器人注册（IDENTITY）消息到 ST25DV04 邮箱。

    返回结构同 send_request。
    """

    return send_request(
        team=team,
        secret=secret,
        robot_id=robot_id,
        seq=seq,
        bus=bus,
        addr=addr,
        sys_addr=sys_addr,
        clear_first=clear_first,
        read_response=read_response,
        wait_response=wait_response,
        scan_timeout_s=scan_timeout_s,
        timeout_s=timeout_s,
        overall_timeout_s=overall_timeout_s,
        poll_interval_s=poll_interval_s,
        i2c_retries=i2c_retries,
        i2c_retry_delay_s=i2c_retry_delay_s,
        verbose_wait=verbose_wait,
        progress_interval_s=progress_interval_s,
        retries=retries,
        # 关键：不传 worker_id / flower，强制 IDENTITY
    )

def clear_request_mailbox(
    *,
    bus: int = DEFAULT_I2C_BUS,
    addr: int = DEFAULT_USER_ADDR,
    sys_addr: int | None = None,
    i2c_retries: int = 8,
    i2c_retry_delay_s: float = 0.02,
) -> None:
    """清除 ST25DV04 Request Mailbox，防止被重复扫描。"""
    system_address = sys_addr if sys_addr is not None else addr | 0x04
    tag = ST25DV04(
        bus_id=bus,
        address=addr,
        system_address=system_address,
        i2c_retries=i2c_retries,
        i2c_retry_delay_s=i2c_retry_delay_s,
    )
    try:
        tag.write_request(b"\x00" * MAILBOX_SIZE)
    finally:
        tag.close()
def clear_response_mailbox(
    *,
    bus: int = DEFAULT_I2C_BUS,
    addr: int = DEFAULT_USER_ADDR,
    sys_addr: int | None = None,
    i2c_retries: int = 8,
    i2c_retry_delay_s: float = 0.02,
) -> None:
    """清除 ST25DV04 Response Mailbox，防止下次误判。"""
    system_address = sys_addr if sys_addr is not None else addr | 0x04
    tag = ST25DV04(
        bus_id=bus,
        address=addr,
        system_address=system_address,
        i2c_retries=i2c_retries,
        i2c_retry_delay_s=i2c_retry_delay_s,
    )
    try:
        tag.write_response(b"\x00" * MAILBOX_SIZE)
    finally:
        tag.close()
def reset_info():
    """清除 ST25DV04 Request/Response Mailbox，防止下次误判。"""
    clear_request_mailbox()
    clear_response_mailbox()

if __name__ == "__main__":
    main()
