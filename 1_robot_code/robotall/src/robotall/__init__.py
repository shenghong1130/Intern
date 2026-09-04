"""Robot contest SDK."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    # motion
    "act",
    "capture_image",
    "turn_left",
    "turn_right",
    "turn_ahead",
    "raise_head",
    "lower_head",
    "reset_head",
    # nfc
    "register_robot",
    "send_request",
    "clear_request_mailbox",
    "clear_response_mailbox",
    "reset_info",
]


def __getattr__(name: str):
    if name in {
        "act",
        "capture_image",
        "turn_left",
        "turn_right",
        "turn_ahead",
        "raise_head",
        "lower_head",
        "reset_head",
    }:
        from .basic_action import (
            act,
            capture_image,
            turn_left,
            turn_right,
            turn_ahead,
            raise_head,
            lower_head,
            reset_head,
        )

        return {
            "act": act,
            "capture_image": capture_image,
            "turn_left": turn_left,
            "turn_right": turn_right,
            "turn_ahead": turn_ahead,
            "raise_head": raise_head,
            "lower_head": lower_head,
            "reset_head": reset_head,
        }[name]

    if name in {
        "register_robot",
        "send_request",
        "clear_request_mailbox",
        "clear_response_mailbox",
        "reset_info",
    }:
        from .robot_tag import (
            register_robot,
            send_request,
            clear_request_mailbox,
            clear_response_mailbox,
            reset_info,
        )

        return {
            "register_robot": register_robot,
            "send_request": send_request,
            "clear_request_mailbox": clear_request_mailbox,
            "clear_response_mailbox": clear_response_mailbox,
            "reset_info": reset_info,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")