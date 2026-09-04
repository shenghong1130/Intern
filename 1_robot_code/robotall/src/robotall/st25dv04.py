#!/usr/bin/env python3
"""Minimal ST25DV04 I2C helper for Raspberry Pi.

The script writes a small NDEF Text record into the ST25DV04 user EEPROM.
CK156 readers on Start/Worker can then read the tag over NFC.

Default ST25DV04 user-memory I2C address is commonly 0x53. Some boards strap a
different address; use --addr when needed.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Iterable

DEFAULT_I2C_BUS = 1
DEFAULT_USER_ADDR = 0x53
DEFAULT_SYSTEM_ADDR = DEFAULT_USER_ADDR | 0x04
DEFAULT_PAGE_SIZE = 4
DEFAULT_I2C_RETRIES = 8
DEFAULT_I2C_RETRY_DELAY_S = 0.02
BLOCK_SIZE = 4
MAX_I2C_READ_SIZE = 32
UID_SYSTEM_OFFSET = 0x0018
UID_SIZE = 8

REQUEST_START_BLOCK = 4
RESPONSE_START_BLOCK = 20
MAILBOX_BLOCK_COUNT = 16
MAILBOX_SIZE = BLOCK_SIZE * MAILBOX_BLOCK_COUNT

REQUEST_MAGIC = b"RBT1"
RESPONSE_MAGIC = b"RBS1"

FLOWER_IDS = {
    "main": 0,
    "bailianhua": 1,
    "chuju": 2,
    "hehua": 3,
    "juhua": 4,
    "lamei": 5,
    "lanhua": 6,
    "meiguihua": 7,
    "shuixianhua": 8,
    "taohua": 9,
    "yinghua": 10,
    "yuanweihua": 11,
    "zijinghua": 12,
    "unknown": 0xFF,
}

ID_FLOWERS = {value: key for key, value in FLOWER_IDS.items()}

REQUEST_KIND_IDENTITY = 1
REQUEST_KIND_ATTEMPT = 2

RESPONSE_STATUS_NONE = 0
RESPONSE_STATUS_ACCEPTED = 1
RESPONSE_STATUS_REJECTED = 2
RESPONSE_STATUS_TASK = 3


def fixed_ascii(value: str, length: int) -> bytes:
    data = value.encode("ascii", errors="ignore")[:length]
    return data + b"\x00" * (length - len(data))


def strip_fixed_ascii(data: bytes) -> str:
    return data.split(b"\x00", 1)[0].decode("ascii", errors="ignore")


def flower_id(name: str | None, default: int = 0xFF) -> int:
    if name is None:
        return default
    if name not in FLOWER_IDS:
        raise ValueError(f"unknown flower: {name}")
    return FLOWER_IDS[name]


def flower_name(value: int) -> str | None:
    return ID_FLOWERS.get(value)


def uid_hex(uid: bytes) -> str:
    return uid.hex().upper()


def canonical_uid(uid: bytes) -> bytes:
    if len(uid) == UID_SIZE and uid and uid[0] == 0xE0:
        return uid
    if len(uid) == UID_SIZE and uid and uid[-1] == 0xE0:
        return uid[::-1]
    return uid


def append_crc(payload_without_crc: bytes) -> bytes:
    if len(payload_without_crc) != MAILBOX_SIZE - 4:
        raise ValueError("mailbox payload must be 60 bytes before crc")
    crc = zlib.crc32(payload_without_crc) & 0xFFFFFFFF
    return payload_without_crc + struct.pack("<I", crc)


def verify_crc(packet: bytes) -> bool:
    if len(packet) != MAILBOX_SIZE:
        return False
    expected = struct.unpack("<I", packet[-4:])[0]
    actual = zlib.crc32(packet[:-4]) & 0xFFFFFFFF
    return expected == actual


def pack_request_packet(
    *,
    kind: int,
    seq: int,
    team_name: str,
    secret: str,
    robot_id: str,
    worker_id: int = 0,
    from_flower: str | None = None,
    to_flower: str | None = None,
) -> bytes:
    header = struct.pack(
        "<4sBBBBBB2x",
        REQUEST_MAGIC,
        1,
        kind,
        seq & 0xFF,
        worker_id & 0xFF,
        flower_id(from_flower),
        flower_id(to_flower),
    )
    body = (
        fixed_ascii(team_name, 16)
        + fixed_ascii(secret, 16)
        + fixed_ascii(robot_id, 16)
    )
    reserved = b"\x00" * (MAILBOX_SIZE - 4 - len(header) - len(body))
    return append_crc(header + body + reserved)


def unpack_request_packet(packet: bytes) -> dict[str, object]:
    packet = packet[:MAILBOX_SIZE].ljust(MAILBOX_SIZE, b"\x00")
    valid_crc = verify_crc(packet)
    magic, version, kind, seq, worker_id, from_id, to_id = struct.unpack(
        "<4sBBBBBB2x", packet[:12]
    )
    return {
        "valid": magic == REQUEST_MAGIC and valid_crc,
        "magic": magic.decode("ascii", errors="ignore"),
        "version": version,
        "kind": kind,
        "seq": seq,
        "workerId": worker_id,
        "fromFlower": flower_name(from_id),
        "toFlower": flower_name(to_id),
        "teamName": strip_fixed_ascii(packet[12:28]),
        "secret": strip_fixed_ascii(packet[28:44]),
        "robotId": strip_fixed_ascii(packet[44:60]),
        "crcOk": valid_crc,
    }


def pack_response_packet(
    *,
    status: int,
    seq: int,
    worker_id: int = 0,
    current_flower: str | None = None,
    target_flower: str | None = None,
    score: int = 0,
    remaining_sec: int = 0,
    message: str = "",
) -> bytes:
    header = struct.pack(
        "<4sBBBBBBhh",
        RESPONSE_MAGIC,
        1,
        status & 0xFF,
        seq & 0xFF,
        worker_id & 0xFF,
        flower_id(current_flower, default=0),
        flower_id(target_flower, default=0),
        int(score),
        int(remaining_sec),
    )
    body = fixed_ascii(message, MAILBOX_SIZE - 4 - len(header))
    return append_crc(header + body)


def unpack_response_packet(packet: bytes) -> dict[str, object]:
    packet = packet[:MAILBOX_SIZE].ljust(MAILBOX_SIZE, b"\x00")
    valid_crc = verify_crc(packet)
    (
        magic,
        version,
        status,
        seq,
        worker_id,
        current_id,
        target_id,
        score,
        remaining_sec,
    ) = struct.unpack("<4sBBBBBBhh", packet[:14])
    return {
        "valid": magic == RESPONSE_MAGIC and valid_crc,
        "magic": magic.decode("ascii", errors="ignore"),
        "version": version,
        "status": status,
        "seq": seq,
        "workerId": worker_id,
        "currentFlower": flower_name(current_id),
        "targetFlower": flower_name(target_id),
        "score": score,
        "remainingSec": remaining_sec,
        "message": strip_fixed_ascii(packet[14:60]),
        "crcOk": valid_crc,
    }


def ndef_text_payload(text: str, language: str = "en") -> bytes:
    text_bytes = text.encode("utf-8")
    lang_bytes = language.encode("ascii")
    if len(lang_bytes) > 63:
        raise ValueError("language code too long")
    status = len(lang_bytes)
    return bytes([status]) + lang_bytes + text_bytes


def ndef_short_text_record(text: str, language: str = "en") -> bytes:
    payload = ndef_text_payload(text, language)
    if len(payload) > 255:
        raise ValueError("text payload too large for short NDEF record")
    # MB=1, ME=1, SR=1, TNF=0x01 well-known type.
    return bytes([0xD1, 0x01, len(payload), 0x54]) + payload


def tlv_ndef_message(record: bytes) -> bytes:
    if len(record) <= 254:
        return bytes([0x03, len(record)]) + record + b"\xFE"
    return bytes([0x03, 0xFF, (len(record) >> 8) & 0xFF, len(record) & 0xFF]) + record + b"\xFE"


def chunks(data: bytes, size: int) -> Iterable[bytes]:
    for index in range(0, len(data), size):
        yield data[index : index + size]


@dataclass
class ST25DV04:
    bus_id: int = DEFAULT_I2C_BUS
    address: int = DEFAULT_USER_ADDR
    system_address: int = DEFAULT_SYSTEM_ADDR
    page_size: int = DEFAULT_PAGE_SIZE
    write_delay_s: float = 0.006
    i2c_retries: int = DEFAULT_I2C_RETRIES
    i2c_retry_delay_s: float = DEFAULT_I2C_RETRY_DELAY_S

    def __post_init__(self) -> None:
        from smbus2 import SMBus

        self.bus = SMBus(self.bus_id)

    def close(self) -> None:
        self.bus.close()

    def _i2c_retry(self, operation):
        last_error: OSError | None = None
        for attempt in range(max(1, self.i2c_retries)):
            try:
                return operation()
            except OSError as error:
                last_error = error
                time.sleep(self.i2c_retry_delay_s * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("I2C operation failed without an exception")

    def write_bytes(self, offset: int, data: bytes) -> None:
        cursor = offset
        for block in chunks(data, self.page_size):
            # ST25DV uses 16-bit memory offsets on I2C EEPROM access.
            payload = [(cursor >> 8) & 0xFF, cursor & 0xFF, *block]
            self._i2c_retry(
                lambda payload=payload: self.bus.write_i2c_block_data(
                    self.address, payload[0], payload[1:]
                )
            )
            cursor += len(block)
            time.sleep(self.write_delay_s)

    def read_bytes(self, offset: int, length: int) -> bytes:
        return self.read_i2c_bytes(self.address, offset, length)

    def read_i2c_bytes(self, address: int, offset: int, length: int) -> bytes:
        data = bytearray()
        cursor = offset
        remaining = length
        while remaining > 0:
            read_len = min(remaining, MAX_I2C_READ_SIZE)
            self._i2c_retry(
                lambda address=address, cursor=cursor: self.bus.write_i2c_block_data(
                    address, (cursor >> 8) & 0xFF, [cursor & 0xFF]
                )
            )
            chunk = self._i2c_retry(
                lambda address=address, read_len=read_len: self.bus.read_i2c_block_data(
                    address, 0, read_len
                )
            )
            data.extend(chunk)
            cursor += read_len
            remaining -= read_len
        return bytes(data)

    def read_system_bytes(self, offset: int, length: int) -> bytes:
        return self.read_i2c_bytes(self.system_address, offset, length)

    def read_uid_raw(self) -> bytes:
        return self.read_system_bytes(UID_SYSTEM_OFFSET, UID_SIZE)

    def read_uid(self) -> bytes:
        return canonical_uid(self.read_uid_raw())

    def clear(self, length: int = 512) -> None:
        self.write_bytes(0, b"\x00" * length)

    def write_blocks(self, first_block: int, data: bytes) -> None:
        self.write_bytes(first_block * BLOCK_SIZE, data)

    def read_blocks(self, first_block: int, block_count: int) -> bytes:
        return self.read_bytes(first_block * BLOCK_SIZE, block_count * BLOCK_SIZE)

    def write_request(self, packet: bytes) -> None:
        if len(packet) != MAILBOX_SIZE:
            raise ValueError("request packet must be 64 bytes")
        self.write_blocks(REQUEST_START_BLOCK, packet)

    def read_request(self) -> bytes:
        return self.read_blocks(REQUEST_START_BLOCK, MAILBOX_BLOCK_COUNT)

    def write_response(self, packet: bytes) -> None:
        if len(packet) != MAILBOX_SIZE:
            raise ValueError("response packet must be 64 bytes")
        self.write_blocks(RESPONSE_START_BLOCK, packet)

    def read_response(self) -> bytes:
        return self.read_blocks(RESPONSE_START_BLOCK, MAILBOX_BLOCK_COUNT)

    def write_ndef_text(self, text: str, language: str = "en", offset: int = 0) -> bytes:
        message = tlv_ndef_message(ndef_short_text_record(text, language))
        self.write_bytes(offset, message)
        return message


def main() -> None:
    parser = argparse.ArgumentParser(description="Read/write ST25DV04 user memory")
    parser.add_argument("--bus", type=int, default=DEFAULT_I2C_BUS)
    parser.add_argument("--addr", type=lambda value: int(value, 0), default=DEFAULT_USER_ADDR)
    parser.add_argument("--sys-addr", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--i2c-retries", type=int, default=DEFAULT_I2C_RETRIES)
    parser.add_argument("--i2c-retry-delay", type=float, default=DEFAULT_I2C_RETRY_DELAY_S)
    parser.add_argument("--offset", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--uid", action="store_true", help="read immutable ST25DV04 UID")
    parser.add_argument("--read", type=int, default=0, help="read byte count")
    parser.add_argument("--read-blocks", nargs=2, type=lambda value: int(value, 0), metavar=("FIRST", "COUNT"))
    parser.add_argument("--write-hex-blocks", nargs=2, metavar=("FIRST", "HEX"))
    parser.add_argument("--read-request", action="store_true")
    parser.add_argument("--read-response", action="store_true")
    parser.add_argument("--clear", action="store_true", help="clear first 512 bytes")
    parser.add_argument("--text", help="write NDEF text")
    parser.add_argument("--json", dest="json_payload", help="write object as compact JSON NDEF text")
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
            raw = tag.read_uid_raw()
            uid = canonical_uid(raw)
            print(f"uid={uid_hex(uid)}")
            if raw != uid:
                print(f"raw={uid_hex(raw)}")
        if args.clear:
            tag.clear()
            print("cleared")
        if args.text is not None:
            data = tag.write_ndef_text(args.text, offset=args.offset)
            print(f"wrote {len(data)} bytes")
        if args.json_payload is not None:
            parsed = json.loads(args.json_payload)
            compact = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            data = tag.write_ndef_text(compact, offset=args.offset)
            print(f"wrote {len(data)} bytes: {compact}")
        if args.read:
            print(tag.read_bytes(args.offset, args.read).hex(" "))
        if args.read_blocks:
            first, count = args.read_blocks
            print(tag.read_blocks(first, count).hex(" "))
        if args.write_hex_blocks:
            first = int(args.write_hex_blocks[0], 0)
            data = bytes.fromhex(args.write_hex_blocks[1])
            tag.write_blocks(first, data)
            print(f"wrote {len(data)} bytes to block {first}")
        if args.read_request:
            packet = tag.read_request()
            print(json.dumps(unpack_request_packet(packet), ensure_ascii=False))
        if args.read_response:
            packet = tag.read_response()
            print(json.dumps(unpack_response_packet(packet), ensure_ascii=False))
    finally:
        tag.close()


if __name__ == "__main__":
    main()
