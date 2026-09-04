# robotall

`robotall` is a small Python library for controlling TonyPi robot servos and writing robot contest request/response packets to ST25DV04 NFC tags.

## Features

- Servo motion helpers for head turning and action group playback
- ST25DV04 NFC mailbox read/write support
- Robot contest identity and attempt request packet helpers

## Installation

```bash
pip install .
```

## Requirements

- Python 3.9+
- `smbus2`

## Usage

```python
from robotall import act, register_robot, send_request
```
