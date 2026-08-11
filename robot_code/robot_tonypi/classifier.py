#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Client for the FPGA flower classifier service."""

from typing import Optional

from .models import ClassificationResult


class ClassifierClient:
    def __init__(self, url: Optional[str], timeout_s: float = 4.0, dry_run: bool = False):
        self.url = url
        self.timeout_s = timeout_s
        self.dry_run = dry_run

    def classify_crop(self, crop_28x28) -> ClassificationResult:
        if self.dry_run:
            return ClassificationResult(ok=False, error="dry_run")
        if not self.url:
            return ClassificationResult(ok=False, error="classifier_url_missing")
        import cv2
        import requests

        ok, encoded = cv2.imencode(".jpg", crop_28x28)
        if not ok:
            return ClassificationResult(ok=False, error="jpeg_encode_failed")
        try:
            files = {"image": ("crop_28x28.jpg", encoded.tobytes(), "image/jpeg")}
            response = requests.post(self.url, files=files, timeout=self.timeout_s)
        except Exception as exc:
            return ClassificationResult(ok=False, error=str(exc))
        if response.status_code != 200:
            return ClassificationResult(ok=False, error="http_{} {}".format(response.status_code, response.text[:120]))
        try:
            raw = response.json()
        except ValueError as exc:
            return ClassificationResult(ok=False, error="bad_json: {}".format(exc))
        flower_api = raw.get("flower_api") or raw.get("flower") or raw.get("predicted_class")
        flower_cn = raw.get("flower_cn") or raw.get("raw_class")
        if not flower_api:
            return ClassificationResult(ok=False, raw=raw, error="missing_flower_api")
        return ClassificationResult(
            ok=True,
            flower_api=str(flower_api),
            flower_cn=None if flower_cn is None else str(flower_cn),
            confidence=float(raw.get("confidence", 0.0)),
            class_index=raw.get("class_index"),
            raw=raw,
        )
