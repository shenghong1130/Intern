#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Client for direct Worker and Central Server flower classification."""

import time
from typing import Optional
from urllib.parse import quote, urlsplit, urlunsplit

from .models import ClassificationResult


class ClassifierClient:
    def __init__(
        self,
        url: Optional[str],
        timeout_s: float = 4.0,
        dry_run: bool = False,
        mode: str = "direct",
        student_id: Optional[str] = None,
        password: Optional[str] = None,
        central_poll_deadline_s: float = 180.0,
        central_poll_interval_s: float = 0.5,
    ):
        self.url = url
        self.timeout_s = max(0.001, float(timeout_s))
        self.dry_run = dry_run
        self.mode = str(mode or "direct").lower()
        self.student_id = student_id
        self.password = password
        self.central_poll_deadline_s = max(0.0, float(central_poll_deadline_s))
        self.central_poll_interval_s = max(0.0, float(central_poll_interval_s))

    def classify_crop(self, crop_28x28) -> ClassificationResult:
        if self.dry_run:
            return ClassificationResult(ok=False, error="dry_run", error_kind="disabled")
        if not self.url:
            return ClassificationResult(
                ok=False,
                error="classifier_url_missing",
                error_kind="service_unavailable",
                retryable=True,
            )
        if self.mode not in ("direct", "central"):
            return ClassificationResult(
                ok=False,
                error="unknown_classifier_mode_{}".format(self.mode),
                error_kind="configuration_error",
            )
        if self.mode == "central" and not str(self.student_id or "").strip():
            return ClassificationResult(
                ok=False,
                error="classifier_student_id_missing",
                error_kind="configuration_error",
            )
        if self.mode == "central" and not str(self.password or "").strip():
            return ClassificationResult(
                ok=False,
                error="classifier_password_missing",
                error_kind="configuration_error",
            )
        import cv2
        import requests

        ok, encoded = cv2.imencode(".jpg", crop_28x28)
        if not ok:
            return ClassificationResult(ok=False, error="jpeg_encode_failed", error_kind="invalid_crop")
        image_file = ("crop_28x28.jpg", encoded.tobytes(), "image/jpeg")
        deadline = time.monotonic() + self.central_poll_deadline_s
        try:
            files = {"image": image_file}
            if self.mode == "central":
                response = requests.post(
                    self.url,
                    headers=self._central_auth_headers(),
                    data={"student_id": str(self.student_id).strip()},
                    files=files,
                    timeout=max(0.001, self.central_poll_deadline_s),
                )
            else:
                # Keep the legacy Worker request shape exactly as before.
                response = requests.post(self.url, files=files, timeout=self.timeout_s)
        except requests.exceptions.RequestException as exc:
            return ClassificationResult(
                ok=False,
                error=(
                    "central_server_request_failed {}".format(type(exc).__name__)
                    if self.mode == "central"
                    else str(exc)
                ),
                error_kind="service_unavailable",
                retryable=True,
            )
        if self.mode == "direct":
            if response.status_code != 200:
                return self._http_error(response, central=False)
            return self._parse_json_result(response)

        if response.status_code not in (200, 202):
            return self._http_error(response, central=True)
        envelope = self._json_object(response)
        if isinstance(envelope, ClassificationResult):
            return envelope
        result = self._parse_central_envelope(envelope)
        if result is not None:
            return result
        request_id = envelope.get("request_id")
        if not request_id:
            return ClassificationResult(
                ok=False,
                raw=envelope,
                error="missing_request_id",
                error_kind="invalid_response",
            )
        return self._poll_central_request(str(request_id), deadline, requests)

    def _poll_central_request(self, request_id, deadline, requests) -> ClassificationResult:
        request_url = self._central_request_url(request_id)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                response = requests.get(
                    request_url,
                    headers=self._central_auth_headers(),
                    timeout=max(0.001, min(self.timeout_s, remaining)),
                )
            except requests.exceptions.RequestException as exc:
                return ClassificationResult(
                    ok=False,
                    error="central_server_poll_failed {}".format(type(exc).__name__),
                    error_kind="service_unavailable",
                    retryable=True,
                )
            if response.status_code != 200:
                return self._http_error(response, central=True)
            envelope = self._json_object(response)
            if isinstance(envelope, ClassificationResult):
                return envelope
            result = self._parse_central_envelope(envelope)
            if result is not None:
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(self.central_poll_interval_s, remaining))
        return ClassificationResult(
            ok=False,
            error="central_server_poll_timeout request_id={}".format(request_id),
            error_kind="service_unavailable",
            retryable=True,
        )

    def _central_request_url(self, request_id: str) -> str:
        parts = urlsplit(str(self.url))
        prefix = parts.path.rsplit("/", 1)[0].rstrip("/")
        path = "{}/requests/{}".format(prefix, quote(request_id, safe=""))
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    def _central_auth_headers(self):
        return {"X-Student-Password": str(self.password)}

    def _parse_central_envelope(self, envelope):
        status = str(envelope.get("status") or "").lower()
        if status == "completed":
            result = envelope.get("result")
            if not isinstance(result, dict):
                return ClassificationResult(
                    ok=False,
                    raw=envelope,
                    error="completed_result_missing",
                    error_kind="invalid_response",
                )
            return self._parse_result(result)
        if status == "failed":
            return ClassificationResult(
                ok=False,
                raw=envelope,
                error="central_server_failed {}".format(envelope.get("error") or "unknown error"),
                error_kind="service_unavailable",
                retryable=True,
            )
        if status in ("queued", "running"):
            return None
        return ClassificationResult(
            ok=False,
            raw=envelope,
            error="unexpected_central_status_{}".format(status or "missing"),
            error_kind="invalid_response",
        )

    def _parse_json_result(self, response) -> ClassificationResult:
        raw = self._json_object(response)
        if isinstance(raw, ClassificationResult):
            return raw
        return self._parse_result(raw)

    @staticmethod
    def _json_object(response):
        try:
            raw = response.json()
        except ValueError as exc:
            return ClassificationResult(ok=False, error="bad_json: {}".format(exc), error_kind="invalid_response")
        if not isinstance(raw, dict):
            return ClassificationResult(
                ok=False,
                error="json_response_is_not_object",
                error_kind="invalid_response",
            )
        return raw

    @staticmethod
    def _parse_result(raw) -> ClassificationResult:
        flower_api = raw.get("flower_api") or raw.get("flower") or raw.get("predicted_class")
        flower_cn = raw.get("flower_cn") or raw.get("raw_class")
        if not flower_api:
            return ClassificationResult(ok=False, raw=raw, error="missing_flower_api", error_kind="invalid_response")
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            return ClassificationResult(
                ok=False,
                raw=raw,
                error="invalid_confidence",
                error_kind="invalid_response",
            )
        return ClassificationResult(
            ok=True,
            flower_api=str(flower_api),
            flower_cn=None if flower_cn is None else str(flower_cn),
            confidence=confidence,
            class_index=raw.get("class_index"),
            raw=raw,
        )

    def _http_error(self, response, central: bool) -> ClassificationResult:
        retryable = response.status_code >= 500 or response.status_code in (408, 429)
        prefix = "central_server_http" if central else "http"
        detail = response.text[:120]
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("detail"):
                detail = str(payload["detail"])[:120]
        except ValueError:
            pass
        if central and self.password:
            detail = detail.replace(str(self.password), "<redacted>")
        return ClassificationResult(
            ok=False,
            error="{}_{} {}".format(prefix, response.status_code, detail),
            error_kind="service_unavailable" if retryable else "http_error",
            retryable=retryable,
        )
