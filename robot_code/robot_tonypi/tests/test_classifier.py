from pathlib import Path
from types import SimpleNamespace
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.classifier import ClassifierClient
from robot_tonypi.main import parse_args


class FakeResponse:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self.payload = payload
        self.text = text or str(payload)

    def json(self):
        return self.payload


class EncodedJpeg:
    def tobytes(self):
        return b"jpeg-bytes"


class RequestException(Exception):
    pass


class ClassifierClientTests(unittest.TestCase):
    def setUp(self):
        self.cv2 = SimpleNamespace(imencode=mock.Mock(return_value=(True, EncodedJpeg())))
        self.requests = SimpleNamespace(
            post=mock.Mock(),
            get=mock.Mock(),
            exceptions=SimpleNamespace(RequestException=RequestException),
        )
        self.modules = mock.patch.dict(
            sys.modules,
            {"cv2": self.cv2, "requests": self.requests},
        )
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    @staticmethod
    def worker_result():
        return {
            "ok": True,
            "flower_api": "taohua",
            "flower_cn": "桃花",
            "class_index": 8,
            "confidence": 0.9,
        }

    def test_direct_worker_200_preserves_old_result(self):
        self.requests.post.return_value = FakeResponse(200, self.worker_result())

        result = ClassifierClient("http://worker:8080/predict").classify_crop(object())

        self.assertTrue(result.ok)
        self.assertEqual(result.flower_api, "taohua")
        self.assertEqual(result.flower_cn, "桃花")
        self.assertEqual(result.class_index, 8)
        self.assertEqual(result.confidence, 0.9)
        self.assertNotIn("data", self.requests.post.call_args.kwargs)
        self.assertNotIn("headers", self.requests.post.call_args.kwargs)
        self.assertIn("image", self.requests.post.call_args.kwargs["files"])
        self.requests.get.assert_not_called()

    def test_central_200_completed_returns_inner_result(self):
        self.requests.post.return_value = FakeResponse(
            200,
            {
                "request_id": "req_1",
                "student_id": "student01",
                "artifact_id": "art_1",
                "version": "v1",
                "status": "completed",
                "result": self.worker_result(),
                "error": None,
            },
        )

        result = ClassifierClient(
            "http://central:8000/predict",
            mode="central",
            student_id="student01",
            password="test-password",
        ).classify_crop(object())

        self.assertTrue(result.ok)
        self.assertEqual(result.flower_api, "taohua")
        self.assertEqual(result.raw, self.worker_result())
        self.assertEqual(
            self.requests.post.call_args.kwargs["data"],
            {"student_id": "student01"},
        )
        self.assertEqual(
            self.requests.post.call_args.kwargs["headers"],
            {"X-Student-Password": "test-password"},
        )
        self.assertIn("image", self.requests.post.call_args.kwargs["files"])
        self.requests.get.assert_not_called()

    def test_central_202_queued_then_completed(self):
        self.requests.post.return_value = FakeResponse(
            202,
            {"request_id": "req_1", "status": "queued"},
        )
        self.requests.get.side_effect = [
            FakeResponse(200, {"request_id": "req_1", "status": "queued"}),
            FakeResponse(
                200,
                {
                    "request_id": "req_1",
                    "status": "completed",
                    "result": self.worker_result(),
                },
            ),
        ]

        with mock.patch("robot_tonypi.classifier.time.sleep"):
            result = ClassifierClient(
                "http://central:8000/predict",
                mode="central",
                student_id="student01",
                password="test-password",
                central_poll_interval_s=0.01,
            ).classify_crop(object())

        self.assertTrue(result.ok)
        self.assertEqual(result.flower_api, "taohua")
        self.assertEqual(self.requests.get.call_count, 2)
        self.assertEqual(
            self.requests.get.call_args_list[0].args[0],
            "http://central:8000/requests/req_1",
        )
        self.assertEqual(
            self.requests.post.call_args.kwargs["headers"],
            {"X-Student-Password": "test-password"},
        )
        for call in self.requests.get.call_args_list:
            self.assertEqual(
                call.kwargs["headers"],
                {"X-Student-Password": "test-password"},
            )

    def test_central_queued_stops_at_deadline(self):
        self.requests.post.return_value = FakeResponse(
            202,
            {"request_id": "req_1", "status": "queued"},
        )
        self.requests.get.return_value = FakeResponse(
            200,
            {"request_id": "req_1", "status": "queued"},
        )
        clock = iter(index * 0.01 for index in range(100))

        with mock.patch(
            "robot_tonypi.classifier.time.monotonic",
            side_effect=lambda: next(clock),
        ), mock.patch("robot_tonypi.classifier.time.sleep"):
            result = ClassifierClient(
                "http://central:8000/predict",
                mode="central",
                student_id="student01",
                password="test-password",
                central_poll_deadline_s=0.05,
                central_poll_interval_s=0.01,
            ).classify_crop(object())

        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "service_unavailable")
        self.assertTrue(result.retryable)
        self.assertIn("central_server_poll_timeout", result.error)
        self.assertGreater(self.requests.get.call_count, 0)
        self.assertLess(self.requests.get.call_count, 10)

    def test_central_404_reports_missing_artifact_clearly(self):
        self.requests.post.return_value = FakeResponse(
            404,
            {"detail": "student has no ready artifact"},
        )

        result = ClassifierClient(
            "http://central:8000/predict",
            mode="central",
            student_id="student01",
            password="test-password",
        ).classify_crop(object())

        self.assertFalse(result.ok)
        self.assertFalse(result.retryable)
        self.assertEqual(result.error_kind, "http_error")
        self.assertIn("central_server_http_404", result.error)
        self.assertIn("student has no ready artifact", result.error)

    def test_central_requires_password(self):
        result = ClassifierClient(
            "http://central:8000/predict",
            mode="central",
            student_id="student01",
        ).classify_crop(object())

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "classifier_password_missing")
        self.assertEqual(result.error_kind, "configuration_error")
        self.requests.post.assert_not_called()

    def test_central_401_is_not_retryable(self):
        self.requests.post.return_value = FakeResponse(
            401,
            {"detail": "invalid student credentials: test-password"},
        )

        result = ClassifierClient(
            "http://central:8000/predict",
            mode="central",
            student_id="student01",
            password="test-password",
        ).classify_crop(object())

        self.assertFalse(result.ok)
        self.assertFalse(result.retryable)
        self.assertEqual(result.error_kind, "http_error")
        self.assertIn("central_server_http_401", result.error)
        self.assertNotIn("test-password", result.error)
        self.assertIn("<redacted>", result.error)

    def test_cli_direct_needs_no_credentials_and_central_requires_both(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            direct = parse_args(["--target-flower", "hehua"])
            self.assertEqual(direct.classifier_mode, "direct")
            self.assertIsNone(direct.classifier_student_id)
            self.assertIsNone(direct.classifier_password)
            with self.assertRaises(SystemExit):
                parse_args(["--target-flower", "hehua", "--classifier-mode", "central"])
            with self.assertRaises(SystemExit):
                parse_args([
                    "--target-flower", "hehua",
                    "--classifier-mode", "central",
                    "--classifier-student-id", "student01",
                ])
            explicit = parse_args([
                "--target-flower", "hehua",
                "--classifier-mode", "central",
                "--classifier-student-id", "student01",
                "--classifier-password", "explicit-password",
            ])
            self.assertEqual(explicit.classifier_password, "explicit-password")

    def test_cli_uses_student_password_environment_variable(self):
        with mock.patch.dict(os.environ, {"STUDENT_PASSWORD": "env-password"}):
            central = parse_args([
                "--target-flower", "hehua",
                "--classifier-mode", "central",
                "--classifier-student-id", "student01",
            ])
        self.assertEqual(central.classifier_student_id, "student01")
        self.assertEqual(central.classifier_password, "env-password")


if __name__ == "__main__":
    unittest.main()
