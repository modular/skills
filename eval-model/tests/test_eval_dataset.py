# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Unit tests for the dataset accuracy evaluator."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_dataset
from _eval_tasks import GSM8KEvaluator, LoglikelihoodEvaluator, get_task_config


class FakeMaxHandler(BaseHTTPRequestHandler):
    last_payload: dict = {}

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._write({"data": [{"id": "test-model"}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        type(self).last_payload = json.loads(self.rfile.read(length))
        if self.path == "/v1/chat/completions":
            self._write(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": "The answer is #### 42"},
                        }
                    ]
                }
            )
            return
        if self.path == "/v1/completions":
            self._write(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": {
                                "token_logprobs": [None, -0.2, -0.1],
                                "text_offset": [0, 1, 31],
                            },
                        }
                    ]
                }
            )
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _write(self, body: dict) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class EvalDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMaxHandler)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def test_normalize_base_url_accepts_v1(self) -> None:
        self.assertEqual(
            eval_dataset.normalize_base_url("http://localhost:8000/v1/"),
            "http://localhost:8000",
        )

    def test_preflight_checks_chat_and_logprobs(self) -> None:
        eval_dataset.preflight(
            self.base_url,
            "test-model",
            ["gsm8k", "mmlu"],
            None,
        )

    def test_preflight_sends_reasoning_as_chat_template_kwarg(self) -> None:
        eval_dataset.preflight(
            self.base_url,
            "test-model",
            ["gsm8k"],
            "low",
        )
        self.assertEqual(
            FakeMaxHandler.last_payload["chat_template_kwargs"],
            {"reasoning_effort": "low"},
        )

    def test_direct_gsm8k_scoring(self) -> None:
        evaluator = GSM8KEvaluator(
            get_task_config("gsm8k"),
            base_url=self.base_url,
            model="test-model",
            tokenizer=None,
            num_concurrent=1,
        )
        evaluator._dataset = [
            {
                "question": "What is 40 + 2?",
                "answer": "Add the values. #### 42",
            }
        ]
        result = evaluator.run_evaluation(limit=1, is_seed=True)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["total"], 1)

    def test_choice_logprobs_exclude_generated_token(self) -> None:
        evaluator = _TestLoglikelihoodEvaluator(
            get_task_config("hellaswag"),
            base_url=self.base_url,
            model="test-model",
            tokenizer=None,
        )
        values = evaluator._choice_logprobs(
            "A",
            " B",
            {
                "token_logprobs": [None, -0.25, -9.0],
                "text_offset": [0, 1, 3],
            },
        )
        self.assertEqual(values, [-0.25])

    def test_lm_eval_command_uses_tokenizer_and_chat_template(self) -> None:
        command = eval_dataset.build_lm_eval_cmd(
            "mmlu",
            self.base_url,
            "served-alias",
            "org/tokenizer",
            5,
            0.0,
            1.0,
            2,
            Path("/tmp/results"),
            None,
            True,
        )
        model_args = command[command.index("--model_args") + 1]
        self.assertIn("tokenizer=org/tokenizer", model_args)
        self.assertIn("--apply_chat_template", command)
        self.assertIn("--fewshot_as_multiturn", command)

    def test_gpqa_uses_completions_backend(self) -> None:
        command = eval_dataset.build_lm_eval_cmd(
            "gpqa",
            self.base_url,
            "test-model",
            "org/tokenizer",
            5,
            0.0,
            1.0,
            2,
            Path("/tmp/results"),
            None,
            False,
        )
        model_backend = command[command.index("--model") + 1]
        self.assertEqual(model_backend, "local-completions")

    def test_aime_seed_matches_aime24(self) -> None:
        config = get_task_config("aime")
        self.assertEqual(config.hf_split, "train")
        self.assertEqual(config.filter_key, "url")
        self.assertEqual(config.filter_contains, "2024_")
        self.assertEqual(config.full_limit, 30)

    def test_gsm8k_prefers_flexible_extract_metric(self) -> None:
        score, metric = eval_dataset.extract_metric(
            {
                "exact_match,strict-match": 0.2,
                "exact_match,flexible-extract": 0.6,
            },
            "exact_match",
            "gsm8k",
        )
        self.assertEqual(score, 0.6)
        self.assertEqual(metric, "exact_match,flexible-extract")

    def test_direct_heartbeat_runs_while_request_is_pending(self) -> None:
        evaluator = _SlowEvaluator(
            get_task_config("gsm8k"),
            base_url=self.base_url,
            model="test-model",
            tokenizer=None,
            num_concurrent=1,
        )
        evaluator._dataset = [{"question": "q", "answer": "1"}]
        heartbeats = []
        evaluator.run_evaluation(
            limit=1,
            is_seed=True,
            heartbeat_cb=heartbeats.append,
            heartbeat_interval=0.02,
        )
        self.assertTrue(any(item["completed"] == 0 for item in heartbeats))
        self.assertTrue(heartbeats[-1]["force"])

    def test_summary_without_target_is_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exit_code = eval_dataset.write_summaries(
                "test",
                "test-model",
                self.base_url,
                ["gsm8k"],
                {
                    "gsm8k": {
                        "accuracy": 0.5,
                        "total": 2,
                        "fast_fail": False,
                    }
                },
                {
                    "gsm8k": {
                        "accuracy": 0.5,
                        "samples": 2,
                        "metric": "exact_match",
                    }
                },
                Path(directory),
                {},
            )
            summary = json.loads(
                (Path(directory) / "gsm8k_summary.json").read_text()
            )
        self.assertEqual(exit_code, eval_dataset.EXIT_OK)
        self.assertEqual(summary["status"], "completed")
        self.assertIsNone(summary["target"])


class _TestLoglikelihoodEvaluator(LoglikelihoodEvaluator):
    def _get_choices(self, item: dict) -> list[str]:
        return []


class _SlowEvaluator(GSM8KEvaluator):
    def evaluate_item(self, item: dict, is_seed: bool) -> tuple[bool, dict]:
        time.sleep(0.08)
        return True, {"correct": True}


if __name__ == "__main__":
    unittest.main()
