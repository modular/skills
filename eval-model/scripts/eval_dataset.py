#!/usr/bin/env python3
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
"""Evaluate datasets against a MAX OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _eval_tasks import (
    TASK_CONFIGS,
    get_evaluator,
    get_task_config,
)

TASKS: dict[str, dict[str, Any]] = {
    name: {
        "lm_eval_task": config.lm_eval_task,
        "type": config.type,
        "seed_limit": config.seed_limit,
        "full_limit": config.full_limit,
        "floor": config.fast_fail_floor,
        "is_canary": config.is_canary,
        "metric": config.metric,
        "lm_eval_only": config.lm_eval_only,
        "lm_eval_metadata": config.lm_eval_metadata,
        "grouped": config.grouped,
    }
    for name, config in TASK_CONFIGS.items()
}
ALL_TASKS = list(TASKS)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BELOW_TARGET = 2
EXIT_FAST_FAIL = 3


class EvaluationError(RuntimeError):
    """An infrastructure or configuration error that invalidates the run."""

    def __init__(self, message: str, task: str = ""):
        super().__init__(message)
        self.task = task


class HeartbeatEmitter:
    """Emit newline-delimited JSON progress records to stdout."""

    def __init__(self, interval: int = 30, task: str = ""):
        self.interval = interval
        self.task = task
        self.last_emit = 0.0

    def emit(
        self,
        phase: str,
        completed: int,
        total: int | None,
        accuracy: float | None = None,
        eta_sec: int | None = None,
        force: bool = False,
        **extra: Any,
    ) -> None:
        now = time.time()
        if force or now - self.last_emit >= self.interval:
            heartbeat = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "phase": phase,
                "task": self.task,
                "completed": completed,
                "total": total,
                "accuracy": accuracy,
                "eta_sec": eta_sec,
            }
            heartbeat.update(extra)
            print(json.dumps(heartbeat), flush=True)
            self.last_emit = now


def log(message: str = "") -> None:
    """Write human-readable output without contaminating heartbeat JSONL."""
    print(message, file=sys.stderr, flush=True)


def resolve_run_id(model: str, run_id: str | None = None) -> str:
    """Return a safe label for the default output directory."""
    value = run_id or model.replace("/", "_")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise EvaluationError(
            "--run-id must contain only letters, numbers, '.', '_', or '-'"
        )
    return value


def normalize_base_url(value: str) -> str:
    """Normalize either a server root or an OpenAI /v1 base URL."""
    normalized = value.rstrip("/")
    normalized = normalized.removesuffix("/v1")
    return normalized


def parse_mapping(
    value: str | None,
    option: str,
    valid_tasks: list[str],
) -> dict[str, float]:
    """Parse comma-separated task:value mappings."""
    parsed: dict[str, float] = {}
    if not value:
        return parsed
    for pair in value.split(","):
        try:
            task, raw_number = pair.split(":", 1)
            task = task.strip()
            number = float(raw_number)
        except ValueError as exc:
            raise EvaluationError(
                f"Invalid {option} value {pair!r}; expected task:number"
            ) from exc
        if task not in valid_tasks:
            raise EvaluationError(
                f"Unknown task {task!r} in {option}. Available: {', '.join(valid_tasks)}"
            )
        if not 0.0 <= number <= 1.0:
            raise EvaluationError(f"{option} values must be between 0 and 1")
        parsed[task] = number
    return parsed


def request_json(
    url: str,
    payload: dict | None = None,
    timeout: int = 30,
) -> dict:
    """Send an HTTP request and return its JSON object."""
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise EvaluationError(
            f"{url} returned HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise EvaluationError(f"Could not reach {url}: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Invalid response from {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise EvaluationError(f"Expected a JSON object from {url}")
    return result


def preflight(
    base_url: str, model: str, tasks: list[str], reasoning_effort: str | None
) -> None:
    """Verify the endpoint and task-specific API capabilities."""
    models = request_json(f"{base_url}/v1/models")
    model_ids = {
        item.get("id")
        for item in models.get("data", [])
        if isinstance(item, dict)
    }
    if model_ids and model not in model_ids:
        available = ", ".join(sorted(str(item) for item in model_ids))
        raise EvaluationError(
            f"Model {model!r} is not listed by /v1/models. Available: {available}"
        )

    needs_chat = any(TASKS[task]["type"] != "loglikelihood" for task in tasks)
    if needs_chat:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1,
            "temperature": 0,
        }
        if reasoning_effort:
            payload["chat_template_kwargs"] = {
                "reasoning_effort": reasoning_effort
            }
        response = request_json(
            f"{base_url}/v1/chat/completions",
            payload,
            timeout=120,
        )
        if not response.get("choices"):
            raise EvaluationError(
                "Chat completions preflight returned no choices"
            )

    needs_logprobs = any(
        TASKS[task]["type"] == "loglikelihood" for task in tasks
    )
    if needs_logprobs:
        response = request_json(
            f"{base_url}/v1/completions",
            {
                "model": model,
                "prompt": "The capital of France is Paris.",
                "max_tokens": 1,
                "temperature": 0,
                "echo": True,
                "logprobs": 1,
            },
            timeout=120,
        )
        choices = response.get("choices") or []
        logprobs = choices[0].get("logprobs") if choices else None
        if not logprobs or not logprobs.get("token_logprobs"):
            raise EvaluationError(
                "Multiple-choice tasks require prompt logprobs. Restart MAX "
                "Serve with --enable-echo and a runtime that supports logprobs."
            )


def preflight_with_heartbeat(
    base_url: str,
    model: str,
    tasks: list[str],
    reasoning_effort: str | None,
    heartbeat_interval: int,
) -> None:
    """Run endpoint checks while emitting liveness records."""
    emitter = HeartbeatEmitter(heartbeat_interval)
    stop_heartbeat = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(heartbeat_interval):
            emitter.emit("preflight", 0, 0)

    emitter.emit("preflight", 0, 0, force=True)
    thread = threading.Thread(
        target=heartbeat_loop,
        name="preflight-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        preflight(base_url, model, tasks, reasoning_effort)
    finally:
        stop_heartbeat.set()
        thread.join()


def validate_dependencies(tasks: list[str]) -> None:
    """Report missing Python dependencies before downloading data."""
    missing = []
    if importlib.util.find_spec("datasets") is None:
        missing.append("datasets")
    if importlib.util.find_spec("lm_eval") is None:
        missing.append("lm-eval[api]")
    if missing:
        manifest = _SCRIPT_DIR.parent / "pixi.toml"
        raise EvaluationError(
            "Missing evaluator dependencies: "
            + ", ".join(missing)
            + f". Run this evaluator with pixi using {manifest}."
        )


def dependency_versions() -> dict[str, str]:
    """Return installed evaluator dependency versions."""
    versions = {}
    for package in ("datasets", "lm-eval", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def validate_thresholds() -> None:
    """Print the effective task configuration."""
    print("Task\tType\tSeed\tFull\tFloor\tFast-fail\tFull evaluator")
    for name, config in TASKS.items():
        evaluator = "lm-eval" if config["lm_eval_only"] else "direct/lm-eval"
        print(
            f"{name}\t{config['type']}\t{config['seed_limit']}\t"
            f"{config['full_limit']}\t{config['floor']:.0%}\t"
            f"{config['is_canary']}\t{evaluator}"
        )


def run_seed_pass(
    base_url: str,
    model: str,
    tasks: list[str],
    seed_limits: dict[str, int],
    floors: dict[str, float],
    temperature: float,
    top_p: float,
    reasoning_effort: str | None,
    num_concurrent: int,
    heartbeat_interval: int,
) -> dict[str, dict[str, Any]]:
    """Run direct-HTTP seed evaluations."""
    results: dict[str, dict[str, Any]] = {}
    for task_name in tasks:
        config = TASKS[task_name]
        limit = seed_limits.get(task_name, config["seed_limit"])
        floor = floors.get(task_name, config["floor"])
        emitter = HeartbeatEmitter(heartbeat_interval, task_name)
        evaluator = get_evaluator(
            get_task_config(task_name),
            base_url=base_url,
            model=model,
            tokenizer=None,
            num_concurrent=num_concurrent,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            apply_chat_template=False,
        )

        log(f"\n=== {task_name} seed ({limit} examples) ===")
        emitter.emit("seed", 0, limit, force=True)

        def heartbeat(
            data: dict,
            floor: float = floor,
            emitter: HeartbeatEmitter = emitter,
        ) -> None:
            accuracy = data.get("accuracy")
            data["fast_fail"] = accuracy is not None and accuracy < floor
            data["floor"] = floor
            emitter.emit(**data)

        try:
            result = evaluator.run_evaluation(
                limit=limit,
                is_seed=True,
                heartbeat_cb=heartbeat,
                heartbeat_interval=heartbeat_interval,
            )
        except Exception as exc:
            raise EvaluationError(str(exc), task_name) from exc
        accuracy = result["accuracy"]
        fast_fail = accuracy < floor
        log(
            f"Seed accuracy: {accuracy:.1%} "
            f"({result['correct']}/{result['total']})"
            f"{' <-- FAST-FAIL' if fast_fail else ''}"
        )
        results[task_name] = {
            "accuracy": accuracy,
            "correct": result["correct"],
            "total": result["total"],
            "fast_fail": fast_fail,
            "floor": floor,
        }
    return results


def build_lm_eval_cmd(
    task: str,
    base_url: str,
    model: str,
    tokenizer: str,
    limit: int,
    temperature: float,
    top_p: float,
    num_concurrent: int,
    output_path: Path,
    metadata: str | None,
    apply_chat_template: bool,
) -> list[str]:
    """Build one lm-eval command for one task."""
    config = TASKS[task]
    is_loglikelihood = config["type"] == "loglikelihood"
    if is_loglikelihood:
        lm_model = "local-completions"
        endpoint = "/v1/completions"
        model_args = [
            f"base_url={base_url}{endpoint}",
            f"model={model}",
            f"tokenizer={tokenizer}",
            "tokenizer_backend=huggingface",
            "tokenized_requests=False",
            f"num_concurrent={num_concurrent}",
            "max_retries=3",
            "timeout=600",
        ]
    else:
        lm_model = "local-chat-completions"
        endpoint = "/v1/chat/completions"
        model_args = [
            f"base_url={base_url}{endpoint}",
            f"model={model}",
            f"num_concurrent={num_concurrent}",
            "max_retries=3",
            "timeout=600",
        ]

    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        lm_model,
        "--model_args",
        ",".join(model_args),
        "--tasks",
        config["lm_eval_task"],
        "--limit",
        str(limit),
        "--output_path",
        str(output_path),
    ]
    if not is_loglikelihood or apply_chat_template:
        command.append("--apply_chat_template")
        command.append("--fewshot_as_multiturn")
    if not is_loglikelihood:
        generation_args = []
        if temperature != 0:
            generation_args.append(f"temperature={temperature}")
        if top_p != 1:
            generation_args.append(f"top_p={top_p}")
        if generation_args:
            command.extend(["--gen_kwargs", ",".join(generation_args)])
    if metadata:
        command.extend(["--metadata", metadata])
    return command


def extract_metric(
    result: dict,
    preferred_metric: str,
    task: str,
) -> tuple[float, str]:
    """Find a numeric metric in an lm-eval result object."""
    if task == "gsm8k":
        for key, value in result.items():
            if (
                key.startswith("exact_match")
                and "flexible-extract" in key
                and "stderr" not in key
                and isinstance(value, (int, float))
            ):
                return float(value), key
    for wanted in (
        preferred_metric,
        "exact_match",
        "acc_norm",
        "acc",
        "relaxed_accuracy",
    ):
        for key, value in result.items():
            if (
                key.startswith(wanted)
                and "stderr" not in key
                and isinstance(value, (int, float))
            ):
                return float(value), key
    raise EvaluationError(
        f"lm-eval result did not contain metric {preferred_metric!r}: "
        f"{', '.join(result)}"
    )


def parse_lm_eval_result(
    output_path: Path,
    task: str,
    requested_limit: int,
) -> dict[str, Any]:
    """Parse one task result from lm-eval output."""
    result_files = sorted(output_path.rglob("results_*.json"))
    if not result_files:
        raise EvaluationError(f"lm-eval wrote no results under {output_path}")
    data = json.loads(result_files[-1].read_text())
    lm_task = TASKS[task]["lm_eval_task"]
    result = data.get("results", {}).get(lm_task)
    if result is None:
        result = data.get("groups", {}).get(lm_task)
    if result is None:
        available = sorted(
            set(data.get("results", {})) | set(data.get("groups", {}))
        )
        raise EvaluationError(
            f"lm-eval result is missing {lm_task!r}. Available: {available}"
        )
    accuracy, metric = extract_metric(result, TASKS[task]["metric"], task)
    sample_info = data.get("n-samples", {}).get(lm_task, {})
    samples = sample_info.get("effective") or sample_info.get("original")
    if not isinstance(samples, int):
        child_samples = [
            info.get("effective") or info.get("original")
            for name, info in data.get("n-samples", {}).items()
            if name.startswith(f"{lm_task}_") and isinstance(info, dict)
        ]
        numeric_child_samples = [
            value for value in child_samples if isinstance(value, int)
        ]
        samples = (
            sum(numeric_child_samples)
            if numeric_child_samples
            else requested_limit
        )
    return {"accuracy": accuracy, "metric": metric, "samples": samples}


def run_lm_eval_full(
    base_url: str,
    model: str,
    tokenizer: str,
    tasks: list[str],
    full_limits: dict[str, int],
    temperature: float,
    top_p: float,
    num_concurrent: int,
    heartbeat_interval: int,
    metadata_override: str | None,
    apply_chat_template: bool,
) -> dict[str, dict[str, Any]]:
    """Run each full task through lm-eval with periodic heartbeats."""
    results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        config = TASKS[task]
        limit = full_limits.get(task, config["full_limit"])
        metadata = metadata_override or config["lm_eval_metadata"]
        output_path = Path(tempfile.mkdtemp(prefix=f"eval_{task}_"))
        emitter = HeartbeatEmitter(heartbeat_interval, task)
        command = build_lm_eval_cmd(
            task,
            base_url,
            model,
            tokenizer,
            limit,
            temperature,
            top_p,
            num_concurrent,
            output_path,
            metadata,
            apply_chat_template,
        )
        limit_label = (
            f"{limit} examples per subtask"
            if config["grouped"]
            else f"{limit} examples"
        )
        log(f"\n=== {task} full ({limit_label}, lm-eval) ===")
        heartbeat_total = None if config["grouped"] else limit
        heartbeat_extra = (
            {"limit_per_subtask": limit} if config["grouped"] else {}
        )
        emitter.emit(
            "full",
            0,
            heartbeat_total,
            force=True,
            **heartbeat_extra,
        )

        stdout_file = tempfile.TemporaryFile(mode="w+")
        stderr_file = tempfile.TemporaryFile(mode="w+")
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
            while process.poll() is None:
                emitter.emit(
                    "full",
                    0,
                    heartbeat_total,
                    **heartbeat_extra,
                )
                time.sleep(min(1, heartbeat_interval))
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
            if process.returncode != 0:
                diagnostic = (stderr or stdout)[-4000:]
                raise EvaluationError(
                    f"lm-eval failed for {task} (exit {process.returncode}):\n"
                    f"{diagnostic}",
                    task,
                )
            try:
                result = parse_lm_eval_result(output_path, task, limit)
            except EvaluationError as exc:
                raise EvaluationError(str(exc), task) from exc
        finally:
            stdout_file.close()
            stderr_file.close()
            shutil.rmtree(output_path, ignore_errors=True)

        results[task] = result
        log(
            f"Full accuracy: {result['accuracy']:.1%} "
            f"({result['metric']}, {result['samples']} samples)"
        )
        emitter.emit(
            "full",
            result["samples"],
            result["samples"],
            accuracy=result["accuracy"],
            eta_sec=0,
            force=True,
        )
    return results


def run_direct_http_full(
    base_url: str,
    model: str,
    tasks: list[str],
    full_limits: dict[str, int],
    temperature: float,
    top_p: float,
    reasoning_effort: str | None,
    num_concurrent: int,
    heartbeat_interval: int,
) -> dict[str, dict[str, Any]]:
    """Run supported generation tasks directly over HTTP."""
    results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        config = TASKS[task]
        limit = full_limits.get(task, config["full_limit"])
        emitter = HeartbeatEmitter(heartbeat_interval, task)
        evaluator = get_evaluator(
            get_task_config(task),
            base_url=base_url,
            model=model,
            tokenizer=None,
            num_concurrent=num_concurrent,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            apply_chat_template=False,
        )
        log(f"\n=== {task} full ({limit} examples, direct HTTP) ===")
        emitter.emit("full", 0, limit, force=True)

        def heartbeat(data: dict, emitter: HeartbeatEmitter = emitter) -> None:
            emitter.emit(**data)

        try:
            result = evaluator.run_evaluation(
                limit=limit,
                is_seed=False,
                heartbeat_cb=heartbeat,
                heartbeat_interval=heartbeat_interval,
            )
        except Exception as exc:
            raise EvaluationError(str(exc), task) from exc
        results[task] = {
            "accuracy": result["accuracy"],
            "metric": config["metric"],
            "samples": result["total"],
        }
        log(
            f"Full accuracy: {result['accuracy']:.1%} "
            f"({result['correct']}/{result['total']})"
        )
    return results


def write_summaries(
    run_id: str,
    model: str,
    base_url: str,
    tasks: list[str],
    seed_results: dict[str, dict[str, Any]],
    full_results: dict[str, dict[str, Any]],
    output_dir: Path,
    targets: dict[str, float],
    run_config: dict[str, Any] | None = None,
) -> int:
    """Write task and combined summaries; return the aggregate exit code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    exit_code = EXIT_OK

    for task in tasks:
        seed = seed_results.get(task, {})
        full = full_results.get(task, {})
        target = targets.get(task)
        if seed.get("fast_fail"):
            accuracy = seed["accuracy"]
            samples = seed["total"]
            status = "fast-fail"
            exit_code = EXIT_FAST_FAIL
        elif not full:
            accuracy = seed.get("accuracy")
            samples = seed.get("total", 0)
            status = "full-eval-skipped"
        else:
            accuracy = full["accuracy"]
            samples = full["samples"]
            if target is None:
                status = "completed"
            elif accuracy >= target:
                status = "met"
            else:
                status = "below-target"
                if exit_code == EXIT_OK:
                    exit_code = EXIT_BELOW_TARGET

        summary = {
            "task": task,
            "accuracy": accuracy,
            "samples": samples,
            "metric": full.get("metric"),
            "target": target,
            "status": status,
            "seed_accuracy": seed.get("accuracy"),
            "seed_samples": seed.get("total", 0),
            "fast_fail": seed.get("fast_fail", False),
        }
        summary_path = output_dir / f"{task}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        summaries[task] = summary
        HeartbeatEmitter(0, task).emit(
            "done",
            samples,
            samples,
            accuracy=accuracy,
            eta_sec=0,
            force=True,
            status=status,
            target=target,
            summary_path=str(summary_path),
        )
        log(f"Summary written: {summary_path}")

    combined_path = output_dir / "eval_summary.json"
    combined_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": model,
                "base_url": base_url,
                "config": run_config or {},
                "versions": dependency_versions(),
                "tasks": summaries,
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
            },
            indent=2,
        )
        + "\n"
    )
    log(f"\nCombined summary: {combined_path}")
    return exit_code


def positive_int(value: str) -> int:
    """Argparse type for positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate datasets against a MAX OpenAI-compatible endpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available tasks: {', '.join(ALL_TASKS)}",
    )
    parser.add_argument("--model", help="Model ID returned by /v1/models")
    parser.add_argument(
        "--tokenizer",
        help="Hugging Face tokenizer ID for MC tasks (default: --model)",
    )
    parser.add_argument("--host", default="localhost", help="Serve host")
    parser.add_argument("--port", type=int, default=8000, help="Serve port")
    parser.add_argument(
        "--base-url",
        help="Server root or OpenAI /v1 base URL (overrides host/port)",
    )
    parser.add_argument(
        "--tasks", default="gsm8k", help="Comma-separated tasks"
    )
    parser.add_argument("--seed-limit", type=positive_int)
    parser.add_argument("--full-limit", type=positive_int)
    parser.add_argument(
        "--no-fast-fail",
        action="store_true",
        help="Skip the direct seed pass",
    )
    parser.add_argument(
        "--fast-fail-floor",
        help="Overrides such as gsm8k:0.15,aime:0.05",
    )
    parser.add_argument(
        "--target",
        help="Optional pass targets such as gsm8k:0.50,mmlu:0.60",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=positive_int,
        default=30,
    )
    parser.add_argument("--num-concurrent", type=positive_int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        help="MAX chat-template reasoning effort for direct generation",
    )
    parser.add_argument(
        "--direct-http",
        action="store_true",
        help="Use direct HTTP for supported generation full passes",
    )
    parser.add_argument(
        "--lm-eval-metadata",
        help="JSON forwarded to lm-eval --metadata",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply the tokenizer chat template to lm-eval completion tasks",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", help="Default output directory label")
    parser.add_argument(
        "--validate-thresholds",
        action="store_true",
        help="Print task configuration without running an evaluation",
    )
    return parser


def effective_run_config(
    args: argparse.Namespace,
    tasks: list[str],
    tokenizer: str,
    floors: dict[str, float],
    targets: dict[str, float],
) -> dict[str, Any]:
    """Build the fully resolved configuration stored with results."""
    use_direct = bool(args.direct_http or args.reasoning_effort)
    per_task = {}
    for task in tasks:
        config = TASKS[task]
        direct_supported = not config["lm_eval_only"]
        per_task[task] = {
            "seed_limit": (
                args.seed_limit
                if args.seed_limit is not None and direct_supported
                else config["seed_limit"]
            ),
            "full_limit_per_subtask": (
                args.full_limit
                if args.full_limit is not None
                else config["full_limit"]
            ),
            "fast_fail_floor": floors.get(task, config["floor"]),
            "target": targets.get(task),
            "full_evaluator": (
                "direct-http" if use_direct and direct_supported else "lm-eval"
            ),
            "lm_eval_metadata": (
                args.lm_eval_metadata or config["lm_eval_metadata"]
            ),
        }
    return {
        "tasks": per_task,
        "tokenizer": tokenizer,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_concurrent": args.num_concurrent,
        "heartbeat_interval": args.heartbeat_interval,
        "reasoning_effort": args.reasoning_effort,
        "apply_chat_template": args.apply_chat_template,
        "fast_fail_enabled": not args.no_fast_fail,
    }


def run(args: argparse.Namespace) -> int:
    """Validate arguments and execute the evaluation."""
    if args.validate_thresholds:
        validate_thresholds()
        return EXIT_OK
    if not args.model:
        raise EvaluationError("--model is required")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise EvaluationError("--tasks must contain at least one task")
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise EvaluationError(
            f"Unknown tasks: {', '.join(unknown)}. Available: {', '.join(ALL_TASKS)}"
        )

    floors = parse_mapping(args.fast_fail_floor, "--fast-fail-floor", tasks)
    targets = parse_mapping(args.target, "--target", tasks)
    if not 0 <= args.temperature:
        raise EvaluationError("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise EvaluationError("--top-p must be greater than 0 and at most 1")

    run_id = resolve_run_id(args.model, args.run_id)
    base_url = (
        normalize_base_url(args.base_url)
        if args.base_url
        else f"http://{args.host}:{args.port}"
    )
    output_dir = args.output_dir or Path("eval-results") / run_id
    tokenizer = args.tokenizer or args.model

    validate_dependencies(tasks)
    log(f"=== Preflight: {base_url} ===")
    preflight_with_heartbeat(
        base_url,
        args.model,
        tasks,
        args.reasoning_effort,
        args.heartbeat_interval,
    )

    lm_only_tasks = [task for task in tasks if TASKS[task]["lm_eval_only"]]
    seedable_tasks = [task for task in tasks if not TASKS[task]["lm_eval_only"]]
    seed_limits = {
        task: args.seed_limit
        for task in seedable_tasks
        if args.seed_limit is not None
    }
    full_limits = {
        task: args.full_limit for task in tasks if args.full_limit is not None
    }
    run_config = effective_run_config(
        args,
        tasks,
        tokenizer,
        floors,
        targets,
    )

    seed_results: dict[str, dict[str, Any]] = {
        task: {
            "accuracy": None,
            "total": 0,
            "fast_fail": False,
            "floor": 0.0,
        }
        for task in lm_only_tasks
    }
    if not args.no_fast_fail and seedable_tasks:
        seed_results = run_seed_pass(
            base_url,
            args.model,
            seedable_tasks,
            seed_limits,
            floors,
            args.temperature,
            args.top_p,
            args.reasoning_effort,
            args.num_concurrent,
            args.heartbeat_interval,
        )
        if any(result["fast_fail"] for result in seed_results.values()):
            log("\nFast-fail threshold missed; skipping the full evaluation.")
            return write_summaries(
                run_id,
                args.model,
                base_url,
                tasks,
                seed_results,
                {},
                output_dir,
                targets,
                run_config,
            )

    direct_tasks = (
        seedable_tasks if args.direct_http or args.reasoning_effort else []
    )
    lm_tasks = [task for task in tasks if task not in direct_tasks]
    full_results: dict[str, dict[str, Any]] = {}
    if direct_tasks:
        full_results.update(
            run_direct_http_full(
                base_url,
                args.model,
                direct_tasks,
                full_limits,
                args.temperature,
                args.top_p,
                args.reasoning_effort,
                args.num_concurrent,
                args.heartbeat_interval,
            )
        )
    if lm_tasks:
        full_results.update(
            run_lm_eval_full(
                base_url,
                args.model,
                tokenizer,
                lm_tasks,
                full_limits,
                args.temperature,
                args.top_p,
                args.num_concurrent,
                args.heartbeat_interval,
                args.lm_eval_metadata,
                args.apply_chat_template,
            )
        )

    return write_summaries(
        run_id,
        args.model,
        base_url,
        tasks,
        seed_results,
        full_results,
        output_dir,
        targets,
        run_config,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except EvaluationError as exc:
        HeartbeatEmitter(0, exc.task).emit(
            "error",
            0,
            0,
            force=True,
            message=str(exc),
        )
        log(f"ERROR: {exc}")
        return EXIT_ERROR
    except KeyboardInterrupt:
        HeartbeatEmitter(0).emit(
            "error",
            0,
            0,
            force=True,
            message="Interrupted",
        )
        log("Interrupted.")
        return 130
    except Exception as exc:
        HeartbeatEmitter(0).emit(
            "error",
            0,
            0,
            force=True,
            message=str(exc),
        )
        log(f"ERROR: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
