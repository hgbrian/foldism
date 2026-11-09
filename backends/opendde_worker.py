# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Persistent OpenDDE process; model imports use the image's packaged Python."""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

RESPONSE_PREFIX = "__OPENDDE_RESPONSE__ "
logger = logging.getLogger(__name__)


class PersistentProcess:
    """Serialize requests to one child, restarting after an error or child exit."""

    def __init__(self, command: Callable[[], list[str]]):
        self.command = command
        self.process = None
        self.lock = threading.RLock()

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except BrokenPipeError:
                        pass

    def predict(self, request: dict, log: Callable[[str], None]) -> dict:
        with self.lock:
            try:
                if self.process is None or self.process.poll() is not None:
                    self.close()
                    self.process = subprocess.Popen(
                        self.command(),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                process = self.process
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                for raw in process.stdout:
                    if raw.startswith(RESPONSE_PREFIX):
                        response = json.loads(raw.removeprefix(RESPONSE_PREFIX))
                        if response.get("status") != "complete":
                            raise RuntimeError(response.get("error", "OpenDDE failed"))
                        return response
                    log(raw.rstrip())
                raise RuntimeError(f"OpenDDE worker exited with code {process.wait()}")
            except BaseException:
                self.close()
                raise


def validate_outputs(output_path: Path, num_samples: int) -> None:
    if num_samples < 1:
        raise ValueError("OpenDDE requires at least one sample")
    summaries = list(output_path.rglob("*_summary_confidence_sample_*.json"))
    if len(summaries) != num_samples:
        errors = [p.read_text() for p in output_path.rglob("*.txt")]
        raise RuntimeError(
            f"OpenDDE produced {len(summaries)}/{num_samples} samples: {errors}"
        )
    for summary in summaries:
        if not isinstance(json.loads(summary.read_text()), dict):
            raise TypeError(f"Invalid OpenDDE confidence output: {summary}")
        cif = summary.with_name(
            summary.name.replace("_summary_confidence", "")
        ).with_suffix(".cif")
        if not cif.is_file() or not cif.stat().st_size:
            raise RuntimeError(f"Missing or empty OpenDDE structure: {cif}")


class ModelSession:
    def __init__(self, create_runner, infer, cleanup):
        self.create_runner = create_runner
        self.infer = infer
        self.cleanup = cleanup
        self.runner = None
        self.settings = None
        self.initialization_id = None
        self.request_index = 0

    def predict(self, request: dict) -> dict:
        os.environ["FOLDISM_CONFIDENCE_DIR"] = request["output_path"] if request.get("download_confidence_data", True) else ""
        os.environ["FOLDISM_ACTIFPTM"] = (
            "1" if request.get("calculate_actifptm", True) else "0"
        )
        settings = request["settings"]
        initialized = self.runner is None or settings != self.settings
        if initialized:
            self.runner = None
            self.cleanup()
            self.runner = self.create_runner(
                dump_dir=request["output_path"], **settings
            )
            self.settings = settings.copy()
            self.initialization_id = uuid4().hex
            logger.info("OpenDDE model initialized: %s", self.initialization_id)
        runner = self.runner
        configs = runner.configs
        configs.input_json_path = request["input_path"]
        configs.dump_dir = request["output_path"]
        runner.init_basics()
        runner.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )
        started = time.monotonic()
        self.infer(runner, configs)
        validate_outputs(Path(request["output_path"]), settings["n_sample"])
        self.request_index += 1
        return {
            "status": "complete",
            "initialization_id": self.initialization_id,
            "model_reused": not initialized,
            "request_index": self.request_index,
            "prediction_seconds": time.monotonic() - started,
        }


def main():
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/opt/foldism")
    # These modules belong to the packaged OpenDDE interpreter.
    confidence_module = importlib.import_module("opendde.model.sample_confidence")
    importlib.import_module("confidence").install_native(
        confidence_module, "opendde"
    )
    cleanup_cuda_memory = importlib.import_module(
        "opendde.utils.torch_utils"
    ).cleanup_cuda_memory
    get_default_runner = importlib.import_module(
        "runner.batch_inference"
    ).get_default_runner
    infer_predict = importlib.import_module("runner.inference").infer_predict

    logging.basicConfig(level=logging.INFO)
    session = ModelSession(get_default_runner, infer_predict, cleanup_cuda_memory)
    for line in sys.stdin:
        try:
            result = session.predict(json.loads(line))
        except Exception:
            logger.exception("OpenDDE prediction failed")
            print(
                RESPONSE_PREFIX
                + json.dumps({"status": "error", "error": traceback.format_exc()}),
                flush=True,
            )
            return
        print(RESPONSE_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
