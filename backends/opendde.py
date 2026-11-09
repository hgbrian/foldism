# /// script
# requires-python = ">=3.12"
# dependencies = ["modal"]
# ///
"""OpenDDE v1 structure prediction backend."""

from __future__ import annotations

import atexit
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from modal import Image

from .common import (
    CACHE_VOLUME,
    MODEL_VOLUME,
    TIMEOUT,
    app,
    extract_chain_sequences,
    fasta_iter,
    job_store,
    opendde_cache_key,
    write_log_line,
)
from .opendde_worker import PersistentProcess, validate_outputs

OPENDDE_MODEL_NAME = "opendde_v1"
OPENDDE_CHECKPOINT = "opendde.pt"
OPENDDE_ASSET_REVISION = "eddd563ce96571f784012edd8f045181c8f8627d"
OPENDDE_MODEL_ROOT = Path("/models/opendde") / OPENDDE_ASSET_REVISION
OPENDDE_CHECKPOINT_PATH = OPENDDE_MODEL_ROOT / "checkpoint" / OPENDDE_CHECKPOINT
OPENDDE_HF_BASE = (
    f"https://huggingface.co/aurekaresearch/OpenDDE/resolve/{OPENDDE_ASSET_REVISION}"
)
OPENDDE_COMMON_ASSETS = (
    "components.cif",
    "components.cif.rdkit_mol.pkl",
    "release_date_cache.json",
    "obsolete_to_successor.json",
)
OPENDDE_ASSET_MARKER = OPENDDE_MODEL_ROOT / f"_DOWNLOADED_{OPENDDE_ASSET_REVISION}"
OPENDDE_WORKER_PATH = "/root/opendde_worker.py"
OPENDDE_CACHE_ROOT = Path("/cache/opendde")


def _worker_command() -> list[str]:
    command = shutil.which("opendde")
    if command is None:
        raise FileNotFoundError("OpenDDE entry point is not installed")
    first_line = Path(command).read_text().splitlines()[0]
    if not first_line.startswith("#!"):
        raise RuntimeError(f"OpenDDE entry point has no interpreter: {command}")
    return [*shlex.split(first_line[2:]), "-u", OPENDDE_WORKER_PATH]


_OPENDDE_PROCESS = PersistentProcess(_worker_command)
atexit.register(_OPENDDE_PROCESS.close)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    subprocess.run(
        [
            "curl",
            "-fL",
            "--retry",
            "6",
            "--retry-delay",
            "5",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    if not partial.exists() or partial.stat().st_size == 0:
        raise RuntimeError(f"OpenDDE asset download was empty: {url}")
    partial.replace(destination)


def _download_opendde_assets() -> None:
    expected = [
        OPENDDE_CHECKPOINT_PATH,
        *(
            OPENDDE_MODEL_ROOT / "common" / filename
            for filename in OPENDDE_COMMON_ASSETS
        ),
    ]
    if OPENDDE_ASSET_MARKER.exists() and all(
        path.exists() and path.stat().st_size > 0 for path in expected
    ):
        print(f"[BUILD] OpenDDE assets present ({OPENDDE_ASSET_REVISION})")
        return
    _download(f"{OPENDDE_HF_BASE}/{OPENDDE_CHECKPOINT}", OPENDDE_CHECKPOINT_PATH)
    for filename in OPENDDE_COMMON_ASSETS:
        destination = OPENDDE_MODEL_ROOT / "common" / filename
        _download(f"{OPENDDE_HF_BASE}/common/{filename}", destination)
    missing = [
        str(path) for path in expected if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Missing or empty OpenDDE assets after download: {missing}")
    OPENDDE_ASSET_MARKER.touch()
    MODEL_VOLUME.commit()


opendde_image = (
    Image.from_registry("aurekaresearch/opendde:v1", add_python="3.12")
    .apt_install("curl")
    .env({"OPENDDE_ROOT_DIR": str(OPENDDE_MODEL_ROOT), "PYTHONUNBUFFERED": "1"})
    .run_function(
        _download_opendde_assets,
        volumes={"/models": MODEL_VOLUME},
        timeout=60 * 60,
    )
    .pip_install("gemmi==0.6.7", "numpy")
    .add_local_file(Path(__file__).with_name("opendde_worker.py"), OPENDDE_WORKER_PATH)
)

opendde_image = opendde_image.add_local_file(
    Path(__file__).with_name("confidence.py"), "/opt/foldism/confidence.py"
)


def _restore_query_sequence(source: Path, destination: Path, sequence: str) -> None:
    """Copy an A3M while replacing its first sequence row with the real query."""
    lines = source.read_text().replace("\x00", "").splitlines()
    query_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line and not line.startswith((">", "#"))
        ),
        None,
    )
    if query_index is None:
        raise ValueError(f"No query sequence found in MSA: {source}")
    if len(lines[query_index]) != len(sequence):
        raise ValueError(
            f"MSA query length mismatch for {source}: "
            f"{len(lines[query_index])} != {len(sequence)}"
        )
    lines[query_index] = sequence
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def _fasta_to_opendde_json(
    fasta_str: str,
    name: str,
    seed: int,
    msa_result: dict | None,
    msa_dir: Path,
) -> str:
    """Convert a protein-only FASTA to one OpenDDE inference job."""
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sequences: list[dict[str, Any]] = []
    chain_sequences = extract_chain_sequences(fasta_str)
    fasta_records = list(fasta_iter(fasta_str))
    if len(chain_sequences) != len(fasta_records):
        raise ValueError("OpenDDE accepts protein-only FASTA inputs")

    unpaired = (msa_result or {}).get("unpaired", {})
    for index, sequence in enumerate(chain_sequences):
        chain: dict[str, Any] = {
            "sequence": sequence,
            "count": 1,
            "id": [chain_ids[index] if index < len(chain_ids) else f"chain_{index}"],
        }
        if msa_result is not None:
            source_dir = unpaired.get(sequence)
            if not source_dir:
                raise ValueError(f"Missing unpaired MSA for protein chain {index}")
            source = Path(source_dir) / "merged.a3m"
            destination = msa_dir / f"chain_{index}.a3m"
            _restore_query_sequence(source, destination, sequence)
            chain["unpairedMsaPath"] = str(destination)
        sequences.append({"proteinChain": chain})

    if not sequences:
        raise ValueError("OpenDDE input contains no protein chains")
    return json.dumps(
        [{"name": name, "modelSeeds": [seed], "sequences": sequences}],
        indent=2,
    )


@app.function(
    image=opendde_image,
    timeout=max(TIMEOUT * 60, 2 * 60 * 60),
    gpu="A100-40GB",
    cpu=6,
    memory=65_536,
    volumes={"/models": MODEL_VOLUME, "/cache": CACHE_VOLUME},
)
def opendde_predict(
    params: dict[str, Any], overwrite: bool = False, job_id: str | None = None
) -> list:
    """Reuse OpenDDE's model in each warm container, with cached unpaired MSAs."""
    input_str = params["input_str"]
    input_name = params.get("input_name", "input")
    seed = params.get("seed", 42)
    num_samples = params.get("num_samples", 1)
    num_steps = params.get("num_steps", 200)
    num_cycles = params.get("num_cycles", 10)
    use_msa = params.get("use_msa", True)
    msa_result = params.get("msa_result") if use_msa else None
    log_key = "opendde_logs"

    MODEL_VOLUME.reload()
    CACHE_VOLUME.reload()
    if not OPENDDE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing OpenDDE checkpoint: {OPENDDE_CHECKPOINT_PATH}"
        )
    if use_msa and not msa_result:
        raise RuntimeError(
            "[opendde] use_msa=True but no msa_result provided. "
            "Pre-fetch Foldism unpaired MSAs or pass use_msa=False."
        )

    from .confidence import downloadable, finalize_native_outputs

    active = bool(params.get("calculate_actifptm", True) or params.get("download_confidence_data", True))

    from .confidence import cache_supports

    cache_key = opendde_cache_key(params)
    cache_path = OPENDDE_CACHE_ROOT / cache_key
    cache_marker = cache_path / "_COMPLETE"
    if (
        not overwrite
        and cache_marker.exists()
        and cache_supports(cache_path, bool(params.get("calculate_actifptm", True) or params.get("download_confidence_data", True)), bool(params.get("download_confidence_data", True)))
    ):
        msg = f"[opendde] Cache HIT: {cache_key}"
        print(msg)
        write_log_line(job_id, log_key, msg, "opendde")
        return [
            (path.relative_to(cache_path), path.read_bytes())
            for path in cache_path.glob("**/*")
            if path.is_file() and path.name != "_COMPLETE"
        ]

    msg = f"[opendde] Cache MISS: {cache_key} (MSA: {'unpaired' if use_msa else 'disabled'})"
    print(msg)
    write_log_line(job_id, log_key, msg, "opendde")

    with TemporaryDirectory() as temporary_directory:
        work_dir = Path(temporary_directory)
        input_path = work_dir / "input.json"
        output_path = work_dir / "out"
        input_path.write_text(
            _fasta_to_opendde_json(
                input_str,
                input_name,
                seed,
                msa_result,
                work_dir / "msas",
            )
        )

        log_buffer: list[str] = []

        def log_line(line: str) -> None:
            if not line:
                return
            print(line, flush=True)
            log_buffer.append(line)
            if len(log_buffer) >= 25 and job_id:
                state = job_store.get(job_id, {})
                state[log_key] = (state.get(log_key, []) + log_buffer)[-1000:]
                job_store[job_id] = state
                log_buffer.clear()

        try:
            runtime = _OPENDDE_PROCESS.predict(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "calculate_actifptm": active,
                    "download_confidence_data": bool(params.get("download_confidence_data", True)),
                    "settings": {
                        "model_name": OPENDDE_MODEL_NAME,
                        "load_checkpoint_path": str(OPENDDE_CHECKPOINT_PATH),
                        "n_cycle": num_cycles,
                        "n_step": num_steps,
                        "n_sample": num_samples,
                        "use_msa": use_msa,
                        "use_template": False,
                        "use_rna_msa": False,
                        "need_atom_confidence": True,
                    },
                },
                log_line,
            )
        finally:
            if log_buffer and job_id:
                state = job_store.get(job_id, {})
                state[log_key] = (state.get(log_key, []) + log_buffer)[-1000:]
                job_store[job_id] = state

        validate_outputs(output_path, num_samples)
        (output_path / "runtime_provenance.json").write_text(json.dumps(runtime))

        finalize_native_outputs(output_path, extract_chain_sequences(input_str), active)
        outputs = [
            (path.relative_to(output_path), path.read_bytes())
            for path in output_path.rglob("*")
            if path.is_file() and downloadable(path)
        ]
        if not outputs:
            raise RuntimeError("OpenDDE produced no output files")

        cache_path.mkdir(parents=True, exist_ok=True)
        for relative_path, content in outputs:
            destination = cache_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        if params.get("download_confidence_data", True) and not cache_supports(cache_path, True, True):
            raise ValueError("Requested confidence arrays are missing from prediction outputs")
        cache_marker.touch()
        CACHE_VOLUME.commit()
        return outputs
