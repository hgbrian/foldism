"""Multi-algorithm protein folding web interface.

Uses backends for Boltz-2, Chai-1, Protenix, and AlphaFold2 predictions.

    uv run modal run foldism.py   # run
    uv run modal deploy foldism.py   # deploy
    uv run modal serve foldism.py    # dev
"""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from modal import Dict, Image, wsgi_app

from backends import (
    CACHE_VOLUME,
    FOLDING_APPS,
    alphafold_predict,
    app,
    boltz_predict,
    chai1_predict,
    convert_for_app,
    get_cache_key,
    get_cache_subdir,
    protenix_predict,
)
from backends.common import job_store

# =============================================================================
# Web Image
# =============================================================================

web_image = (
    Image.micromamba(python_version="3.12")
    .micromamba_install(["maxit==11.300"], channels=["conda-forge", "bioconda"])
    .pip_install(
        "flask==3.1.0",
        "polars==1.19.0",
        "gemmi==0.7.0",
        "pyyaml==6.0.2",
        "numpy==2.2.1",
    )
    .add_local_python_source("backends")
    .add_local_file("index.html", "/app/index.html")
)

# =============================================================================
# Helpers
# =============================================================================


def _build_result_payload(structure_bytes: bytes, fmt: str, files: dict, original_cif_bytes: bytes | None = None) -> dict:
    """Build result payload dict with base64-encoded data."""
    ext = "pdb" if fmt == "pdb" else "cif"
    data_payload = {
        "structure": base64.b64encode(structure_bytes).decode("ascii"),
        "ext": ext,
    }
    if original_cif_bytes:
        data_payload["original_cif"] = base64.b64encode(original_cif_bytes).decode("ascii")
    if "scores" in files and files["scores"]:
        data_payload["scores"] = base64.b64encode(files["scores"]).decode("ascii")
    if "all_files" in files and files["all_files"]:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in files["all_files"]:
                zf.writestr(name, content)
        data_payload["zip"] = base64.b64encode(zip_buffer.getvalue()).decode("ascii")
    return data_payload


def _pdb_to_cif(pdb_bytes: bytes) -> bytes:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "input.pdb"
        cif_path = Path(tmpdir) / "input.cif"
        pdb_path.write_bytes(pdb_bytes)
        subprocess.run(
            ["maxit", "-input", str(pdb_path), "-output", str(cif_path), "-o", "1"],
            check=True,
            capture_output=True,
        )
        return cif_path.read_bytes()


def _cif_to_pdb(cif_bytes: bytes) -> bytes:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        in_cif = Path(tmpdir) / "input.cif"
        out_pdb = Path(tmpdir) / "output.pdb"
        in_cif.write_bytes(cif_bytes)
        subprocess.run(
            ["maxit", "-input", str(in_cif), "-output", str(out_pdb), "-o", "2"],
            check=True,
            capture_output=True,
        )
        return out_pdb.read_bytes()


def _parse_chai1_npz(npz_bytes: bytes) -> dict:
    """Parse Chai-1 NPZ scores file and return JSON-serializable dict."""
    import io

    import numpy as np

    data = np.load(io.BytesIO(npz_bytes))
    scores = {}
    for key in data.keys():
        val = data[key]
        if val.dtype == bool:
            scores[key] = bool(val.item()) if val.size == 1 else val.tolist()
        else:
            scores[key] = val.tolist()
    return scores


def _find_best_model_index(algo: str, outputs: list[tuple]) -> int:
    """Find best model index for given algorithm based on confidence scores."""
    files = {str(path): content for path, content in outputs}

    if algo == "chai1":
        best_idx, best_score = 0, -float("inf")
        for i in range(5):
            score_file = f"scores.model_idx_{i}.npz"
            if score_file in files:
                scores = _parse_chai1_npz(files[score_file])
                score = scores.get("aggregate_score", [0])[0]
                if score > best_score:
                    best_score, best_idx = score, i
        return best_idx

    elif algo == "boltz2":
        best_idx, best_score = 0, -float("inf")
        for path, content in outputs:
            path_str = str(path)
            if (
                "confidence_" in path_str
                and "_model_" in path_str
                and path_str.endswith(".json")
            ):
                try:
                    scores = json.loads(content)
                    score = scores.get("confidence_score", 0)
                    idx = int(path_str.split("_model_")[1].split(".")[0])
                    if score > best_score:
                        best_score, best_idx = score, idx
                except:
                    pass
        return best_idx

    elif algo in ("protenix", "protenix-mini"):
        best_idx, best_score = 0, -float("inf")
        for path, content in outputs:
            path_str = str(path)
            if "summary_confidence_sample_" in path_str and path_str.endswith(".json"):
                try:
                    scores = json.loads(content)
                    score = scores.get("ranking_score", 0)
                    idx = int(
                        path_str.split("summary_confidence_sample_")[1].split(".")[0]
                    )
                    if score > best_score:
                        best_score, best_idx = score, idx
                except:
                    pass
        return best_idx

    return 0


def _select_best_model(algo: str, outputs: list[tuple]) -> dict[str, bytes]:
    if not outputs:
        raise ValueError(f"{algo}: No output files")

    files = {str(path): content for path, content in outputs}
    result = {}

    if algo == "chai1":
        best_idx = _find_best_model_index(algo, outputs)
        cif_key = f"pred.model_idx_{best_idx}.cif"
        if cif_key in files:
            result["structure.cif"] = files[cif_key]
        npz_key = f"scores.model_idx_{best_idx}.npz"
        if npz_key in files:
            scores = _parse_chai1_npz(files[npz_key])
            result["scores.json"] = json.dumps(scores).encode()

    elif algo == "boltz2":
        best_idx = _find_best_model_index(algo, outputs)
        for path, content in outputs:
            path_str = str(path)
            if f"_model_{best_idx}.cif" in path_str or path_str.endswith(
                f"model_{best_idx}.cif"
            ):
                result["structure.cif"] = content
            elif f"confidence_" in path_str and f"_model_{best_idx}.json" in path_str:
                result["scores.json"] = content

    elif algo in ("protenix", "protenix-mini"):
        best_idx = _find_best_model_index(algo, outputs)
        for path, content in outputs:
            path_str = str(path)
            if f"_sample_{best_idx}.cif" in path_str:
                result["structure.cif"] = content
            elif f"summary_confidence_sample_{best_idx}.json" in path_str:
                result["scores.json"] = content

    elif algo == "alphafold2":
        for path, content in outputs:
            path_str = str(path)
            if not content:
                continue
            if "ranked_0.pdb" in path_str or ("rank_001" in path_str and path_str.endswith(".pdb")):
                result["structure.cif"] = _pdb_to_cif(content)
            elif "ranking_debug.json" in path_str:
                result["scores.json"] = content

    return result


def _build_method_params(
    method: str, converted_input: str, use_msa: bool, input_name: str | None = None
) -> dict[str, Any]:
    """Build params dict for a method (centralized to avoid duplication)."""
    # Extract hash from converted input for naming
    if input_name is None:
        first_line = converted_input.split("\n")[0]
        # Look for 6-char hex hash pattern, or fall back to "input"
        match = re.search(r"[a-f0-9]{6}", first_line)
        input_name = match.group(0) if match else "input"

    if method == "boltz2":
        # Boltz always uses MSA server
        return {"input_str": converted_input, "use_msa": True}
    elif method == "chai1":
        return {
            "input_str": converted_input,
            "input_name": f"{input_name}.faa",
            "use_msa_server": use_msa,
        }
    elif method == "protenix":
        return {
            "input_str": converted_input,
            "input_name": input_name,
            "use_msa": use_msa,
        }
    elif method == "protenix-mini":
        return {
            "input_str": converted_input,
            "input_name": input_name,
            "model": "protenix_mini",
            "use_msa": use_msa,
        }
    elif method == "alphafold2":
        return {"input_str": converted_input, "input_name": f"{input_name}.fasta"}
    else:
        raise ValueError(f"Unknown method: {method}")


@app.function(image=web_image, timeout=60 * 60, volumes={"/cache": CACHE_VOLUME})
def run_algorithm(
    algo: str, fasta_str: str, run_name: str, use_msa: bool = True
) -> tuple[dict[str, bytes], list[tuple]]:
    """Run a folding algorithm and return (best_files, all_outputs).

    Runs on Modal with web_image so numpy/pyyaml are available.
    """
    converted = convert_for_app(fasta_str, algo)
    params = _build_method_params(algo, converted, use_msa, run_name)

    if algo == "boltz2":
        outputs = boltz_predict.remote(params=params)
    elif algo == "chai1":
        outputs = chai1_predict.remote(params=params)
    elif algo == "protenix":
        outputs = protenix_predict.remote(params=params)
    elif algo == "protenix-mini":
        outputs = protenix_predict.remote(params=params)
    elif algo == "alphafold2":
        outputs = alphafold_predict.remote(params=params)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    best = _select_best_model(algo, outputs)
    # Convert Path objects to strings for serialization
    all_outputs = [(str(path), content) for path, content in outputs]
    return best, all_outputs


# =============================================================================
# CLI Entry Point
# =============================================================================


@app.local_entrypoint()
def main(
    input_faa: str,
    algorithms: str | None = None,
    run_name: str | None = None,
    out_dir: str = "./out/foldism",
    keep_all: bool = True,
    use_msa: bool = True,
):
    """Run multiple folding algorithms on the same input."""
    with open(input_faa) as f:
        input_str = f.read()

    if run_name is None:
        run_name = Path(input_faa).stem

    if algorithms is None:
        algos_to_run = list(FOLDING_APPS.keys())
    else:
        algos_to_run = [a.strip() for a in algorithms.split(",")]
        for algo in algos_to_run:
            if algo not in FOLDING_APPS:
                raise ValueError(f"Unknown algorithm: {algo}")

    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running: {', '.join(algos_to_run)}")
    print(f"Output: {run_dir}")

    for algo in algos_to_run:
        app_def = FOLDING_APPS[algo]
        print(f"\n{'=' * 60}\nRunning {app_def.name}...\n{'=' * 60}")

        best, all_outputs = run_algorithm.remote(algo, input_str, run_name, use_msa=use_msa)

        for key, content in best.items():
            ext = key.split(".")[-1]
            if key.startswith("structure"):
                out_path = run_dir / f"{run_name}.{algo}.{ext}"
            else:
                out_path = run_dir / f"{run_name}.{algo}.{key}"
            out_path.write_bytes(content)
            print(f"  -> {out_path.name}")

        if keep_all:
            algo_dir = run_dir / algo
            algo_dir.mkdir(parents=True, exist_ok=True)
            for out_file, out_content in all_outputs:
                out_path = algo_dir / out_file
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(out_content or b"")

    print(f"\n{'=' * 60}\nComplete! Results in: {run_dir}\n{'=' * 60}")


# =============================================================================
# Web Interface (HTML template in index.html)
# =============================================================================


def _superpose_structures(
    structures: dict[str, bytes], reference_key: str | None = None
) -> dict[str, bytes]:
    import gemmi

    if len(structures) <= 1:
        return structures

    keys = list(structures.keys())
    ref_key = reference_key or keys[0]
    if ref_key not in structures:
        ref_key = keys[0]

    ref_doc = gemmi.cif.read_string(structures[ref_key].decode())
    ref_st = gemmi.make_structure_from_block(ref_doc[0])
    ref_model = ref_st[0]

    ref_chain = None
    for chain in ref_model:
        polymer = chain.get_polymer()
        if polymer and polymer.check_polymer_type() == gemmi.PolymerType.PeptideL:
            ref_chain = chain
            break

    if not ref_chain:
        return structures

    ref_polymer = ref_chain.get_polymer()
    ptype = ref_polymer.check_polymer_type()

    result = {ref_key: structures[ref_key]}

    for key, cif_bytes in structures.items():
        if key == ref_key:
            continue

        try:
            doc = gemmi.cif.read_string(cif_bytes.decode())
            st = gemmi.make_structure_from_block(doc[0])
            model = st[0]

            target_chain = None
            for chain in model:
                polymer = chain.get_polymer()
                if (
                    polymer
                    and polymer.check_polymer_type() == gemmi.PolymerType.PeptideL
                ):
                    target_chain = chain
                    break

            if not target_chain:
                result[key] = cif_bytes
                continue

            target_polymer = target_chain.get_polymer()
            sup = gemmi.calculate_superposition(
                ref_polymer, target_polymer, ptype, gemmi.SupSelect.CaP, trim_cycles=3
            )
            print(f"[align] {key}: RMSD={sup.rmsd:.2f}, matched={sup.count} atoms")

            for m in st:
                for chain in m:
                    for residue in chain:
                        for atom in residue:
                            atom.pos = sup.transform.apply(atom.pos)

            st.update_mmcif_block(doc[0])
            result[key] = doc.as_string().encode()
        except Exception as e:
            print(f"Failed to superpose {key}: {e}")
            result[key] = cif_bytes

    return result


def _check_cache_inline(method: str, params: dict) -> list | None:
    """Check cache inline without spawning a container."""
    from pathlib import Path

    cache_key = get_cache_key(method, params)
    cache_subdir = get_cache_subdir(method)
    cache_path = Path(f"/cache/{cache_subdir}/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    if not cache_marker.exists():
        print(f"[cache-check] {method} MISS: {cache_key}")
        return None

    print(f"[cache-check] {method} HIT: {cache_key}")
    CACHE_VOLUME.reload()

    return [
        (str(f.relative_to(cache_path)), f.read_bytes())
        for f in cache_path.glob("**/*")
        if f.is_file() and f.name != "_COMPLETE"
    ]


def _check_cache(method: str, fasta_str: str, use_msa: bool) -> list | None:
    """Check cache for method, converting input and building params."""
    converted = convert_for_app(fasta_str, method)
    params = _build_method_params(method, converted, use_msa)
    return _check_cache_inline(method, params)


@app.function(image=web_image, timeout=60 * 60, volumes={"/cache": CACHE_VOLUME})
def fold_structure_web(
    fasta_str: str, method: str, use_msa: bool = True, job_id: str | None = None
) -> tuple[str, dict | None, str | None]:
    """Run a single folding method for web interface."""
    try:
        converted = convert_for_app(fasta_str, method)
        params = _build_method_params(method, converted, use_msa)

        # Check cache first - avoids spawning GPU container if cached
        cached = _check_cache_inline(method, params)
        if cached is not None:
            print(f"[{method}] Returning cached result ({len(cached)} files)")
            outputs = [(Path(f), c) for f, c in cached]
        elif method == "boltz2":
            outputs = boltz_predict.remote(params=params, job_id=job_id)
        elif method == "chai1":
            outputs = chai1_predict.remote(params=params, job_id=job_id)
        elif method == "protenix":
            outputs = protenix_predict.remote(params=params, job_id=job_id)
        elif method == "protenix-mini":
            outputs = protenix_predict.remote(params=params, job_id=job_id)
        elif method == "alphafold2":
            outputs = alphafold_predict.remote(params=params, job_id=job_id)
        else:
            return (method, None, f"Unknown method: {method}")

        print(f"[{method}] Got {len(outputs)} output files")

        # Use shared result processing logic
        files = _process_outputs_to_files(method, outputs)
        if not files or "structure" not in files:
            return (method, None, f"No structure in {len(outputs)} files")

        files["all_files"] = [(str(f), c) for f, c in outputs if c]
        return (method, files, None)

    except Exception as e:
        import traceback

        traceback.print_exc()
        return (method, None, str(e))


def _process_outputs_to_files(method: str, outputs: list) -> dict | None:
    """Process raw outputs into files dict with structure and scores."""
    import zipfile
    from io import BytesIO

    files = {}
    outputs_dict = {str(f): c for f, c in outputs}

    if method == "boltz2":
        best_idx = _find_best_model_index(method, outputs)
        for path, content in outputs:
            path_str = str(path)
            if f"_model_{best_idx}.cif" in path_str or path_str.endswith(
                f"model_{best_idx}.cif"
            ):
                files["structure"] = (content, "cif")
            elif "confidence_" in path_str and f"_model_{best_idx}.json" in path_str:
                files["scores"] = content

    elif method == "chai1":
        best_idx = _find_best_model_index(method, outputs)
        cif_key = f"pred.model_idx_{best_idx}.cif"
        if cif_key in outputs_dict:
            files["structure"] = (outputs_dict[cif_key], "cif")
        npz_key = f"scores.model_idx_{best_idx}.npz"
        if npz_key in outputs_dict:
            scores = _parse_chai1_npz(outputs_dict[npz_key])
            files["scores"] = json.dumps(scores).encode()

    elif method in ("protenix", "protenix-mini"):
        best_idx = _find_best_model_index(method, outputs)
        for path, content in outputs:
            path_str = str(path)
            if f"_sample_{best_idx}.cif" in path_str:
                files["structure"] = (content, "cif")
            elif f"summary_confidence_sample_{best_idx}.json" in path_str:
                files["scores"] = content

    elif method == "alphafold2":
        for path, content in outputs:
            path_str = str(path)
            if not content:
                continue
            if "ranked_0.pdb" in path_str or ("rank_001" in path_str and path_str.endswith(".pdb")):
                files["structure"] = (_pdb_to_cif(content), "cif")
                print(f"[alphafold2] Found structure: {path_str}")
            elif "ranking_debug.json" in path_str:
                files["scores"] = content
            elif "_scores_rank_001" in path_str and path_str.endswith(".json"):
                if "scores" not in files:
                    files["scores"] = content

    if "structure" not in files:
        return None

    files["all_files"] = [(str(f), c) for f, c in outputs if c]
    return files


@app.function(image=web_image, timeout=60 * 60, volumes={"/cache": CACHE_VOLUME})
def run_folding_job(job_id: str, fasta: str, methods: list[str], use_msa: bool):
    """Background orchestrator for folding jobs (avoids 10-min SSE timeout)."""
    import hashlib
    import time

    def update_job(
        progress: int,
        status: str,
        logs: list[dict] | None = None,
        results: list[dict] | None = None,
        done: bool = False,
    ):
        # Merge with existing state to preserve backend logs (boltz_logs, chai1_logs, etc.)
        state = job_store.get(job_id, {})
        state["progress"] = progress
        state["status"] = status
        state["done"] = done
        if logs is not None:
            state["logs"] = logs
        if results is not None:
            # APPEND to existing results, don't replace
            existing_results = state.get("results", [])
            existing_results.extend(results)
            state["results"] = existing_results
        job_store[job_id] = state

    total = len(methods)
    logs = [{"msg": "Checking cache...", "cls": "info"}]
    update_job(5, f"Starting {total} methods...", logs)

    # Check cache for each method first
    cached_results = {}
    methods_to_run = []
    for method in methods:
        cached = _check_cache_inline_for_job(method, fasta, use_msa)
        if cached is not None:
            cached_results[method] = [(Path(f), c) for f, c in cached]
            logs.append(
                {"msg": f"{FOLDING_APPS[method].name}: cached", "cls": "success"}
            )
        else:
            methods_to_run.append(method)
            # Boltz always uses MSA, others respect checkbox
            if method == "boltz2":
                msa_note = " (with MSA)"
            else:
                msa_note = " (with MSA)" if use_msa else ""
            logs.append(
                {
                    "msg": f"Running {FOLDING_APPS[method].name}{msa_note}",
                    "cls": "dim",
                    "method_key": method,
                }
            )
    update_job(10, f"Running {len(methods_to_run)} methods...", logs)

    # Spawn only methods that need computation
    handles = []
    for method in methods_to_run:
        handle = fold_structure_web.spawn(fasta, method, use_msa, job_id)
        handles.append((method, handle))

    all_results = {}
    structures_to_superpose = {}

    # Process cached results immediately - with alignment and incremental display
    for method, outputs in cached_results.items():
        files = _process_outputs_to_files(method, outputs)
        if files and "structure" in files:
            all_results[method] = (files, None)
            structure_bytes, fmt = files["structure"]

            if fmt == "cif":
                # Align to first structure
                if not structures_to_superpose:
                    structures_to_superpose[method] = structure_bytes
                    logs.append(
                        {
                            "msg": f"{FOLDING_APPS[method].name}: Using as reference for alignment",
                            "cls": "info",
                        }
                    )
                else:
                    try:
                        ref_key = list(structures_to_superpose.keys())[0]
                        ref_structure = structures_to_superpose[ref_key]
                        to_align = {ref_key: ref_structure, method: structure_bytes}
                        aligned = _superpose_structures(to_align, ref_key)
                        structure_bytes = aligned[method]
                        structures_to_superpose[method] = structure_bytes
                        logs.append(
                            {
                                "msg": f"{FOLDING_APPS[method].name}: Aligned to {FOLDING_APPS[ref_key].name}",
                                "cls": "info",
                            }
                        )
                    except Exception as e:
                        logs.append(
                            {
                                "msg": f"{FOLDING_APPS[method].name}: Alignment failed: {e}",
                                "cls": "warning",
                            }
                        )
                        structures_to_superpose[method] = structure_bytes

            # Build and send result immediately
            original_cif_bytes = structure_bytes if fmt == "cif" else None
            display_bytes = structure_bytes
            display_fmt = fmt
            if fmt == "cif":
                try:
                    display_bytes = _cif_to_pdb(structure_bytes)
                    display_fmt = "pdb"
                except Exception as e:
                    logs.append(
                        {"msg": f"{method}: CIF to PDB failed: {e}", "cls": "error"}
                    )

            data_payload = _build_result_payload(display_bytes, display_fmt, files, original_cif_bytes)

            result_to_send = [
                {
                    "method": FOLDING_APPS[method].name,
                    "method_key": method,
                    "data": data_payload,
                    "format": display_fmt,
                }
            ]
            logs.append({"method_complete": method})
            progress = int(10 + (len(all_results) / total) * 70)
            update_job(
                progress, f"{len(all_results)}/{total} complete", logs, result_to_send
            )

    pending = list(handles)
    completed = len(cached_results)
    max_wait_time = 60 * 20  # 20 minutes max wait per method
    start_time = {}
    for method, _ in handles:
        start_time[method] = __import__("time").time()

    while pending:
        still_pending = []
        progress = int(10 + (completed / total) * 70)
        status = f"{completed}/{total} complete"
        for method, handle in pending:
            # Check if method has exceeded max wait time
            elapsed = __import__("time").time() - start_time[method]
            if elapsed > max_wait_time:
                logs.append(
                    {
                        "msg": f"{FOLDING_APPS[method].name}: Timeout after {int(elapsed / 60)} minutes",
                        "cls": "error",
                        "method_complete": method,
                    }
                )
                all_results[method] = (
                    None,
                    f"Timeout after {int(elapsed / 60)} minutes",
                )
                completed += 1
                progress = int(10 + (completed / total) * 70)
                running = len(still_pending)
                status = f"{completed}/{total} complete" + (
                    f" ({running} running)" if running > 0 else ""
                )
                update_job(progress, status, logs)
                continue

            try:
                method_name, files, error = handle.get(timeout=1.0)
                completed += 1
                progress = int(10 + (completed / total) * 70)
                all_results[method] = (files, error)

                # Build and send result immediately
                result_to_send = None
                if not error and files and "structure" in files:
                    structure_bytes, fmt = files["structure"]
                    if fmt == "cif":
                        # Superpose to first structure immediately
                        if not structures_to_superpose:
                            # First structure - use as reference
                            structures_to_superpose[method] = structure_bytes
                            logs.append(
                                {
                                    "msg": f"{FOLDING_APPS[method].name}: Using as reference for alignment",
                                    "cls": "info",
                                }
                            )
                        else:
                            # Superpose this structure to the reference
                            try:
                                ref_key = list(structures_to_superpose.keys())[0]
                                ref_structure = structures_to_superpose[ref_key]
                                to_align = {
                                    ref_key: ref_structure,
                                    method: structure_bytes,
                                }
                                aligned = _superpose_structures(to_align, ref_key)
                                structure_bytes = aligned[method]
                                structures_to_superpose[method] = structure_bytes
                                logs.append(
                                    {
                                        "msg": f"{FOLDING_APPS[method].name}: Aligned to {FOLDING_APPS[ref_key].name}",
                                        "cls": "info",
                                    }
                                )
                            except Exception as e:
                                logs.append(
                                    {
                                        "msg": f"{FOLDING_APPS[method].name}: Alignment failed: {e}",
                                        "cls": "warning",
                                    }
                                )
                                structures_to_superpose[method] = structure_bytes

                    # Convert to PDB for viewer
                    original_cif_bytes = structure_bytes if fmt == "cif" else None
                    if fmt == "cif":
                        try:
                            structure_bytes = _cif_to_pdb(structure_bytes)
                            fmt = "pdb"
                        except Exception as e:
                            logs.append(
                                {
                                    "msg": f"{method}: CIF to PDB failed: {e}",
                                    "cls": "error",
                                }
                            )

                    data_payload = _build_result_payload(structure_bytes, fmt, files, original_cif_bytes)

                    result_to_send = [
                        {
                            "method": FOLDING_APPS[method].name,
                            "method_key": method,
                            "data": data_payload,
                            "format": fmt,
                        }
                    ]

                if error:
                    logs.append(
                        {
                            "msg": f"{FOLDING_APPS[method].name}: {error}",
                            "cls": "error",
                            "method_complete": method,
                        }
                    )
                else:
                    logs.append({"method_complete": method})

                running = len(still_pending)
                status = f"{completed}/{total} complete" + (
                    f" ({running} running)" if running > 0 else ""
                )
                update_job(progress, status, logs, result_to_send)
            except TimeoutError:
                still_pending.append((method, handle))

        pending = still_pending

        # Check for and forward logs from all running methods
        try:
            state = job_store.get(job_id, {})
            log_configs = [
                ("boltz2", "boltz_logs", "Boltz"),
                ("chai1", "chai1_logs", "Chai-1"),
                ("protenix", "protenix_logs", "Protenix"),
                ("protenix-mini", "protenix_mini_logs", "Protenix-Mini"),
                ("alphafold2", "alphafold2_logs", "AlphaFold2"),
            ]

            for method_key, log_key, display_name in log_configs:
                # Forward logs for all selected methods, not just pending (catches final logs)
                if method_key in methods:
                    if log_key in state and state[log_key]:
                        new_logs = state[log_key]
                        last_count_attr = f"_last_{log_key}_count"
                        last_count = getattr(update_job, last_count_attr, 0)
                        if len(new_logs) > last_count:
                            logs_to_send = [
                                {
                                    "msg": f"[{display_name}] {line}",
                                    "cls": "dim",
                                    "method_key": method_key,
                                }
                                for line in new_logs[last_count:]
                            ]
                            logs.extend(logs_to_send)
                            setattr(update_job, last_count_attr, len(new_logs))
            # Send accumulated logs
            update_job(progress, status, logs)
        except Exception as e:
            print(f"[orchestrator] Failed to read logs: {e}")

        if pending:
            time.sleep(2)

    # Superposition now happens incrementally as each structure arrives (removed batch superposition)

    # Build final results (use aligned structures if available)
    results = []
    for method, (files, error) in all_results.items():
        if error or not files or "structure" not in files:
            continue

        structure_bytes, fmt = files["structure"]
        # Use aligned structure if available
        if method in structures_to_superpose:
            structure_bytes = structures_to_superpose[method]
        original_cif_bytes = structure_bytes if fmt == "cif" else None

        if fmt == "cif":
            try:
                structure_bytes = _cif_to_pdb(structure_bytes)
                fmt = "pdb"
            except Exception as e:
                logs.append(
                    {"msg": f"{method}: CIF to PDB failed: {e}", "cls": "error"}
                )

        data_payload = _build_result_payload(structure_bytes, fmt, files, original_cif_bytes)

        results.append(
            {
                "method": FOLDING_APPS[method].name,
                "method_key": method,
                "data": data_payload,
                "format": fmt,
            }
        )

    # Combined zip is built client-side from individual results (to avoid Modal Dict size limits)
    update_job(100, "Complete", logs, results, done=True)


def _check_cache_inline_for_job(
    method: str, fasta_str: str, use_msa: bool
) -> list | None:
    """Check cache inline for job orchestrator."""
    return _check_cache(method, fasta_str, use_msa)


@app.function(image=web_image, timeout=60 * 60, volumes={"/cache": CACHE_VOLUME})
@wsgi_app()
def web():
    from io import BytesIO
    from pathlib import Path

    from flask import Flask, Response, abort, request, send_file

    flask_app = Flask(__name__)

    # Simple session file cache with TTL cleanup
    session_files: dict[str, tuple[float, bytes]] = {}
    SESSION_TTL = 3600  # 1 hour
    SESSION_MAX = 1000

    def store_file(name: str, data: bytes):
        import time
        now = time.time()
        # Cleanup expired
        expired = [k for k, (ts, _) in session_files.items() if now - ts > SESSION_TTL]
        for k in expired:
            del session_files[k]
        # Enforce max size
        while len(session_files) >= SESSION_MAX:
            oldest = min(session_files, key=lambda k: session_files[k][0])
            del session_files[oldest]
        session_files[name] = (now, data)

    def get_file(name: str) -> bytes | None:
        import time
        if name not in session_files:
            return None
        ts, data = session_files[name]
        if time.time() - ts > SESSION_TTL:
            del session_files[name]
            return None
        return data

    def check_cache_in_server(
        method: str, fasta_str: str, use_msa: bool
    ) -> list | None:
        """Check cache directly in the web server - no container spawn needed."""
        return _check_cache(method, fasta_str, use_msa)

    @flask_app.route("/", methods=["GET"])
    def home():
        return Path("/app/index.html").read_text()

    @flask_app.route("/fold", methods=["POST"])
    def fold():
        """Spawn a folding job and return job_id (polling pattern to avoid 10-min SSE timeout)."""
        import uuid

        from flask import jsonify

        fasta = request.form.get("fasta", "").strip()
        methods = request.form.getlist("method")
        use_msa = request.form.get("use_msa", "true").lower() == "true"

        if not fasta:
            return jsonify({"error": "No sequence"}), 400

        if not methods:
            methods = ["boltz2"]

        job_id = str(uuid.uuid4())[:12]
        job_store[job_id] = {"progress": 0, "status": "Starting...", "done": False}

        # Spawn the orchestrator in the background
        run_folding_job.spawn(job_id, fasta, methods, use_msa)

        return jsonify({"job_id": job_id})

    @flask_app.route("/status/<job_id>")
    def status(job_id):
        """Get job status for polling."""
        from flask import jsonify

        state = job_store.get(job_id)
        if not state:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(state)

    # Legacy SSE endpoint (kept for backwards compatibility)
    @flask_app.route("/fold-sse", methods=["POST"])
    def fold_sse():
        fasta = request.form.get("fasta", "").strip()
        methods = request.form.getlist("method")
        use_msa = request.form.get("use_msa", "true").lower() == "true"

        if not fasta:
            return Response(
                "data: " + json.dumps({"error": "No sequence"}) + "\n\n",
                mimetype="text/event-stream",
            )

        if not methods:
            methods = ["boltz2"]

        def generate():
            import time
            from pathlib import Path

            total = len(methods)
            yield f"data: {json.dumps({'progress': 5, 'status': f'Starting {total} methods...', 'log': f'Checking cache...', 'log_class': 'info'})}\n\n"

            # Check cache for each method first (instant, no container spawn)
            cached_results = {}
            methods_to_run = []
            for method in methods:
                cached = check_cache_in_server(method, fasta, use_msa)
                if cached is not None:
                    cached_results[method] = [(Path(f), c) for f, c in cached]
                    yield f"data: {json.dumps({'log': f'{FOLDING_APPS[method].name}: cached', 'log_class': 'success'})}\n\n"
                else:
                    methods_to_run.append(method)

            # Spawn only methods that need computation
            handles = []
            for method in methods_to_run:
                handle = fold_structure_web.spawn(fasta, method, use_msa)
                handles.append((method, handle))
                yield f"data: {json.dumps({'log': f'Running {FOLDING_APPS[method].name}', 'log_class': 'dim', 'method_key': method})}\n\n"

            all_results = {}
            structures_to_superpose = {}

            # Process cached results immediately
            for method, outputs in cached_results.items():
                files = _process_outputs_to_files(method, outputs)
                if files and "structure" in files:
                    all_results[method] = (files, None)
                    structure_bytes, fmt = files["structure"]
                    if fmt == "cif":
                        structures_to_superpose[method] = structure_bytes

            pending = list(handles)
            completed = len(cached_results)  # Count cached results as already completed
            last_heartbeat = time.time()

            while pending:
                still_pending = []
                for method, handle in pending:
                    try:
                        method_name, files, error = handle.get(timeout=0.5)
                        completed += 1
                        progress = int((completed / total) * 80)
                        all_results[method] = (files, error)

                        if error:
                            yield f"data: {json.dumps({'progress': progress, 'method_complete': method, 'method_error': True})}\n\n"
                        elif files and "structure" in files:
                            structure_bytes, fmt = files["structure"]
                            if fmt == "cif":
                                structures_to_superpose[method] = structure_bytes
                            yield f"data: {json.dumps({'progress': progress, 'status': f'{completed}/{total} folded', 'method_complete': method})}\n\n"
                    except TimeoutError:
                        still_pending.append((method, handle))

                pending = still_pending
                now = time.time()
                if pending and (now - last_heartbeat) >= 5:
                    pending_names = ", ".join(FOLDING_APPS[m].name for m, _ in pending)
                    yield f"data: {json.dumps({'status': f'Running: {pending_names}...'})}\n\n"
                    last_heartbeat = now
                elif pending:
                    # SSE keepalive comment to prevent connection timeout
                    yield ": keepalive\n\n"
                if pending:
                    time.sleep(1)

            # Superposition now happens incrementally (removed batch alignment)

            import base64
            import uuid

            for method, (files, error) in all_results.items():
                if error or not files or "structure" not in files:
                    continue

                result_id = str(uuid.uuid4())[:8]
                structure_bytes, fmt = files["structure"]
                original_cif_bytes = structure_bytes if fmt == "cif" else None

                if fmt == "cif":
                    try:
                        structure_bytes = _cif_to_pdb(structure_bytes)
                        fmt = "pdb"
                    except Exception as e:
                        yield f"data: {json.dumps({'log': f'{method}: CIF to PDB failed: {e}', 'log_class': 'error'})}\n\n"

                ext = "pdb" if fmt == "pdb" else "cif"
                store_file(f"{result_id}.{ext}", structure_bytes)

                data_payload = _build_result_payload(structure_bytes, fmt, files, original_cif_bytes)

                yield f"data: {json.dumps({'progress': 100, 'status': 'Complete', 'result': {'method': FOLDING_APPS[method].name, 'method_key': method, 'data': data_payload, 'format': fmt}})}\n\n"

            # Combined zip is built client-side from individual results
            yield f"data: {json.dumps({'progress': 100, 'done': True})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    @flask_app.route("/result/<filename>")
    def get_result(filename):
        data = get_file(filename)
        if data is None:
            abort(404)
        mimetype = (
            "chemical/x-pdb"
            if filename.endswith(".pdb")
            else "application/zip"
            if filename.endswith(".zip")
            else "chemical/x-mmcif"
        )
        return send_file(BytesIO(data), mimetype=mimetype, download_name=filename)

    return flask_app
