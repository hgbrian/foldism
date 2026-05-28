"""Multi-algorithm protein folding web interface.

Backends: Boltz-2, Chai-1, Protenix, AlphaFold 2, OpenFold 3, ESMFold 2.

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
    boltz2_predict,
    chai1_predict,
    colabsearch_fetch,
    convert_for_app,
    esmfold2_predict,
    get_cache_key,
    get_cache_subdir,
    openfold3_predict,
    protenix_predict,
)
from backends.common import job_store

# =============================================================================
# Web Image
# =============================================================================

_web_image = (
    Image.micromamba(python_version="3.12")
    .micromamba_install(["maxit==11.300"], channels=["conda-forge", "bioconda"])
    .pip_install(
        "flask==3.1.0",
        "polars==1.19.0",
        "gemmi==0.7.4",
        "pyyaml==6.0.2",
        "numpy==2.2.1",
    )
    .add_local_python_source("backends")
    .add_local_file("index.html", "/app/index.html")
)
_overrides = Path(__file__).parent / "overrides.json"
web_image = _web_image.add_local_file(str(_overrides), "/app/overrides.json") if _overrides.exists() else _web_image

# =============================================================================
# Helpers
# =============================================================================


def _build_result_payload(
    structure_bytes: bytes, fmt: str, files: dict,
    original_cif_bytes: bytes | None = None,
    rmsd_ref: dict | None = None,
) -> dict:
    """Build result payload dict with base64-encoded data.

    `rmsd_ref` (optional) carries `{"rmsd": float, "n_atoms": int}` from
    superposing this prediction onto the reference structure.
    """
    ext = "pdb" if fmt == "pdb" else "cif"
    data_payload = {
        "structure": base64.b64encode(structure_bytes).decode("ascii"),
        "ext": ext,
    }
    if original_cif_bytes:
        data_payload["original_cif"] = base64.b64encode(original_cif_bytes).decode("ascii")
    if "scores" in files and files["scores"]:
        data_payload["scores"] = base64.b64encode(files["scores"]).decode("ascii")
    if rmsd_ref:
        data_payload["rmsd_ref"] = rmsd_ref
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


def _normalize_scores(algo: str, raw: dict) -> bytes:
    """Convert algo-native scores dict to unified schema bytes.

    Unified keys: method, iptm, ptm, plddt, ranking_score, has_clash, raw.
    Per-chain / matrix scores live under `raw` for downstream consumers.
    """
    def _first(v):
        return v[0] if isinstance(v, list) and v else v

    if algo == "protenix":
        unified = {
            "iptm": raw.get("iptm"),
            "ptm": raw.get("ptm"),
            "plddt": raw.get("plddt"),
            "ranking_score": raw.get("ranking_score"),
            "has_clash": bool(raw.get("has_clash", False)),
        }
    elif algo == "chai1":
        unified = {
            "iptm": _first(raw.get("iptm")),
            "ptm": _first(raw.get("ptm")),
            "plddt": None,
            "ranking_score": _first(raw.get("aggregate_score")),
            "has_clash": bool(raw.get("has_inter_chain_clashes", False)),
        }
    elif algo == "openfold3":
        plddt = raw.get("avg_plddt", raw.get("plddt"))
        if isinstance(plddt, list):
            plddt = sum(plddt) / len(plddt) if plddt else None
        unified = {
            "iptm": raw.get("iptm"),
            "ptm": raw.get("ptm"),
            "plddt": plddt,
            "ranking_score": raw.get("sample_ranking_score"),
            "has_clash": bool(raw.get("has_clash", False)),
        }
    elif algo == "esmfold2":
        plddt_raw = raw.get("plddt")
        if isinstance(plddt_raw, list):
            plddt_raw = (sum(plddt_raw) / len(plddt_raw)) if plddt_raw else None
        # ranking_score stays on the model-native 0-1 scale (matches the other
        # backends' ranking outputs). plddt is rescaled to 0-100 for cross-
        # backend pLDDT comparability.
        ranking_score = plddt_raw if isinstance(plddt_raw, (int, float)) else None
        plddt = plddt_raw * 100 if isinstance(plddt_raw, (int, float)) and 0 <= plddt_raw <= 1 else plddt_raw
        unified = {
            "iptm": raw.get("iptm"),
            "ptm": raw.get("ptm"),
            "plddt": plddt,
            "ranking_score": ranking_score,
            "has_clash": False,
        }
    elif algo == "boltz2":
        plddt = raw.get("complex_plddt") or raw.get("plddt")
        if isinstance(plddt, (int, float)) and 0 <= plddt <= 1:
            plddt = plddt * 100
        unified = {
            "iptm": raw.get("iptm") or raw.get("complex_iptm"),
            "ptm": raw.get("ptm") or raw.get("complex_ptm"),
            "plddt": plddt,
            "ranking_score": raw.get("confidence_score"),
            "has_clash": bool(raw.get("has_clashes", False)),
        }
    elif algo == "alphafold2":
        if "ptm" in raw or "iptm" in raw:
            plddt = raw.get("plddt")
            if isinstance(plddt, list):
                plddt = (sum(plddt) / len(plddt)) if plddt else None
            unified = {
                "iptm": raw.get("iptm"),
                "ptm": raw.get("ptm"),
                "plddt": plddt,
                "ranking_score": raw.get("iptm") if raw.get("iptm") is not None else raw.get("ptm"),
                "has_clash": False,
            }
        else:
            order = raw.get("order") or []
            best = order[0] if order else None
            plddts = raw.get("plddts") or {}
            iptm_ptm = raw.get("iptm+ptm") or {}
            unified = {
                "iptm": None,
                "ptm": iptm_ptm.get(best) if best else None,
                "plddt": plddts.get(best) if best else None,
                "ranking_score": iptm_ptm.get(best) if best in iptm_ptm else (plddts.get(best) if best else None),
                "has_clash": False,
            }
    else:
        unified = {}

    unified["method"] = algo
    unified["raw"] = raw
    return json.dumps(unified, indent=2).encode()


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
                except (KeyError, ValueError, IndexError, json.JSONDecodeError):
                    pass
        return best_idx

    elif algo == "protenix":
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
                except (KeyError, ValueError, IndexError, json.JSONDecodeError):
                    pass
        return best_idx

    elif algo == "openfold3":
        best_idx, best_score = 0, -float("inf")
        for path, content in outputs:
            path_str = str(path)
            if "_confidences_aggregated.json" in path_str:
                try:
                    scores = json.loads(content)
                    score = scores.get("avg_plddt", scores.get("plddt", 0))
                    if isinstance(score, list):
                        score = sum(score) / len(score) if score else 0
                    idx = int(path_str.split("_sample_")[1].split("_")[0])
                    if score > best_score:
                        best_score, best_idx = score, idx
                except (KeyError, ValueError, IndexError, json.JSONDecodeError):
                    pass
        return best_idx

    elif algo == "esmfold2":
        best_idx, best_score = 0, -float("inf")
        for path, content in outputs:
            path_str = str(path)
            if path_str.endswith("_scores.json") and "_sample_" in path_str:
                try:
                    scores = json.loads(content)
                    score = scores.get("plddt", 0)
                    idx = int(path_str.split("_sample_")[1].split("_")[0])
                    if score > best_score:
                        best_score, best_idx = score, idx
                except (KeyError, ValueError, IndexError, json.JSONDecodeError):
                    pass
        return best_idx

    return 0


def _select_best_model(algo: str, outputs: list[tuple]) -> dict[str, bytes]:
    if not outputs:
        raise ValueError(f"{algo}: No output files")

    files = {str(path): content for path, content in outputs}
    result = {}
    raw_scores: dict | None = None

    if algo == "chai1":
        best_idx = _find_best_model_index(algo, outputs)
        cif_key = f"pred.model_idx_{best_idx}.cif"
        if cif_key in files:
            result["structure.cif"] = files[cif_key]
        npz_key = f"scores.model_idx_{best_idx}.npz"
        if npz_key in files:
            raw_scores = _parse_chai1_npz(files[npz_key])
            result["scores.json"] = json.dumps(raw_scores).encode()

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
                raw_scores = json.loads(content)

    elif algo == "protenix":
        best_idx = _find_best_model_index(algo, outputs)
        for path, content in outputs:
            path_str = str(path)
            if f"_sample_{best_idx}.cif" in path_str:
                result["structure.cif"] = content
            elif f"summary_confidence_sample_{best_idx}.json" in path_str:
                result["scores.json"] = content
                raw_scores = json.loads(content)

    elif algo == "alphafold2":
        for path, content in outputs:
            path_str = str(path)
            if not content:
                continue
            if "ranked_0.pdb" in path_str or ("rank_001" in path_str and path_str.endswith(".pdb")):
                result["structure.cif"] = _pdb_to_cif(content)
            elif "ranking_debug.json" in path_str:
                result["scores.json"] = content
                raw_scores = json.loads(content)
            elif "_scores_rank_001" in path_str and path_str.endswith(".json"):
                if raw_scores is None:
                    result["scores.json"] = content
                    raw_scores = json.loads(content)

    elif algo == "openfold3":
        best_idx = _find_best_model_index(algo, outputs)
        for path, content in outputs:
            path_str = str(path)
            if f"_sample_{best_idx}_model.cif" in path_str:
                result["structure.cif"] = content
            elif f"_sample_{best_idx}_confidences_aggregated.json" in path_str:
                result["scores.json"] = content
                raw_scores = json.loads(content)

    elif algo == "esmfold2":
        best_idx = _find_best_model_index(algo, outputs)
        for path, content in outputs:
            path_str = str(path)
            if path_str.endswith(f"_sample_{best_idx}.cif"):
                result["structure.cif"] = content
            elif path_str.endswith(f"_sample_{best_idx}_scores.json"):
                result["scores.json"] = content
                raw_scores = json.loads(content)

    if raw_scores is not None:
        result["unified.json"] = _normalize_scores(algo, raw_scores)

    return result


def _extract_protein_sequences(fasta_str: str) -> list[str]:
    """Ordered protein chain sequences from FASTA, duplicates PRESERVED.

    Duplicates are kept so colabsearch sees the true chain count and fetches
    a paired MSA for homomers (e.g. [A, A] needs paired_dir, not just unpaired).
    Backends still naturally dedupe inside their own per-sequence loops.
    """
    from backends.common import extract_chain_sequences

    return extract_chain_sequences(fasta_str)


def _fetch_msas(protein_seqs: list[str]) -> dict:
    """Pre-fetch unpaired + paired MSAs via ColabSearch. Returns MsaResult dict."""
    if not protein_seqs:
        return {}
    n_unique = len(set(protein_seqs))
    print(f"Fetching MSAs for {len(protein_seqs)} chain(s), {n_unique} unique...")
    return colabsearch_fetch.remote(protein_seqs)


def _build_method_params(
    method: str, converted_input: str, use_msa: bool, input_name: str | None = None,
    msa_result: dict | None = None, original_fasta: str | None = None,
) -> dict[str, Any]:
    """Build params dict for a method (centralized to avoid duplication)."""
    # Extract hash from converted input for naming
    if input_name is None:
        first_line = converted_input.split("\n")[0]
        # Look for 6-char hex hash pattern, or fall back to "input"
        match = re.search(r"[a-f0-9]{6}", first_line)
        input_name = match.group(0) if match else "input"

    if method == "boltz2":
        # Boltz always uses MSA internally; `use_msa` here tracks the SOURCE
        # (True = pre-fetched ColabSearch, False = boltz's built-in server)
        # so the cache key separates the two.
        params = {"input_str": converted_input, "use_msa": use_msa}
        if msa_result and use_msa:
            params["msa_result"] = msa_result
            if original_fasta:
                params["original_fasta"] = original_fasta
        return params
    elif method == "chai1":
        params = {
            "input_str": converted_input,
            "input_name": f"{input_name}.faa",
            "use_msa_server": use_msa,
        }
        if msa_result and use_msa:
            params["msa_result"] = msa_result
        return params
    elif method == "protenix":
        params = {
            "input_str": converted_input,
            "input_name": input_name,
            "use_msa": use_msa,
        }
        if msa_result and use_msa:
            params["msa_result"] = msa_result
        return params
    elif method == "alphafold2":
        params: dict[str, Any] = {
            "input_str": converted_input,
            "input_name": f"{input_name}.fasta",
            "use_msa": use_msa,
        }
        if msa_result and use_msa:
            params["msa_result"] = msa_result
            # AF2 converter joins chains with ':' on a single FASTA line, which
            # _build_af2_queries can't parse back into chains. Pass the original
            # multi-record FASTA so it can recover the chain list with duplicates.
            if original_fasta:
                params["original_fasta"] = original_fasta
        return params
    elif method == "openfold3":
        params = {
            "input_str": converted_input,
            "input_name": input_name,
            "use_msa": use_msa,
        }
        if msa_result and use_msa:
            params["msa_result"] = msa_result
        return params
    elif method == "esmfold2":
        params = {
            "input_str": converted_input,
            "input_name": input_name,
            "use_msa": use_msa,
        }
        if msa_result and use_msa:
            params["msa_result"] = msa_result
        return params
    else:
        raise ValueError(f"Unknown method: {method}")


@app.function(image=web_image, timeout=60 * 60, volumes={"/cache": CACHE_VOLUME})
def run_algorithm(
    algo: str, fasta_str: str, run_name: str, use_msa: bool = True,
    msa_result: dict | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, bytes], list[tuple]]:
    """Run a folding algorithm and return (best_files, all_outputs).

    Runs on Modal with web_image so numpy/pyyaml are available.
    """
    converted = convert_for_app(fasta_str, algo)
    params = _build_method_params(algo, converted, use_msa, run_name, msa_result=msa_result, original_fasta=fasta_str)

    if algo == "boltz2":
        outputs = boltz2_predict.remote(params=params, overwrite=overwrite)
    elif algo == "chai1":
        outputs = chai1_predict.remote(params=params, overwrite=overwrite)
    elif algo == "protenix":
        outputs = protenix_predict.remote(params=params, overwrite=overwrite)
    elif algo == "alphafold2":
        outputs = alphafold_predict.remote(params=params, overwrite=overwrite)
    elif algo == "openfold3":
        outputs = openfold3_predict.remote(params=params, overwrite=overwrite)
    elif algo == "esmfold2":
        outputs = esmfold2_predict.remote(params=params, overwrite=overwrite)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    best = _select_best_model(algo, outputs)
    # Convert Path objects to strings for serialization
    all_outputs = [(str(path), content) for path, content in outputs]
    return best, all_outputs


# =============================================================================
# CLI Entry Point
# =============================================================================


MSA_BACKENDS = ("boltz2", "protenix", "chai1", "alphafold2", "openfold3", "esmfold2")


def _resolve_algorithms(algorithms: str | None) -> list[str]:
    if algorithms is None:
        return list(FOLDING_APPS.keys())
    algos = [a.strip() for a in algorithms.split(",")]
    for algo in algos:
        if algo not in FOLDING_APPS:
            raise ValueError(f"Unknown algorithm: {algo}")
    return algos


@app.local_entrypoint()
def main(
    input_faa: str | None = None,
    input_dir: str | None = None,
    algorithms: str | None = None,
    run_name: str | None = None,
    out_dir: str = "./out",
    keep_all: bool = True,
    use_msa: bool = True,
    overwrite: bool = False,
    pattern: str = "*.faa",
):
    """Run folding algorithms on one FASTA or a directory of them.

    Single file:
        modal run foldism.py --input-faa lys.faa --algorithms esmfold2

    Directory:
        modal run foldism.py --input-dir /tmp/fastas --algorithms protenix
    """
    if bool(input_faa) == bool(input_dir):
        raise ValueError("Pass exactly one of --input-faa or --input-dir")

    algos_to_run = _resolve_algorithms(algorithms)

    if input_dir:
        _run_batch(input_dir, algos_to_run, out_dir, pattern, use_msa, overwrite, keep_all)
        return

    with open(input_faa) as f:
        input_str = f.read()

    if run_name is None:
        run_name = Path(input_faa).stem

    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    msa_result = None
    if not use_msa and "boltz2" in algos_to_run:
        print("Note: dropping Boltz-2 from this run (Boltz only runs with MSA).")
        algos_to_run = [a for a in algos_to_run if a != "boltz2"]
    msa_backends = [a for a in algos_to_run if a in MSA_BACKENDS]
    if not use_msa and msa_backends:
        print()
        print("=" * 70)
        print("WARNING: --no-use-msa selected. Per-backend behavior:")
        print("  - AlphaFold 2:   single_sequence mode — accuracy drops a lot")
        print("  - Protenix:      no MSA           — accuracy drops a lot")
        print("  - OpenFold 3:    no MSA           — accuracy drops a lot")
        print("  - Chai-1:        no MSA, ESM2 embeddings only — moderate quality")
        print("  - ESMFold 2:     no MSA, ESMC embeddings only — its intended fast mode")
        print("=" * 70)
        print()
    if use_msa and msa_backends:
        protein_seqs = _extract_protein_sequences(input_str)
        if protein_seqs:
            msa_result = _fetch_msas(protein_seqs)

    print(f"Running: {', '.join(algos_to_run)}")
    print(f"Output: {run_dir}")

    handles = []
    for algo in algos_to_run:
        print(f"  Spawning {FOLDING_APPS[algo].name}...")
        handle = run_algorithm.spawn(algo, input_str, run_name, use_msa=use_msa, msa_result=msa_result, overwrite=overwrite)
        handles.append((algo, handle))

    for algo, handle in handles:
        app_def = FOLDING_APPS[algo]
        print(f"\n{'=' * 60}\n{app_def.name} finished\n{'=' * 60}")

        best, all_outputs = handle.get()

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


def _run_batch(
    input_dir: str, algos_to_run: list[str], out_dir: str, pattern: str,
    use_msa: bool, overwrite: bool, keep_all: bool,
):
    """Run folding on a directory of FASTA files using .map() for parallelism."""
    import glob

    # Boltz only runs with MSA; drop it under --no-use-msa for consistency with main().
    if not use_msa and "boltz2" in algos_to_run:
        print("Note: dropping Boltz-2 from this batch (Boltz only runs with MSA).")
        algos_to_run = [a for a in algos_to_run if a != "boltz2"]
        if not algos_to_run:
            print("No algorithms left to run.")
            return

    files = sorted(glob.glob(f"{input_dir}/{pattern}"))
    if not files:
        print(f"No files matching {input_dir}/{pattern}")
        return

    todo = []
    for f in files:
        name = Path(f).stem
        run_dir = Path(out_dir) / name
        # Only skip if EVERY requested algorithm already has output.
        if not overwrite and all(
            (run_dir / f"{name}.{algo}.cif").exists() for algo in algos_to_run
        ):
            print(f"SKIP {name}")
            continue
        todo.append(f)

    print(f"Running {len(todo)}/{len(files)} files with {', '.join(algos_to_run)}")
    if not todo:
        return

    # MSAs are fetched PER FILE (homomers preserved, queries scoped to each input).
    # Colabsearch caches by sequence-set server-side, so repeats are cheap.
    file_msa: dict[str, dict | None] = {f: None for f in todo}
    msa_backends = [a for a in algos_to_run if a in MSA_BACKENDS]
    if use_msa and msa_backends:
        print(f"[msa] Pre-fetching MSAs per file ({len(todo)} files)...")
        for f in todo:
            seqs = _extract_protein_sequences(open(f).read())
            file_msa[f] = _fetch_msas(seqs) if seqs else None

    # Spawn all (algo, file) jobs in parallel; each gets its own per-file MSA.
    handles: list[tuple[str, str, Any]] = []  # (algo, file_path, handle)
    for algo in algos_to_run:
        print(f"\n=== {FOLDING_APPS[algo].name}: spawning {len(todo)} inputs ===")
        for f in todo:
            handles.append((
                algo, f,
                run_algorithm.spawn(
                    algo, open(f).read(), Path(f).stem,
                    use_msa=use_msa, msa_result=file_msa[f], overwrite=overwrite,
                ),
            ))

    for algo, f, handle in handles:
        best, all_outputs = handle.get()
        name = Path(f).stem
        run_dir = Path(out_dir) / name
        run_dir.mkdir(parents=True, exist_ok=True)

        for key, content in best.items():
            ext = key.split(".")[-1]
            if key.startswith("structure"):
                out_path = run_dir / f"{name}.{algo}.{ext}"
            else:
                out_path = run_dir / f"{name}.{algo}.{key}"
            out_path.write_bytes(content)

        if keep_all:
            algo_dir = run_dir / algo
            algo_dir.mkdir(parents=True, exist_ok=True)
            for out_file, out_content in all_outputs:
                out_path = algo_dir / out_file
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(out_content or b"")

        score = json.loads(best.get("scores.json", b"{}")).get("ranking_score", 0)
        print(f"Done: {name}.{algo} (ranking={score:.3f})")

    print(f"\n{'=' * 60}\nComplete! {len(todo)} structures in: {out_dir}\n{'=' * 60}")


# =============================================================================
# Web Interface (HTML template in index.html)
# =============================================================================


def _superpose_structures(
    structures: dict[str, bytes], reference_key: str | None = None
) -> tuple[dict[str, bytes], dict[str, dict]]:
    """Superpose each non-reference structure onto the reference.

    Returns:
        (aligned_structures, rmsd_info) where rmsd_info maps method_key →
        {"rmsd": float, "n_atoms": int} for each non-reference key that was
        successfully aligned.
    """
    import gemmi

    rmsd_info: dict[str, dict] = {}
    if len(structures) <= 1:
        return structures, rmsd_info

    keys = list(structures.keys())
    ref_key = reference_key or keys[0]
    if ref_key not in structures:
        ref_key = keys[0]

    def _longest_peptide_chain(model):
        best, best_len = None, 0
        for chain in model:
            polymer = chain.get_polymer()
            if polymer and polymer.check_polymer_type() == gemmi.PolymerType.PeptideL:
                n = len(polymer)
                if n > best_len:
                    best, best_len = chain, n
        return best

    ref_doc = gemmi.cif.read_string(structures[ref_key].decode())
    ref_st = gemmi.make_structure_from_block(ref_doc[0])
    ref_model = ref_st[0]

    ref_chain = _longest_peptide_chain(ref_model)
    if not ref_chain:
        return structures, rmsd_info

    ref_polymer = ref_chain.get_polymer()
    ptype = ref_polymer.check_polymer_type()

    result = {ref_key: structures[ref_key]}

    for key, cif_bytes in structures.items():
        if key == ref_key:
            continue

        try:
            cif_str = cif_bytes.decode()
            doc = gemmi.cif.read_string(cif_str)
            st = gemmi.make_structure_from_block(doc[0])
            if len(st) == 0:
                print(f"[align] {key}: CIF has 0 models (block={doc[0].name}, size={len(cif_str)}, first_100={cif_str[:100]!r})")
                result[key] = cif_bytes
                continue
            model = st[0]

            target_chain = _longest_peptide_chain(model)
            if not target_chain:
                result[key] = cif_bytes
                continue

            target_polymer = target_chain.get_polymer()
            sup = gemmi.calculate_superposition(
                ref_polymer, target_polymer, ptype, gemmi.SupSelect.CaP, trim_cycles=3
            )
            print(f"[align] {key}: RMSD={sup.rmsd:.2f}, matched={sup.count} atoms")
            rmsd_info[key] = {"rmsd": float(sup.rmsd), "n_atoms": int(sup.count)}

            for m in st:
                for chain in m:
                    for residue in chain:
                        for atom in residue:
                            new = sup.transform.apply(atom.pos)
                            atom.pos = gemmi.Position(new[0], new[1], new[2])

            st.update_mmcif_block(doc[0])
            result[key] = doc.as_string().encode()
        except Exception as e:
            import traceback
            print(f"Failed to superpose {key}: {type(e).__name__}: {e}")
            traceback.print_exc()
            result[key] = cif_bytes

    return result, rmsd_info


def _fetch_reference_pdb_bytes(pdb_id: str) -> bytes:
    """Fetch a structure from RCSB by 4-char PDB ID (returns CIF bytes)."""
    import urllib.request

    pdb_id = pdb_id.strip().lower()
    if not re.fullmatch(r"[a-z0-9]{4}", pdb_id):
        raise ValueError(f"Invalid PDB ID: {pdb_id!r}")
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def _chain_one_letter_seq(chain) -> str:
    """One-letter protein sequence from a gemmi chain's polymer."""
    import gemmi

    seq = []
    for res in chain.get_polymer():
        info = gemmi.find_tabulated_residue(res.name)
        seq.append(info.one_letter_code.upper() if info and info.one_letter_code else "X")
    return "".join(seq)


def _seq_match_score(a: str, b: str) -> float:
    """Score similarity: sum of >=5-char matching blocks / len(shorter).

    Filters out 1-char matches so short peptides don't spuriously match long
    unrelated chains via scattered single-letter coincidences.
    """
    from difflib import SequenceMatcher

    if not a or not b:
        return 0.0
    min_block = min(5, min(len(a), len(b)))
    matched = sum(blk.size for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks() if blk.size >= min_block)
    return matched / min(len(a), len(b))


def _load_reference_structure(
    pdb_id: str | None, uploaded_bytes: bytes | None, fasta_str: str
) -> tuple[bytes, list[str]]:
    """Return (filtered_cif_bytes, matched_chain_descriptions).

    Keeps only chains matching a FASTA seq at >=85% identity — drops
    antibody/light/etc. chains that the user did not include in the FASTA.
    """
    import gemmi

    if uploaded_bytes:
        raw = uploaded_bytes
        source = "uploaded"
    elif pdb_id:
        raw = _fetch_reference_pdb_bytes(pdb_id)
        source = pdb_id.upper()
    else:
        raise ValueError("No reference PDB provided")

    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("data_"):
        doc = gemmi.cif.read_string(text)
        st = gemmi.make_structure_from_block(doc[0])
    else:
        st = gemmi.read_pdb_string(text)

    if len(st) == 0:
        raise ValueError(f"Reference {source}: no models")
    model = st[0]

    fasta_seqs = _extract_protein_sequences(fasta_str)
    if not fasta_seqs:
        raise ValueError("No protein sequences in FASTA to match against reference")

    chain_seqs: dict[str, str] = {}
    for chain in model:
        polymer = chain.get_polymer()
        if polymer and polymer.check_polymer_type() == gemmi.PolymerType.PeptideL:
            chain_seqs[chain.name] = _chain_one_letter_seq(chain)

    chains_to_keep: set[str] = set()
    matched_desc: list[str] = []
    print(f"[ref] {source}: chains={[(n, len(s)) for n, s in chain_seqs.items()]}")
    for cname, cseq in chain_seqs.items():
        best_score = max((_seq_match_score(fseq, cseq) for fseq in fasta_seqs), default=0.0)
        print(f"[ref]   {cname} ({len(cseq)} aa): best match score = {best_score:.2f}")
        if best_score >= 0.85:
            chains_to_keep.add(cname)
            matched_desc.append(f"{cname} ({best_score:.0%})")

    if not chains_to_keep:
        raise ValueError(f"Reference {source}: no chains matched any FASTA sequence (>=85%)")

    for cname in [c.name for c in model if c.name not in chains_to_keep]:
        model.remove_chain(cname)

    st.setup_entities()
    doc = st.make_mmcif_document()
    return doc.as_string().encode(), matched_desc


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
    fasta_str: str, method: str, use_msa: bool = True, job_id: str | None = None,
    msa_result: dict | None = None,
) -> tuple[str, dict | None, str | None]:
    """Run a single folding method for web interface."""
    try:
        converted = convert_for_app(fasta_str, method)
        params = _build_method_params(method, converted, use_msa, msa_result=msa_result, original_fasta=fasta_str)

        # Check cache first - avoids spawning GPU container if cached
        cached = _check_cache_inline(method, params)
        if cached is not None:
            print(f"[{method}] Returning cached result ({len(cached)} files)")
            outputs = [(Path(f), c) for f, c in cached]
        elif method == "boltz2":
            outputs = boltz2_predict.remote(params=params, job_id=job_id)
        elif method == "chai1":
            outputs = chai1_predict.remote(params=params, job_id=job_id)
        elif method == "protenix":
            outputs = protenix_predict.remote(params=params, job_id=job_id)
        elif method == "alphafold2":
            outputs = alphafold_predict.remote(params=params, job_id=job_id)
        elif method == "openfold3":
            outputs = openfold3_predict.remote(params=params, job_id=job_id)
        elif method == "esmfold2":
            outputs = esmfold2_predict.remote(params=params, job_id=job_id)
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
    raw_scores: dict | None = None

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
                raw_scores = json.loads(content)

    elif method == "chai1":
        best_idx = _find_best_model_index(method, outputs)
        cif_key = f"pred.model_idx_{best_idx}.cif"
        if cif_key in outputs_dict:
            files["structure"] = (outputs_dict[cif_key], "cif")
        npz_key = f"scores.model_idx_{best_idx}.npz"
        if npz_key in outputs_dict:
            raw_scores = _parse_chai1_npz(outputs_dict[npz_key])
            files["scores"] = json.dumps(raw_scores).encode()

    elif method == "protenix":
        best_idx = _find_best_model_index(method, outputs)
        for path, content in outputs:
            path_str = str(path)
            if f"_sample_{best_idx}.cif" in path_str:
                files["structure"] = (content, "cif")
            elif f"summary_confidence_sample_{best_idx}.json" in path_str:
                files["scores"] = content
                raw_scores = json.loads(content)

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
                raw_scores = json.loads(content)
            elif "_scores_rank_001" in path_str and path_str.endswith(".json"):
                if raw_scores is None:
                    files["scores"] = content
                    raw_scores = json.loads(content)

    elif method == "openfold3":
        best_idx = _find_best_model_index(method, outputs)
        for path, content in outputs:
            path_str = str(path)
            if f"_sample_{best_idx}_model.cif" in path_str:
                files["structure"] = (content, "cif")
            elif f"_sample_{best_idx}_confidences_aggregated.json" in path_str:
                files["scores"] = content
                raw_scores = json.loads(content)

    elif method == "esmfold2":
        best_idx = _find_best_model_index(method, outputs)
        for path, content in outputs:
            path_str = str(path)
            if path_str.endswith(f"_sample_{best_idx}.cif"):
                files["structure"] = (content, "cif")
            elif path_str.endswith(f"_sample_{best_idx}_scores.json"):
                files["scores"] = content
                raw_scores = json.loads(content)

    if raw_scores is not None:
        files["unified"] = _normalize_scores(method, raw_scores)

    if "structure" not in files:
        return None

    files["all_files"] = [(str(f), c) for f, c in outputs if c]
    return files


@app.function(image=web_image, timeout=60 * 60, volumes={"/cache": CACHE_VOLUME})
def run_folding_job(
    job_id: str,
    fasta: str,
    methods: list[str],
    use_msa: bool,
    reference_pdb_id: str | None = None,
    reference_pdb_bytes: bytes | None = None,
):
    """Background orchestrator for folding jobs (avoids 10-min SSE timeout)."""
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

    # Defense in depth: UI already prevents this, but if a client submits
    # boltz2 with use_msa=false, drop it (Boltz only runs with MSA).
    if not use_msa and "boltz2" in methods:
        methods = [m for m in methods if m != "boltz2"]
        logs0 = [{"msg": "Boltz-2 skipped (requires MSA; --no-use-msa was set)", "cls": "warning"}]
    else:
        logs0 = []
    total = len(methods)
    logs = logs0 + [{"msg": "Checking cache...", "cls": "info"}]
    update_job(5, f"Starting {total} methods...", logs)

    # Check result cache for each method FIRST so fully-cached jobs skip
    # the (slow) ColabSearch MSA fetch entirely.
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

    # Only fetch MSAs if at least one method needs to actually run.
    msa_result = None
    needs_colabsearch = use_msa and any(m in methods_to_run for m in MSA_BACKENDS)
    if needs_colabsearch:
        protein_seqs = _extract_protein_sequences(fasta)
        if protein_seqs:
            logs.append({"msg": f"Fetching MSAs for {len(protein_seqs)} sequences...", "cls": "info"})
            update_job(5, "Fetching MSAs...", logs)
            msa_result = _fetch_msas(protein_seqs)
            n_unpaired = len(msa_result.get("unpaired", {}))
            has_paired = bool(msa_result.get("paired_dir"))
            paired_str = ", paired" if has_paired else ""
            logs.append({"msg": f"MSAs ready ({n_unpaired} unpaired{paired_str})", "cls": "success"})
            update_job(8, f"Starting {total} methods...", logs)

    for method in methods_to_run:
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
        handle = fold_structure_web.spawn(fasta, method, use_msa, job_id, msa_result=msa_result)
        handles.append((method, handle))

    all_results = {}
    structures_to_superpose = {}
    # Anchor for superposition. Starts as the reference if one was loaded;
    # demoted to the first successfully-aligned method if reference alignment
    # fails (e.g. reference has alien chain types / very different topology).
    anchor_key: str | None = None
    # RMSD to the *reference* structure (not to another predicted method).
    # Only populated when anchor_key == "reference" at align-time.
    rmsd_to_ref: dict[str, dict] = {}

    # If a reference structure was provided, load it first so it becomes the
    # alignment anchor (existing code uses the first inserted key as anchor).
    if reference_pdb_id or reference_pdb_bytes:
        try:
            ref_cif, matched = _load_reference_structure(reference_pdb_id, reference_pdb_bytes, fasta)
            ref_name = reference_pdb_id.upper() if reference_pdb_id else "reference"
            structures_to_superpose["reference"] = ref_cif
            anchor_key = "reference"
            logs.append({"msg": f"Reference {ref_name}: kept {', '.join(matched)}", "cls": "success"})
            # Convert CIF → PDB for the viewer (matches the other backends' path
            # and gives molstar an unambiguous polymer to render as cartoon).
            try:
                display_bytes = _cif_to_pdb(ref_cif)
                display_fmt = "pdb"
            except Exception as e:
                logs.append({"msg": f"Reference CIF→PDB failed, sending CIF: {e}", "cls": "warning"})
                display_bytes = ref_cif
                display_fmt = "cif"
            data_payload = _build_result_payload(
                display_bytes, display_fmt,
                {"all_files": [(f"{ref_name}.cif", ref_cif)]},
                original_cif_bytes=ref_cif,
            )
            update_job(
                10, "Reference loaded", logs,
                [{"method": f"Reference ({ref_name})", "method_key": "reference",
                  "data": data_payload, "format": display_fmt}],
            )
        except Exception as e:
            logs.append({"msg": f"Reference load failed: {e}", "cls": "error"})
            update_job(10, "Reference failed", logs)

    # Process cached results immediately - with alignment and incremental display
    for method, outputs in cached_results.items():
        logs.append({"msg": f"{FOLDING_APPS[method].name}: processing cached files ({len(outputs)})", "cls": "dim"})
        update_job(int(10 + (len(all_results) / total) * 70), f"Processing {FOLDING_APPS[method].name}...", logs)
        files = _process_outputs_to_files(method, outputs)
        if not files or "structure" not in files:
            paths_preview = ", ".join(str(p) for p, _ in outputs[:8])
            logs.append({
                "msg": f"{FOLDING_APPS[method].name}: cached result missing structure (files: {paths_preview}...)",
                "cls": "error",
                "method_complete": method,
            })
            all_results[method] = (None, "cached result missing structure")
            update_job(int(10 + (len(all_results) / total) * 70), f"{len(all_results)}/{total}", logs)
            continue
        if files and "structure" in files:
            all_results[method] = (files, None)
            structure_bytes, fmt = files["structure"]

            if fmt == "cif":
                if anchor_key is None:
                    anchor_key = method
                    structures_to_superpose[method] = structure_bytes
                    logs.append({
                        "msg": f"{FOLDING_APPS[method].name}: Using as reference for alignment",
                        "cls": "info",
                    })
                else:
                    try:
                        ref_structure = structures_to_superpose[anchor_key]
                        to_align = {anchor_key: ref_structure, method: structure_bytes}
                        aligned, rmsd_info = _superpose_structures(to_align, anchor_key)
                        structure_bytes = aligned[method]
                        structures_to_superpose[method] = structure_bytes
                        if anchor_key == "reference" and method in rmsd_info:
                            rmsd_to_ref[method] = rmsd_info[method]
                        anchor_name = FOLDING_APPS[anchor_key].name if anchor_key in FOLDING_APPS else anchor_key
                        logs.append({
                            "msg": f"{FOLDING_APPS[method].name}: Aligned to {anchor_name}",
                            "cls": "info",
                        })
                    except Exception as e:
                        logs.append({
                            "msg": f"{FOLDING_APPS[method].name}: Alignment failed: {e}",
                            "cls": "warning",
                        })
                        structures_to_superpose[method] = structure_bytes
                        # If we couldn't align to the reference, demote it as anchor
                        # and let this method anchor subsequent alignments.
                        if anchor_key == "reference":
                            anchor_key = method
                            logs.append({
                                "msg": f"Reference not usable as anchor — falling back to {FOLDING_APPS[method].name}",
                                "cls": "info",
                            })

            # Build and send result immediately
            # Send predicted CIF straight to molstar (mmCIF is its native format).
            # Avoids the maxit subprocess hop and preserves entity-level metadata.
            print(f"[result] {method}: structure={len(structure_bytes)} bytes, fmt={fmt}")
            data_payload = _build_result_payload(
                structure_bytes, fmt, files, original_cif_bytes=None,
                rmsd_ref=rmsd_to_ref.get(method),
            )

            result_to_send = [
                {
                    "method": FOLDING_APPS[method].name,
                    "method_key": method,
                    "data": data_payload,
                    "format": fmt,
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
                        if anchor_key is None:
                            anchor_key = method
                            structures_to_superpose[method] = structure_bytes
                            logs.append({
                                "msg": f"{FOLDING_APPS[method].name}: Using as reference for alignment",
                                "cls": "info",
                            })
                        else:
                            try:
                                ref_structure = structures_to_superpose[anchor_key]
                                to_align = {anchor_key: ref_structure, method: structure_bytes}
                                aligned, rmsd_info = _superpose_structures(to_align, anchor_key)
                                structure_bytes = aligned[method]
                                structures_to_superpose[method] = structure_bytes
                                if anchor_key == "reference" and method in rmsd_info:
                                    rmsd_to_ref[method] = rmsd_info[method]
                                anchor_name = FOLDING_APPS[anchor_key].name if anchor_key in FOLDING_APPS else anchor_key
                                logs.append({
                                    "msg": f"{FOLDING_APPS[method].name}: Aligned to {anchor_name}",
                                    "cls": "info",
                                })
                            except Exception as e:
                                logs.append({
                                    "msg": f"{FOLDING_APPS[method].name}: Alignment failed: {e}",
                                    "cls": "warning",
                                })
                                structures_to_superpose[method] = structure_bytes
                                if anchor_key == "reference":
                                    anchor_key = method
                                    logs.append({
                                        "msg": f"Reference not usable as anchor — falling back to {FOLDING_APPS[method].name}",
                                        "cls": "info",
                                    })

                    # Send native CIF straight to molstar (mmCIF is its primary format).
                    data_payload = _build_result_payload(
                        structure_bytes, fmt, files, original_cif_bytes=None,
                        rmsd_ref=rmsd_to_ref.get(method),
                    )

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
                ("alphafold2", "alphafold2_logs", "AlphaFold2"),
                ("openfold3", "openfold3_logs", "OpenFold3"),
                ("esmfold2", "esmfold2_logs", "ESMFold2"),
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

        # Send native CIF straight to molstar.
        data_payload = _build_result_payload(
            structure_bytes, fmt, files, original_cif_bytes=None,
            rmsd_ref=rmsd_to_ref.get(method),
        )

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

    @flask_app.route("/overrides.json", methods=["GET"])
    def overrides():
        p = Path("/app/overrides.json")
        if p.exists():
            return flask_app.response_class(p.read_text(), mimetype="application/json")
        abort(404)

    @flask_app.route("/fold", methods=["POST"])
    def fold():
        """Spawn a folding job and return job_id (polling pattern to avoid 10-min SSE timeout)."""
        import uuid

        from flask import jsonify

        fasta = request.form.get("fasta", "").strip()
        methods = request.form.getlist("method")
        use_msa = request.form.get("use_msa", "true").lower() == "true"
        reference_pdb_id = (request.form.get("reference_pdb_id") or "").strip() or None
        reference_file = request.files.get("reference_file")
        reference_pdb_bytes = reference_file.read() if reference_file and reference_file.filename else None

        if not fasta:
            return jsonify({"error": "No sequence"}), 400

        if not methods:
            methods = ["boltz2"]

        job_id = str(uuid.uuid4())[:12]
        job_store[job_id] = {"progress": 0, "status": "Starting...", "done": False}

        # Spawn the orchestrator in the background
        run_folding_job.spawn(
            job_id, fasta, methods, use_msa,
            reference_pdb_id=reference_pdb_id,
            reference_pdb_bytes=reference_pdb_bytes,
        )

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

        # Boltz only runs with MSA; drop it under --no-use-msa (mirrors main /fold).
        if not use_msa and "boltz2" in methods:
            methods = [m for m in methods if m != "boltz2"]

        def generate():
            import time
            from pathlib import Path

            total = len(methods)
            yield f"data: {json.dumps({'progress': 5, 'status': f'Starting {total} methods...', 'log': f'Checking cache...', 'log_class': 'info'})}\n\n"

            # Check cache FIRST so fully-cached jobs skip the slow MSA fetch.
            cached_results = {}
            methods_to_run = []
            for method in methods:
                cached = check_cache_in_server(method, fasta, use_msa)
                if cached is not None:
                    cached_results[method] = [(Path(f), c) for f, c in cached]
                    yield f"data: {json.dumps({'log': f'{FOLDING_APPS[method].name}: cached', 'log_class': 'success'})}\n\n"
                else:
                    methods_to_run.append(method)

            # Only fetch MSAs for backends that will actually run.
            msa_result = None
            needs_colabsearch = use_msa and any(m in methods_to_run for m in MSA_BACKENDS)
            if needs_colabsearch:
                protein_seqs = _extract_protein_sequences(fasta)
                if protein_seqs:
                    yield f"data: {json.dumps({'log': f'Fetching MSAs for {len(protein_seqs)} sequences...', 'log_class': 'info'})}\n\n"
                    msa_result = _fetch_msas(protein_seqs)
                    n_unpaired = len(msa_result.get("unpaired", {}))
                    has_paired = bool(msa_result.get("paired_dir"))
                    paired_str = ", paired" if has_paired else ""
                    yield f"data: {json.dumps({'log': f'MSAs ready ({n_unpaired} unpaired{paired_str})', 'log_class': 'success'})}\n\n"

            # Spawn only methods that need computation
            handles = []
            for method in methods_to_run:
                handle = fold_structure_web.spawn(fasta, method, use_msa, msa_result=msa_result)
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
