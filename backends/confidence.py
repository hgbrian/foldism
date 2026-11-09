# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "gemmi"]
# ///
"""Interface confidence for foldism: ipTM, ipSAE, ipDAE and actifpTM for every chain pair.

Sections: scores, native-output adapters, AlphaFold2 capture, percentile ranks.
Backends copy this single file into their Modal images, so it imports only the
standard library and numpy (gemmi is imported on use). The frozen percentile
reference table is the long single line at the end.

Credits and licenses:
- ipSAE: Dunbrack lab IPSAE (MIT), commit 6174cf9e71cb1bd660cc805856a18c4871a6dec3.
- ipDAE: reimplements the contact-masked ipDAE definition of BindCraft2
  (PacesaLab/BindCraft2, commit b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06).
  Copyright (c) 2026 Martin Pacesa, University of Zurich, and the BindCraft2
  contributors. BindCraft2 is under a source-available license that restricts
  offering it to third parties as a hosted service; read it before hosting.
- actifpTM: Varga et al., Bioinformatics 41(3), btaf107.
- Percentile reference: derived from public ProteinBase non-binders (Adaptyv Bio,
  https://proteinbase.com), shared under ODC-BY 1.0:
  https://opendatacommons.org/licenses/by/1-0/
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import runpy
import shutil
import sys
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from math import isfinite
from pathlib import Path
from uuid import uuid4

import numpy as np

SCORE_VERSION = "protein-pairs-v1-ipsae-20260103"


def as_array(value):
    """Copy tensors to CPU without rounding."""
    if hasattr(value, "detach"):
        value = value.detach()
        value = (value.float() if value.is_floating_point() else value).cpu().numpy()
    return np.asarray(value)


def ipsae_direction(block: np.ndarray) -> float:
    """Reference d0res score with strict PAE < 10 angstroms."""
    mask = block < 10
    counts = mask.sum(axis=1)
    scale = np.maximum(1.0, 1.24 * (np.maximum(counts, 26) - 15) ** (1 / 3) - 1.8)
    kernel = 1 / (1 + (block / scale[:, None]) ** 2)
    return float(((kernel * mask).sum(axis=1) / np.maximum(counts, 1)).max())


def anchored_score(block: np.ndarray, contacts: np.ndarray) -> float:
    """Pinned BindCraft2 contact-normalized score."""
    counts = contacts.sum(axis=1)
    scale = np.maximum(1.0, 1.24 * (np.maximum(counts, 19) - 15) ** (1 / 3) - 1.8)
    kernel = 1 / (1 + (block / scale[:, None]) ** 2)
    return float(((kernel * contacts).sum(axis=1) / np.maximum(counts, 1)).max())


def softmax(values: np.ndarray) -> np.ndarray:
    """Stable normalization over the final axis."""
    exponent = np.exp(values - values.max(axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def actifptm_from_terms(terms, contacts, chains, pair=None) -> float:
    """Contact-weighted score, masking to a pair while retaining full-length d0."""
    weights = np.asarray(contacts, dtype=np.float64) * (
        chains[:, None] != chains[None, :]
    )
    if pair is not None:
        selected = np.isin(chains, pair)
        weights *= selected[:, None] & selected[None, :]
    return float(
        (terms * weights / (1e-8 + weights.sum(axis=-1, keepdims=True)))
        .sum(axis=-1)
        .max()
    )


def actifptm_pairs(logits, centers, contacts, chains) -> dict[str, float]:
    """Author's pair masking with d0 from the full protein-complex length."""
    logits, centers, contacts, chains = (
        as_array(value) for value in (logits, centers, contacts, chains)
    )
    if logits.shape != (*contacts.shape, len(centers)) or contacts.shape != (
        len(chains),
        len(chains),
    ):
        raise ValueError("Confidence distributions and chain identities do not align")
    if not all(np.isfinite(value).all() for value in (logits, centers, contacts)):
        raise ValueError("Nonfinite confidence distributions")
    if (contacts < -1e-6).any() or (contacts > 1 + 1e-6).any() or (centers < 0).any():
        raise ValueError("Invalid contact probabilities or PAE bins")
    contacts = np.clip(contacts, 0, 1)
    scale = 1.24 * (max(len(chains), 19) - 15) ** (1 / 3) - 1.8
    kernel = 1 / (1 + (centers.astype(np.float64) / scale) ** 2)
    terms = np.empty(contacts.shape, dtype=np.float64)
    for start in range(0, len(chains), 32):
        terms[start : start + 32] = (
            softmax(logits[start : start + 32].astype(np.float64)) * kernel
        ).sum(axis=-1)
    return {
        f"{first}:{second}": actifptm_from_terms(
            terms, contacts, chains, (first, second)
        )
        for first, second in combinations(dict.fromkeys(chains.tolist()), 2)
    }


def native_pair_value(
    native, first: int, second: int, count: int, chain_keys=None
) -> float | None:
    """Read native ipTM using explicit identity, never dict insertion order."""
    if native is None:
        return None
    if isinstance(native, dict):
        keys = list(range(count)) if chain_keys is None else chain_keys
        normalized = {str(key): value for key, value in native.items()}
        if len(keys) != count or any(str(key) not in normalized for key in keys):
            return None
        forward = {
            str(key): value for key, value in normalized[str(keys[first])].items()
        }
        reverse = {
            str(key): value for key, value in normalized[str(keys[second])].items()
        }
        values = [forward.get(str(keys[second])), reverse.get(str(keys[first]))]
        values = [float(value) for value in values if value is not None]
        return max(values) if values else None
    matrix = as_array(native)
    if matrix.shape == (count, count):
        return float(max(matrix[first, second], matrix[second, first]))
    if count == 2 and matrix.size == 1:
        return float(matrix.reshape(-1)[0])
    return None


def calculate(
    pae,
    ca,
    chain_ids,
    native_iptm=None,
    actifptm=None,
    calculate_actifptm=False,
    native_chain_ids=None,
) -> dict:
    """Return all unordered pairs in structure order, with explicit score status.

    ipDAE uses the first chain's forward PAE block and its transpose, as in BC2.
    It does not use the reverse native PAE block. Chain order therefore matters.
    """
    pae, ca, chain_ids = as_array(pae), as_array(ca), as_array(chain_ids)
    count = len(chain_ids)
    if pae.shape != (count, count) or ca.shape != (count, 3):
        raise ValueError("PAE and C-alpha coordinates do not match residue mapping")
    if not np.isfinite(pae).all() or (pae < 0).any() or not np.isfinite(ca).all():
        raise ValueError("Invalid PAE or C-alpha coordinates")
    chains = list(dict.fromkeys(chain_ids.tolist()))
    native_ids = (
        native_chain_ids if native_chain_ids is not None else list(range(len(chains)))
    )
    if len(native_ids) != len(chains):
        raise ValueError("Native chain mapping has the wrong size")
    metrics = ("iptm", "ipsae", "ipdae", "actifptm")
    result: dict = {
        "version": SCORE_VERSION,
        "pairs": [],
        **{metric: [] for metric in metrics},
        "status": {metric: [] for metric in metrics},
        "actifptm_requested": calculate_actifptm,
        "ipdae_orientation": "first chain rows; forward block and transpose",
        "iptm_definition": "native chain-pair ipTM; maximum of available directions",
        "definitions": {
            "ipsae_pae_cutoff_angstrom": 10,
            "ipdae_ca_contact_angstrom": 8,
            "actifptm_contact_angstrom": 8,
            "actifptm_d0_length": "full protein complex",
        },
    }
    for first, second in combinations(range(len(chains)), 2):
        left = np.flatnonzero(chain_ids == chains[first])
        right = np.flatnonzero(chain_ids == chains[second])
        forward, reverse = pae[np.ix_(left, right)], pae[np.ix_(right, left)]
        contacts = np.linalg.norm(ca[left, None] - ca[None, right], axis=-1) <= 8
        ipdae = (
            max(
                anchored_score(forward, contacts), anchored_score(forward.T, contacts.T)
            )
            if contacts.any()
            else None
        )
        active = (
            (actifptm or {}).get(f"{native_ids[first]}:{native_ids[second]}")
            if calculate_actifptm
            else None
        )
        values = {
            "iptm": native_pair_value(
                native_iptm, first, second, len(chains), native_ids
            ),
            "ipsae": max(ipsae_direction(forward), ipsae_direction(reverse)),
            "ipdae": ipdae,
            "actifptm": active,
        }
        result["pairs"].append([chains[first], chains[second]])
        for metric, value in values.items():
            if metric == "actifptm" and calculate_actifptm and value is None:
                raise ValueError("actifpTM was requested, but native confidence capture is missing")
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Invalid {metric} value: {value}")
            result[metric].append(value)
            status = "complete" if value is not None else "missing_native_input"
            if metric == "ipdae" and value is None:
                status = "no_ca_contacts_within_8_angstrom"
            if metric == "actifptm" and not calculate_actifptm:
                status = "not_requested"
            result["status"][metric].append(status)
    return result


def protein_structure(path: Path):
    """Read protein C-alpha atoms in exact structure order."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError("Expected one structure model")
    ca, chains, sequences = [], [], []
    for chain in structure[0]:
        sequence = []
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if not info.is_amino_acid():
                continue
            atom = residue.find_atom("CA", "*")
            if not atom:
                raise ValueError("Protein residue is missing C-alpha")
            ca.append([atom.pos.x, atom.pos.y, atom.pos.z])
            chains.append(chain.name)
            sequence.append(info.one_letter_code.upper())
        if sequence:
            sequences.append("".join(sequence))
    return np.asarray(ca), np.asarray(chains), sequences


def finalize_native_outputs(
    root: Path, expected_sequences: list[str], calculate_actifptm: bool
) -> None:
    """Add small scores to native summaries before filtering downloadable files."""
    summaries = list(root.rglob("*_summary_confidence_sample_*.json"))
    if not summaries:
        raise ValueError("No native confidence summaries were produced")
    for path in summaries:
        summary = json.loads(path.read_text())
        cif = path.with_name(path.name.replace("_summary_confidence", "")).with_suffix(
            ".cif"
        )
        full = json.loads(
            path.with_name(
                path.name.replace("_summary_confidence", "_full_data")
            ).read_text()
        )
        ca, chains, sequences = protein_structure(cif)
        if sequences != expected_sequences:
            raise ValueError("Structure protein sequences differ from ordered inputs")
        pae, asym = (
            np.asarray(full["token_pair_pae"]),
            np.asarray(full["token_asym_id"]),
        )
        native_ids = list(dict.fromkeys(asym.tolist()))
        if asym.tolist() != sorted(asym.tolist(), key=native_ids.index):
            raise ValueError("Native token chains are interleaved")
        protein_ids = native_ids[: len(sequences)]
        for identity, sequence in zip(protein_ids, sequences, strict=True):
            if int((asym == identity).sum()) != len(sequence):
                raise ValueError("Native token chain differs from input protein length")
        indices = np.flatnonzero(np.isin(asym, protein_ids))
        native = summary.get("chain_pair_iptm")
        if native is not None:
            native = np.asarray(native)[: len(sequences), : len(sequences)]
        summary["interface_confidence"] = calculate(
            pae[np.ix_(indices, indices)],
            ca,
            chains,
            native,
            summary.pop("foldism_actifptm", None),
            calculate_actifptm,
            protein_ids,
        )
        path.write_text(json.dumps(summary))


def downloadable(path: Path) -> bool:
    """Exclude distributions from ordinary result downloads."""
    return (path.name.endswith("_confidence_arrays.npz") or path.suffix in {".cif", ".pdb", ".json"}) and not any(
        part in path.name
        for part in ("_full_data_", "distogram", "pae_logits", "input.json")
    )


def cache_supports(directory: Path, calculate_actifptm: bool, download_confidence_data: bool = False) -> bool:
    """Require the current score schema and requested confidence capability."""
    summaries = [
        path
        for path in directory.rglob("*.json")
        if "_scores" in path.name or "_summary_confidence" in path.name
    ]
    if download_confidence_data and len(list(directory.rglob("*_confidence_arrays.npz"))) < len(summaries):
        return False
    if not summaries:
        return False
    for path in summaries:
        try:
            scores = json.loads(path.read_text()).get("interface_confidence", {})
        except (ValueError, OSError):
            return False
        if (
            scores.get("version") != SCORE_VERSION
            or scores.get("actifptm_requested") != calculate_actifptm
        ):
            return False
        pairs = scores.get("pairs")
        if not isinstance(pairs, list):
            return False
        statuses = scores.get("status", {})
        for metric in ("iptm", "ipsae", "ipdae", "actifptm"):
            values = scores.get(metric)
            metric_statuses = statuses.get(metric)
            if (
                not isinstance(values, list)
                or not isinstance(metric_statuses, list)
                or len(values) != len(pairs)
                or len(metric_statuses) != len(pairs)
            ):
                return False
        if calculate_actifptm and any(
            status != "complete"
            for status in scores.get("status", {}).get("actifptm", [])
        ):
            return False
    return True


# =============================================================================
# Native-output adapters (capture confidence inside each predictor)
# =============================================================================

def install_native(module, model: str) -> None:
    """Wrap the native confidence function before the inference runner imports it."""
    original = module.compute_full_data_and_summary
    if getattr(original, "foldism_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        scores = []
        array_names = []
        if os.environ.get("FOLDISM_ACTIFPTM") == "1":
            config = values["configs"]
            bins = config.confidence.pae if model == "opendde" else config.loss.pae
            centers = as_array(module.get_bin_centers(**module.get_bin_params(bins)))
            chains = as_array(values["token_asym_id"]).astype(np.int64)
            atom_to_token = as_array(values["atom_to_token_idx"]).astype(np.int64)
            polymer = as_array(values["atom_is_polymer"]) > 0
            indices = np.unique(atom_to_token[polymer])
            logits, contacts = values["pae_logits"], values["contact_probs"]
            for index in range(logits.shape[0]):
                sample_contacts = contacts if contacts.ndim == 2 else contacts[index]
                if os.environ.get("FOLDISM_CONFIDENCE_DIR"):
                    destination = Path(os.environ["FOLDISM_CONFIDENCE_DIR"])
                    destination.mkdir(parents=True, exist_ok=True)
                    array_name = f"{uuid4().hex}_sample_{index}_confidence_arrays.npz"
                    array_names.append(array_name)
                    np.savez_compressed(
                        destination / array_name,
                        pae_logits=as_array(logits[index])[indices][:, indices],
                        pae_bin_centers=centers,
                        contact_probs=as_array(sample_contacts)[indices][:, indices],
                        chain_ids=chains[indices],
                    )
                scores.append(
                    actifptm_pairs(
                        as_array(logits[index])[indices][:, indices],
                        centers,
                        as_array(sample_contacts)[indices][:, indices],
                        chains[indices],
                    )
                )
        summaries, full_data = original(*args, **kwargs)
        if scores:
            if len(scores) != len(summaries):
                raise ValueError("Native confidence sample count changed")
            for summary, active in zip(summaries, scores, strict=True):
                summary["foldism_actifptm"] = active
            if array_names:
                for summary, name in zip(summaries, array_names, strict=True):
                    summary["confidence_arrays_file"] = name
        return summaries, full_data

    vars(wrapped)["foldism_wrapped"] = True
    module.compute_full_data_and_summary = wrapped
    # Update earlier from-import aliases in predictor modules; later imports see wrapped.
    prefixes = ("mol.", "opendde.", "runner.")
    for name, loaded in list(sys.modules.items()):
        if name.startswith(prefixes) and loaded is not None:
            namespace = vars(loaded)
            for key, value in list(namespace.items()):
                if value is original:
                    namespace[key] = wrapped


def esm_fold(model, prediction_input, **options):
    """Capture exact ESMFold2 token order and optional confidence distributions."""
    # ESM SDK is installed only inside its Modal image.
    ESMFold2InputBuilder = importlib.import_module(
        "esm.models.esmfold2"
    ).ESMFold2InputBuilder

    captured = []
    enabled = options.pop("calculate_actifptm", True)
    retain = options.pop("download_confidence_data", True)

    def capture(module, args, kwargs, output):
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        bound.apply_defaults()
        repeats = bound.arguments["num_diffusion_samples"]
        mask = np.repeat(
            as_array(bound.arguments["token_attention_mask"]), repeats, axis=0
        )
        asym = np.repeat(as_array(bound.arguments["asym_id"]), repeats, axis=0)
        pae = as_array(output["pae"])
        logits = as_array(output["pae_logits"]) if enabled else None
        for index in range(len(pae)):
            captured.append(
                {
                    "pae": pae[index],
                    "chains": asym[index],
                    "mask": mask[index],
                    "logits": logits[index] if logits is not None else None,
                }
            )

    handle = model.confidence_head.register_forward_hook(capture, with_kwargs=True)
    try:
        result = ESMFold2InputBuilder().fold(model, prediction_input, **options)
    finally:
        handle.remove()
    samples = result if isinstance(result, list) else [result]
    if len(samples) != len(captured):
        raise ValueError("ESMFold2 confidence and structure sample counts differ")
    for sample, data in zip(samples, captured, strict=True):
        if not np.array_equal(data["pae"], as_array(sample.pae).astype(np.float32)):
            raise ValueError("ESMFold2 captured PAE differs from returned sample")
        if enabled:
            logits = data.pop("logits")
            bins = (np.arange(logits.shape[-1]) + 0.5) * (32 / logits.shape[-1])
            distance_centers = 2 + (np.arange(64) + 0.5) * (20 / 64)
            distogram = as_array(sample.distogram)
            contacts = np.empty(distogram.shape[:2], dtype=np.float64)
            for start in range(0, len(contacts), 32):
                contacts[start : start + 32] = softmax(
                    distogram[start : start + 32].astype(np.float64)
                )[..., distance_centers < 8].sum(-1)
            indices = np.flatnonzero(data["mask"])
            if retain:
                data["confidence_arrays"] = {
                    "pae_logits": logits[indices][:, indices],
                    "pae_bin_centers": bins,
                    "contact_probs": contacts[indices][:, indices],
                    "chain_ids": data["chains"][indices],
                }
            data["active"] = actifptm_pairs(
                logits[indices][:, indices],
                bins,
                contacts[indices][:, indices],
                data["chains"][indices],
            )
    return samples, captured


def main() -> None:
    """Run the installed predictor CLI with confidence capture installed first."""
    model = sys.argv.pop(1)
    module_name = (
        "mol.model.sample_confidence"
        if model == "mol1"
        else f"{model}.model.sample_confidence"
    )
    install_native(importlib.import_module(module_name), model)
    executable = shutil.which("mol" if model == "mol1" else model)
    if executable is None:
        raise FileNotFoundError(f"Missing {model} executable")
    sys.argv[0] = executable
    runpy.run_path(executable, run_name="__main__")


# =============================================================================
# AlphaFold2 capture
# =============================================================================

def callbacks(enabled: bool = True, confidence_directory: Path | None = None):
    """Return native extra-metric wrapper, prediction callback, and compact results."""
    # ColabFold is installed only inside the AF2 Modal image.
    extra_ptm = importlib.import_module("colabfold.alphafold.extra_ptm")

    original = extra_ptm.get_chain_and_interface_metrics
    compact = {}

    def capture_pairs(result, asym_id):
        asym = np.asarray(asym_id).reshape(-1)
        raw = result.get("pae_matrix_with_logits")
        if raw is None:
            raise ValueError("AF2 native PAE distributions were not retained")
        if raw is not None:
            data = {"predicted_aligned_error": raw}
            contacts = extra_ptm.get_contact_map(result, 8) if enabled else None
            ids = list(dict.fromkeys(asym.tolist()))
            native = np.zeros((len(ids), len(ids)))
            active = {}
            for first, second in combinations(range(len(ids)), 2):
                left, right = (
                    np.flatnonzero(asym == ids[first]),
                    np.flatnonzero(asym == ids[second]),
                )
                bounds = (int(left[0]), int(left[-1]), int(right[0]), int(right[-1]))
                native[first, second] = native[second, first] = float(
                    extra_ptm.get_pairwise_iptm(data, asym, *bounds).max()
                )
                if enabled:
                    active[f"{ids[first]}:{ids[second]}"] = float(
                        extra_ptm.get_actifptm_probs(
                            data, asym, contacts, *bounds
                        ).max()
                    )
            result["foldism_pair_scores"] = (native, active, ids)
            if confidence_directory is not None:
                breaks = np.asarray(raw["breaks"])
                step = breaks[1] - breaks[0]
                centers = np.concatenate([breaks + step / 2, [breaks[-1] + 3 * step / 2]])
                result["foldism_confidence_arrays"] = {
                    "pae_logits": np.asarray(raw["logits"]),
                    "pae_bin_centers": centers,
                    "contact_probs": np.asarray(extra_ptm.get_contact_map(result, 8)),
                    "chain_ids": asym,
                }

    def extra(result, asym_id, *args, **kwargs):
        output = original(result, asym_id, *args, **kwargs)
        capture_pairs(result, asym_id)
        return output

    def prediction(protein, lengths, result, features, tag):
        chains = np.asarray(protein.chain_index)
        labels = np.asarray([str(int(value)) for value in chains])
        if (len(np.unique(chains)) > 1 or confidence_directory is not None) and "foldism_pair_scores" not in result:
            capture_pairs(result, features.get("asym_id", chains + 1))
            result.pop("pae_matrix_with_logits", None)
        native, active, ids = result.get(
            "foldism_pair_scores", (None, None, list(dict.fromkeys(chains.tolist())))
        )
        if len(ids) == 2 and native is None:
            native = [result.get("iptm")] if result.get("iptm") is not None else None
        pae = result.get("predicted_aligned_error")
        if pae is None:
            return
        scores = calculate(
            np.asarray(pae)[: len(chains), : len(chains)],
            np.asarray(protein.atom_positions)[:, 1],
            labels,
            native,
            active,
            enabled,
            ids,
        )
        # Match native PDB chain labels, preserving the chain order used for ipDAE.
        scores["pairs"] = [
            [chr(65 + int(value)) for value in pair] for pair in scores["pairs"]
        ]
        compact[tag[0]] = scores
        if confidence_directory is not None:
            arrays = result.get("foldism_confidence_arrays")
            if arrays is None:
                raise ValueError("AF2 confidence arrays were requested but unavailable")
            confidence_directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(confidence_directory / f"{tag[0]}_confidence_arrays.npz", **arrays)

    return original, extra, prediction, compact


def attach(directory: Path, compact: dict) -> None:
    """Insert full-precision confidence values in matching ranked score JSON files."""
    for path in directory.rglob("*_scores_*.json"):
        matching = [
            scores for tag, scores in compact.items() if path.stem.endswith(tag)
        ]
        if len(matching) != 1:
            raise ValueError(f"Could not map AF2 confidence callback to {path.name}")
        data = json.loads(path.read_text())
        data["interface_confidence"] = matching[0]
        path.write_text(json.dumps(data))


# =============================================================================
# Percentile ranks against measured ProteinBase non-binders
# =============================================================================

METRICS = ("iptm", "ipsae", "ipdae", "actifptm")


def target_percentile(value: float, groups: Mapping[str, Sequence[float]]) -> float:
    """Return the mean target-specific midrank percentile.

    Each target has equal weight. Values within each target must be sorted.
    Tied scores contribute half their mass, including ties at zero.
    """
    if not isfinite(value) or not 0 <= value <= 1 or not groups:
        raise ValueError("A finite score in [0, 1] and nonempty targets are required")
    ranks = []
    for values in groups.values():
        if not values:
            raise ValueError("Reference targets must contain scores")
        below = bisect_left(values, value)
        through = bisect_right(values, value)
        ranks.append((below + through) / (2 * len(values)))
    return 100 * sum(ranks) / len(ranks)


def annotate_percentiles(
    method: str, confidence: dict, *, use_msa: bool = False
) -> dict:
    """Copy scores and append reference ranks for eligible two-chain MSA folds.

    Unsupported methods, versions, and inputs retain their raw scores. Their
    percentile values are None, with an explicit reason in the metadata.
    """
    result = deepcopy(confidence)
    pairs = result.get("pairs", [])
    metadata = {
        key: value
        for key, value in REFERENCE["metadata"].items()
        if key not in ("case_ids", "target_counts")
    }
    result["percentiles"] = {metric: [None] * len(pairs) for metric in METRICS}
    result["percentile_status"] = {
        metric: ["unavailable"] * len(pairs) for metric in METRICS
    }
    result["percentile_reference"] = metadata
    if method not in REFERENCE["distributions"]:
        metadata["status"] = "no_model_reference"
    elif not use_msa:
        metadata["status"] = "requires_msa"
    elif result.get("version") != metadata["score_version"]:
        metadata["status"] = "score_version_mismatch"
    elif len(pairs) != 1 or len(pairs[0]) != 2 or pairs[0][0] == pairs[0][1]:
        metadata["status"] = "requires_two_protein_chains"
    else:
        metadata["status"] = "available"
        for metric in METRICS:
            values = result.get(metric, [])
            statuses = result.get("status", {}).get(metric, [])
            value = values[0] if len(values) == 1 else None
            status = statuses[0] if len(statuses) == 1 else "missing_status"
            no_contacts = (
                metric == "ipdae"
                and value is None
                and status == "no_ca_contacts_within_8_angstrom"
            )
            if no_contacts:
                value = 0.0
            elif status != "complete":
                result["percentile_status"][metric][0] = status
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value <= 1
            ):
                result["percentile_status"][metric][0] = "invalid_or_missing_score"
                continue
            result["percentiles"][metric][0] = target_percentile(
                value, REFERENCE["distributions"][method][metric]
            )
            result["percentile_status"][metric][0] = (
                "no_contacts_ranked_as_zero" if no_contacts else "complete"
            )
    return result

# =============================================================================
# Frozen reference: public ProteinBase non-binders (200 designs, 23 targets)
# =============================================================================

REFERENCE = json.loads(
    '{"metadata": {"version": "proteinbase-nonbinders-within-target400-20261101-v1", "score_version": "protein-pairs-v1-ipsae-20260103", "nonbinders": 200, "targets": 23, "target_counts": {"15-pgdh": 1, "der21": 4, "der7": 5, "egfr": 24, "fcrn": 6, "fgf-r1": 6, "hnmt": 1, "human-ambp": 5, "human-calmodulin-1-calm1-holo-cacl2": 5, "human-idi2": 4, "human-insulin-receptor": 3, "human-mzb1-perp1": 1, "human-pdgfr-beta": 8, "human-phyh": 2, "human-vegf110": 25, "ifnar2": 8, "il7r": 21, "mdm2": 4, "myostatin-gdf8": 6, "pd-l1": 24, "rbx1": 9, "spcas9": 4, "trem2": 24}, "weighting": "equal_target", "ties": "midrank", "formula": "100 * mean_target((n_below + 0.5 * n_equal) / n_target)", "source": "Public ProteinBase non-binders: 200 designs across 23 targets, equal per target", "msa": "unpaired, shared frozen ColabFold MSAs", "protein_chains": 2, "chain_order": "binder first, target second", "ipdae_revision": "b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06", "ipdae_definition": "maximum; forward PAE block and its transpose", "ipdae_no_contacts": "zero for reference ranking; raw score remains unavailable", "interpretation": "Reference rank, not binding probability or expected PPV", "case_ids": ["15pgdh_rough_owl_pine_8353ce6d", "der21_bright_eagle_iron_339499be", "der21_hollow_fox_thorn_12a57790", "der21_pale_toad_willow_572822d0", "der21_strong_gecko_snow_19b449a9", "der7_amber_tiger_crystal_fd4f2884", "der7_bright_bear_lotus_a2b62611", "der7_deep_shark_cedar_c3c3c75a", "der7_jade_jaguar_marble_2f0fd8ea", "der7_small_otter_ice_56c54666", "egfr_amber_bear_plume_47985f3d", "egfr_bright_otter_iron_d6160d1b", "egfr_calm_panther_vine_a60a7f6a", "egfr_dark_raven_orchid_365e7f89", "egfr_deep_crow_wave_b75fb17c", "egfr_gentle_crane_moss_7514ebc1", "egfr_hollow_ant_topaz_d271a7ed", "egfr_hollow_gecko_pine_3718350a", "egfr_jade_bee_fern_d2d39e05", "egfr_lunar_zebra_ember_787c5654", "egfr_mellow_bison_birch_87b3b67f", "egfr_mellow_jaguar_plume_6d627413", "egfr_mellow_panther_ember_53fd10ad", "egfr_quick_toad_marble_c76c18da", "egfr_quiet_panther_cloud_31b7a2dd", "egfr_rapid_ox_vine_4ae9fd7b", "egfr_silent_yak_frost_98a7c4ee", "egfr_small_ram_ash_97199243", "egfr_soft_bee_leaf_9b0d8b28", "egfr_soft_swan_moss_8ee52f4e", "egfr_steady_goat_topaz_ccb9caba", "egfr_swift_panther_ice_8dbbfbcd", "egfr_violet_vole_opal_2fb5fa0f", "egfr_young_owl_dust_c7921552", "fcrn_gentle_heron_wave_64805d34", "fcrn_lunar_ox_ruby_877a9554", "fcrn_pale_lynx_opal_c2e118af", "fcrn_shy_deer_granite_28214ff8", "fcrn_swift_fox_lotus_d4818986", "fcrn_vast_eagle_snow_e445566d", "fgfr1_amber_ram_thorn_7349a44f", "fgfr1_frozen_bison_pine_b9d3ab27", "fgfr1_lunar_yak_flint_9215d881", "fgfr1_noble_zebra_ash_c3b0cdc9", "fgfr1_steady_gecko_onyx_2fef977a", "fgfr1_swift_cobra_birch_e3f8ec95", "hnmt_rough_bear_flint_f64fb3a4", "humanambp_deep_wolf_ash_95802104", "humanambp_ivory_yak_ivy_7d90ce79", "humanambp_jade_shark_clay_97cb90b9", "humanambp_rapid_lion_maple_bbac5ceb", "humanambp_silver_seal_clay_b9fdbd48", "humancalmodulin1calm1holocacl2_green_hawk_frost_c4c5ade8", "humancalmodulin1calm1holocacl2_ivory_toad_plume_48b3a6f1", "humancalmodulin1calm1holocacl2_lunar_ox_birch_3c4ddf68", "humancalmodulin1calm1holocacl2_steady_tiger_pine_4e1af9cf", "humancalmodulin1calm1holocacl2_violet_shark_willow_29e0f7cc", "humanidi2_green_ibis_flint_31c0e181", "humanidi2_noble_quail_dust_3fd3022c", "humanidi2_rapid_bison_topaz_71afdfcc", "humanidi2_silver_gecko_cypress_9df7485b", "humaninsulinreceptor_crimson_ibis_iron_3758109f", "humaninsulinreceptor_silent_zebra_flint_4bac560e", "humaninsulinreceptor_steady_ant_oak_a4c90254", "humanmzb1perp1_ivory_heron_opal_ebd788b3", "humanpdgfrbeta_bright_swan_reed_85f2def5", "humanpdgfrbeta_brisk_bat_vine_4e93e76e", "humanpdgfrbeta_crimson_quail_cypress_e8e9c4c6", "humanpdgfrbeta_golden_zebra_bronze_fb8de63b", "humanpdgfrbeta_quick_gecko_orchid_57ef37d7", "humanpdgfrbeta_small_kiwi_jade_73b81b88", "humanpdgfrbeta_steady_otter_frost_ed9aeec9", "humanpdgfrbeta_strong_wolf_leaf_3927ac58", "humanphyh_deep_boar_frost_27440ffe", "humanphyh_lunar_seal_ruby_7a091f70", "humanvegf110_amber_crane_ruby_53fae0a2", "humanvegf110_amber_shark_birch_e5b4beb5", "humanvegf110_brisk_heron_marble_54bee2a9", "humanvegf110_dark_raven_lava_709a16ec", "humanvegf110_gentle_bee_vine_69f43ae0", "humanvegf110_green_ox_dust_6506a65e", "humanvegf110_hollow_boar_rose_b24f4ec7", "humanvegf110_ivory_dove_willow_3313b7cd", "humanvegf110_misty_swan_onyx_1e8435bd", "humanvegf110_noble_dove_clay_670a81be", "humanvegf110_quiet_zebra_flint_6b4dabbd", "humanvegf110_rapid_deer_clay_11948000", "humanvegf110_rough_owl_crystal_512779f7", "humanvegf110_scarlet_boar_leaf_4d4c58cc", "humanvegf110_scarlet_heron_pearl_4692edcf", "humanvegf110_shy_lion_rose_eb8a60e2", "humanvegf110_small_ant_orchid_15759c0e", "humanvegf110_small_heron_ember_59a1bc91", "humanvegf110_soft_cobra_stone_7dddbc4b", "humanvegf110_soft_crane_flint_e3e19a88", "humanvegf110_soft_mole_vine_708dfe9d", "humanvegf110_steady_fox_opal_3c82118e", "humanvegf110_swift_jaguar_marble_61aeb9cb", "humanvegf110_violet_dove_orchid_76c64133", "humanvegf110_wild_crow_granite_2963dd79", "ifnar2_crimson_mole_lava_3c05963c", "ifnar2_gentle_toad_oak_b5db6d54", "ifnar2_jade_jaguar_frost_4609d916", "ifnar2_noble_ram_crystal_d409e976", "ifnar2_rapid_gecko_fern_95331472", "ifnar2_scarlet_heron_quartz_cbfe0613", "ifnar2_scarlet_panda_thorn_c1d08f36", "ifnar2_young_cobra_plume_9f94972a", "il7r_brisk_yak_opal_04d2fe93", "il7r_dark_bat_orchid_73265b3a", "il7r_dark_falcon_maple_1416384f", "il7r_frozen_cobra_moss_d10aa9a1", "il7r_frozen_swan_fern_d2131cf1", "il7r_gentle_boar_flint_338b337a", "il7r_hollow_deer_pine_3e650e4d", "il7r_lunar_wolf_quartz_dd7aa363", "il7r_noble_mole_oak_6d590982", "il7r_pale_falcon_cypress_1c92238d", "il7r_quiet_moth_granite_a115a192", "il7r_rapid_kiwi_flint_5d802b82", "il7r_rapid_panda_stone_92859fd2", "il7r_shy_tiger_pine_a342422b", "il7r_silent_ibis_onyx_82dc41dd", "il7r_soft_otter_willow_1586b2d8", "il7r_soft_quail_ember_dc40fedf", "il7r_solid_ox_ruby_e95eb061", "il7r_solid_panda_quartz_7aeb0c33", "il7r_strong_cobra_iron_8e100cac", "il7r_wild_ram_vine_b942195b", "mdm2_lunar_cat_iron_59a8abe1", "mdm2_rapid_heron_pine_243722af", "mdm2_solid_boar_wave_c685d80c", "mdm2_wild_ox_stone_b09016d6", "myostatingdf8_calm_orca_willow_0432ec7f", "myostatingdf8_frozen_bison_sand_1b7fba3c", "myostatingdf8_golden_bat_wave_59bcc6d2", "myostatingdf8_mellow_panther_clay_94a54555", "myostatingdf8_pale_quail_dust_9d3970bc", "myostatingdf8_strong_ram_dust_296d9a0e", "pdl1_bright_bear_opal_de253bbe", "pdl1_bright_orca_ash_71054d1d", "pdl1_bright_panther_frost_5f206c7c", "pdl1_bright_raven_ember_d7b024d3", "pdl1_calm_cobra_plume_3bb8f2be", "pdl1_calm_deer_ice_b61fc982", "pdl1_calm_moth_maple_4a0203a5", "pdl1_deep_jaguar_cypress_21839df3", "pdl1_hollow_ant_thorn_72c76b3c", "pdl1_hollow_quail_moss_16bda3bf", "pdl1_ivory_hawk_stone_ceb844ef", "pdl1_misty_bee_cloud_4921c3bf", "pdl1_pale_moth_maple_25cb9b95", "pdl1_pale_toad_oak_f1ea8025", "pdl1_quick_quail_oak_14ecd208", "pdl1_rapid_crane_jade_d0f4d17c", "pdl1_rapid_crow_lotus_f09d9ca5", "pdl1_rough_jaguar_frost_048deaaf", "pdl1_silent_crane_frost_e50218d0", "pdl1_silver_ibis_ember_66b683ea", "pdl1_small_ant_oak_fb293b1d", "pdl1_soft_cat_maple_d5916f6f", "pdl1_steady_owl_cedar_ee70094e", "pdl1_violet_deer_cypress_82fa0da7", "rbx1_calm_jaguar_iron_8086b1ba", "rbx1_crimson_bear_sand_3824e13e", "rbx1_dark_cat_willow_b90446e0", "rbx1_gentle_ibis_cloud_56347e57", "rbx1_hollow_panda_maple_5e70e399", "rbx1_lunar_ram_clay_5e282122", "rbx1_rapid_wolf_rose_32cb0e5d", "rbx1_silver_lynx_rose_72f3eab0", "rbx1_wild_jaguar_reed_99db9d88", "spcas9_bright_ox_ruby_8fc798cb", "spcas9_ivory_toad_pearl_7d1d7d9b", "spcas9_mellow_crane_orchid_527dd365", "spcas9_radiant_ant_flint_bd692eac", "trem2_azure_dove_bronze_87f9ed20", "trem2_azure_lion_jade_965f3f38", "trem2_calm_hawk_quartz_fe1d8f9b", "trem2_crimson_falcon_willow_fefef5a7", "trem2_gentle_bear_dust_ab4a510d", "trem2_golden_swan_birch_56c8444f", "trem2_hollow_boar_wave_3feea8ed", "trem2_hollow_kiwi_leaf_43db01aa", "trem2_ivory_gecko_opal_367cce53", "trem2_jade_kiwi_ivy_f4b108f9", "trem2_lunar_deer_snow_986f0f0b", "trem2_mellow_goat_vine_1d639b98", "trem2_mellow_kiwi_lotus_42efc86a", "trem2_misty_lion_opal_49af0889", "trem2_noble_quail_orchid_569796e8", "trem2_rapid_lynx_granite_2cfa50fb", "trem2_small_dove_leaf_1525bb42", "trem2_small_wolf_cloud_68c124e8", "trem2_soft_goat_clay_d3b026cc", "trem2_vast_otter_topaz_643a0488", "trem2_violet_fox_cloud_7d7e1054", "trem2_violet_mole_cloud_9472b1ea", "trem2_wild_otter_pearl_5b926628", "trem2_young_boar_granite_0a539ef0"]}, "distributions": {"esmfold2": {"iptm": {"15-pgdh": [0.7393465042114258], "der21": [0.47543811798095703, 0.5264196395874023, 0.5340701937675476, 0.5796564221382141], "der7": [0.5036216378211975, 0.5301116704940796, 0.5527442097663879, 0.6039679050445557, 0.6549459099769592], "egfr": [0.14846165478229523, 0.17926539480686188, 0.18296964466571808, 0.2053166925907135, 0.23491160571575165, 0.24512873589992523, 0.2875725328922272, 0.2980363965034485, 0.36729246377944946, 0.3715971112251282, 0.3808078169822693, 0.39168885350227356, 0.4018079936504364, 0.4247046411037445, 0.42585045099258423, 0.43472447991371155, 0.5234546065330505, 0.5425412058830261, 0.5426525473594666, 0.5552316308021545, 0.5557543039321899, 0.5746779441833496, 0.580392599105835, 0.5929527878761292], "fcrn": [0.21366897225379944, 0.260882705450058, 0.2951594591140747, 0.29905256628990173, 0.35551774501800537, 0.36800676584243774], "fgf-r1": [0.27481809258461, 0.2810973823070526, 0.2820875346660614, 0.28647780418395996, 0.28879061341285706, 0.2888852059841156], "hnmt": [0.726051926612854], "human-ambp": [0.3682287335395813, 0.4503113329410553, 0.4755841791629791, 0.47738051414489746, 0.6847972869873047], "human-calmodulin-1-calm1-holo-cacl2": [0.12724408507347107, 0.238092303276062, 0.24672235548496246, 0.2672562897205353, 0.28961890935897827], "human-idi2": [0.6619319915771484, 0.7386255860328674, 0.7412304878234863, 0.7439225912094116], "human-insulin-receptor": [0.12744870781898499, 0.13129888474941254, 0.15137654542922974], "human-mzb1-perp1": [0.5329262614250183], "human-pdgfr-beta": [0.3543705642223358, 0.42257723212242126, 0.4230499565601349, 0.44283828139305115, 0.44489428400993347, 0.4569118022918701, 0.46851131319999695, 0.5095648765563965], "human-phyh": [0.1878402680158615, 0.5954873561859131], "human-vegf110": [0.17722173035144806, 0.2139963060617447, 0.2213257998228073, 0.22922103106975555, 0.2344762086868286, 0.2529717683792114, 0.2971557676792145, 0.303636372089386, 0.3042010962963104, 0.33920133113861084, 0.33965399861335754, 0.35407671332359314, 0.3739682734012604, 0.38188639283180237, 0.3835187554359436, 0.39557263255119324, 0.3983086049556732, 0.3984917998313904, 0.422552227973938, 0.4335087239742279, 0.44180479645729065, 0.45962268114089966, 0.48524484038352966, 0.4871083199977875, 0.5139033794403076], "ifnar2": [0.15678328275680542, 0.22195671498775482, 0.23648174107074738, 0.3473980128765106, 0.36914071440696716, 0.520805299282074, 0.5375518798828125, 0.5636522769927979], "il7r": [0.1552395522594452, 0.47363850474357605, 0.4768897294998169, 0.5179683566093445, 0.5249537825584412, 0.5270782709121704, 0.5452606081962585, 0.555514395236969, 0.5575809478759766, 0.5625209808349609, 0.5827348232269287, 0.5874708890914917, 0.5900222063064575, 0.5936778783798218, 0.5955706238746643, 0.5981696844100952, 0.598662793636322, 0.6172396540641785, 0.6300877332687378, 0.6544656753540039, 0.6603223085403442], "mdm2": [0.042264942079782486, 0.08310747146606445, 0.09649890661239624, 0.22192637622356415], "myostatin-gdf8": [0.191812202334404, 0.30788150429725647, 0.47681576013565063, 0.4866306781768799, 0.5016794204711914, 0.554316520690918], "pd-l1": [0.18770501017570496, 0.19621092081069946, 0.25233912467956543, 0.3129544258117676, 0.3223513066768646, 0.4013030230998993, 0.4369198977947235, 0.45222190022468567, 0.4557987451553345, 0.46689373254776, 0.48001736402511597, 0.4957025647163391, 0.543093740940094, 0.5441325902938843, 0.5502586364746094, 0.5679064393043518, 0.5873433351516724, 0.5915018916130066, 0.5938740968704224, 0.5953144431114197, 0.6207155585289001, 0.6213270425796509, 0.6362322568893433, 0.6482166051864624], "rbx1": [0.06211225688457489, 0.08532755821943283, 0.10917135328054428, 0.10984977334737778, 0.20188939571380615, 0.2768856883049011, 0.3481808304786682, 0.3892889618873596, 0.4271499514579773], "spcas9": [0.20281115174293518, 0.29100555181503296, 0.3096484839916229, 0.32432159781455994], "trem2": [0.10826538503170013, 0.15612384676933289, 0.1653495579957962, 0.2670794427394867, 0.30524665117263794, 0.3190453350543976, 0.3198974132537842, 0.37119778990745544, 0.40815269947052, 0.4256219267845154, 0.42604902386665344, 0.4346528947353363, 0.4729505181312561, 0.4791574478149414, 0.4811217486858368, 0.49209466576576233, 0.4956960082054138, 0.5015116930007935, 0.5034547448158264, 0.5202040076255798, 0.5223820805549622, 0.5263656973838806, 0.5372913479804993, 0.5397199392318726]}, "ipsae": {"15-pgdh": [0.7265282071504137], "der21": [0.2887127564515525, 0.2894523377718181, 0.3667325223733184, 0.36792124688689726], "der7": [0.3140936378335728, 0.435441109979222, 0.46556770214477666, 0.5856136062476999, 0.6462442740829712], "egfr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01082672498972711, 0.08273799341201073, 0.22229361610324055, 0.3315307086962272, 0.39563330409147207, 0.420203635605473, 0.48093610172387713, 0.504147867571505, 0.5273912670120534, 0.5420088850479171, 0.5493309568609673, 0.6055510973942787, 0.6177157624325028, 0.6218262177178776, 0.6456781249679774, 0.6459491959411824, 0.6526857298830753, 0.6814740701431288], "fcrn": [0.0, 0.0, 0.015584520476464407, 0.01889501281122806, 0.023431312795099284, 0.03968996687460526], "fgf-r1": [0.45743049857863377, 0.48902979668219965, 0.4944110730463002, 0.5020525460837814, 0.5044346217921389, 0.5099226805402407], "hnmt": [0.7386947230059021], "human-ambp": [0.07515563478158155, 0.25308848774947723, 0.2892266091149574, 0.3499130237219934, 0.7273366845348015], "human-calmodulin-1-calm1-holo-cacl2": [0.0, 0.011118635469356748, 0.01127168500604935, 0.014159064817971386, 0.018547732723927686], "human-idi2": [0.4576098660630719, 0.7106872155889351, 0.7309532923499127, 0.7361014096068413], "human-insulin-receptor": [0.0, 0.0, 0.0], "human-mzb1-perp1": [0.5773435572723103], "human-pdgfr-beta": [0.33745533918941645, 0.3740710137147998, 0.43480605412107476, 0.45650294514768003, 0.4628836138324153, 0.468651870086404, 0.5344301890394819, 0.5564808508114445], "human-phyh": [0.0, 0.6192309190709662], "human-vegf110": [0.0, 0.0, 0.0, 0.011213501108193575, 0.011724498852071497, 0.012607057698915656, 0.01338796637480982, 0.01452140658487489, 0.01520897554390507, 0.015296733941148158, 0.01637204720694525, 0.01825268276826809, 0.018461693447686424, 0.02312530048557376, 0.03576703172211259, 0.07010633833371802, 0.08767899958943813, 0.10975625142185316, 0.11074567796581139, 0.15060579511556219, 0.1540716559370395, 0.17032874105599202, 0.20480984341415256, 0.21015381649741696, 0.2675640766737725], "ifnar2": [0.0, 0.0, 0.0, 0.05376876307138583, 0.1659969004203329, 0.43075217060638465, 0.5157165120614318, 0.5237340894421494], "il7r": [0.0, 0.23823534890168846, 0.26932361412496236, 0.3068626223548661, 0.37741089480858586, 0.3791172263229057, 0.40656603922155915, 0.41507801964435564, 0.4415188266219771, 0.4542993334715375, 0.4681163592052511, 0.4710800346110336, 0.5036771452088613, 0.542350560074099, 0.5547686123374378, 0.5758312036463721, 0.6134549474965799, 0.6137920765578438, 0.6239174395220688, 0.6581000167999198, 0.6704700420623712], "mdm2": [0.0, 0.0, 0.0, 0.33115650008373226], "myostatin-gdf8": [0.0, 0.043539625769392756, 0.43943371130222547, 0.4724173257800625, 0.5293841722315815, 0.556504767113581], "pd-l1": [0.0, 0.010269815051288425, 0.012170427765924712, 0.014635405694057784, 0.07206353760008978, 0.25851667290293806, 0.26090449965091295, 0.31138252084690077, 0.32634032425181064, 0.4092262434082726, 0.41942373518582154, 0.4636488152397816, 0.46477319304523396, 0.5054412183953336, 0.5492550740172324, 0.5571926165943631, 0.5788429995594158, 0.5799324113003692, 0.6140932199010594, 0.6204339321472526, 0.6935683011916502, 0.7011232879109774, 0.7666604174297093, 0.7920050639695201], "rbx1": [0.0, 0.0, 0.0, 0.0, 0.20597187095130567, 0.28887569857405804, 0.3103968450303563, 0.4238979728735411, 0.7769328469739044], "spcas9": [0.0, 0.31083085642546726, 0.4295388049104726, 0.4647465466860854], "trem2": [0.0, 0.0, 0.011896646112586817, 0.023226528118338082, 0.07703186556260343, 0.08483382960004555, 0.08700598025811664, 0.16587796255478005, 0.22352030127437436, 0.3355883875781578, 0.3663280468866239, 0.37717768808629226, 0.38475434072513204, 0.3991352532148731, 0.41972853695748297, 0.4234992390009598, 0.462742930224421, 0.4720605145570084, 0.4818144509236361, 0.5024960754113778, 0.5073767839706239, 0.5103476970013507, 0.533996192004149, 0.614828811411087]}, "ipdae": {"15-pgdh": [0.19992815114526952], "der21": [0.04231371727495459, 0.050082258321280604, 0.09475290636984253, 0.11329372790943311], "der7": [0.05814989779787426, 0.11031921039512954, 0.14377042756885874, 0.2237237226324007, 0.26093955687543435], "egfr": [0.0015011584752573604, 0.0018577424644157755, 0.002007693126869139, 0.0033441613130908154, 0.004390001195701316, 0.005177502108202504, 0.009633900913026546, 0.01896775955419982, 0.025561295211366308, 0.06170359439561234, 0.07370008587858436, 0.0790916608668977, 0.0851480956889592, 0.09079975356374023, 0.10491630340944672, 0.1102660001538958, 0.11956616614513313, 0.12804709525132943, 0.1621746372436343, 0.1815935787261031, 0.18433208151321603, 0.18709594337322605, 0.2146823471598291, 0.21896220837390168], "fcrn": [0.0047244411416154835, 0.00669002776544893, 0.012780743748391164, 0.013641063262735838, 0.013706883420682736, 0.01766927961042627], "fgf-r1": [0.0809819991030101, 0.08856908612181963, 0.09947240180972754, 0.11157557518402066, 0.11243716697882175, 0.1166704078007868], "hnmt": [0.2778825946495331], "human-ambp": [0.017903840685425856, 0.0350297363754863, 0.041360174362864904, 0.05308039312141026, 0.25462279736782906], "human-calmodulin-1-calm1-holo-cacl2": [0.0027095055491950203, 0.005209565345892166, 0.0071801969275275684, 0.007546495312964851, 0.01049500989873236], "human-idi2": [0.04411047066044822, 0.20461356313785117, 0.2330547427358126, 0.28252927609634526], "human-insulin-receptor": [0.001148441376548769, 0.001304985055396433, 0.0016500583974562765], "human-mzb1-perp1": [0.325762540223304], "human-pdgfr-beta": [0.04846150799607277, 0.07477466874299371, 0.0887023274266219, 0.09253073755413516, 0.09617826448755745, 0.12829877601058248, 0.12905957955803773, 0.15965359682672287], "human-phyh": [0.002791168352726768, 0.1211756134733411], "human-vegf110": [0.00298604945528363, 0.0032244985148651615, 0.003854777529301018, 0.004381035162348943, 0.004710600048638342, 0.004899394372933745, 0.0057237240492087425, 0.006199196071745596, 0.007514793485292788, 0.008451912644128458, 0.009946805935659773, 0.010528575216617634, 0.012955079790714684, 0.0156441301618858, 0.01954748232934428, 0.02350457351359535, 0.02395352151082121, 0.029083710471408064, 0.029452150303960648, 0.03502907702704144, 0.03714959422196997, 0.037429798358254156, 0.04099608457884932, 0.04247356205252031, 0.0693520029423501], "ifnar2": [0.004811670984065701, 0.007454996085050077, 0.009201974505673277, 0.02580135839697805, 0.027913399772082684, 0.10583821425523876, 0.1806461891163022, 0.1823832235563216], "il7r": [0.003459934850624126, 0.03708499950360874, 0.04610810953728832, 0.050449450236796076, 0.07270838180975062, 0.07344467336045138, 0.07766998919561766, 0.08126652905062381, 0.10294171969099959, 0.10494970181306318, 0.10831783851097636, 0.11091188502673163, 0.11249488505479943, 0.1602393311337159, 0.18527531987190776, 0.19333612909971748, 0.22503058602983295, 0.2350856906487197, 0.28854192284223856, 0.3203918006377412, 0.35968063428952923], "mdm2": [0.00126228722057449, 0.0026301940156685357, 0.0031346821533765512, 0.11205046748995713], "myostatin-gdf8": [0.0036527540936287456, 0.018691965074248265, 0.09215749670518511, 0.1129616769692168, 0.16092937281855602, 0.18754575821688838], "pd-l1": [0.006279956234950841, 0.00935505076464808, 0.009609122118769752, 0.01199911410448177, 0.030259967946190142, 0.03051098407310006, 0.0428058953925581, 0.050248250814484585, 0.07569445028588982, 0.10642566718740071, 0.1252393667358106, 0.12534787046499898, 0.13092147312272365, 0.16728529232933134, 0.19038631501648295, 0.24780304889520638, 0.2562928423497118, 0.25858830597248134, 0.2794057781717851, 0.2928930214230284, 0.3329208059166565, 0.3739872437138492, 0.38672987463887626, 0.4458058252585857], "rbx1": [0.00266557385456032, 0.003394448980901451, 0.004169034083744587, 0.0042159383662610595, 0.047639439433101095, 0.10637822305900078, 0.1398021682655222, 0.16844381689994437, 0.4915066466795202], "spcas9": [0.0028709349427819375, 0.03721234284668901, 0.06702306024675882, 0.08963415373970997], "trem2": [0.0026883000409817035, 0.008593188791819707, 0.008908832025226335, 0.0176884047191714, 0.027781054216932888, 0.029358538735501972, 0.03507353675804033, 0.044905537588861885, 0.04750620789723969, 0.08746444370861234, 0.11747530564065134, 0.12389778353027477, 0.12426044337811577, 0.12631321273074783, 0.12724910708515472, 0.15419861112451513, 0.16236684484023187, 0.17736671202500665, 0.18351546701560456, 0.22807401108400097, 0.23358183443584402, 0.24233331178125164, 0.3345977737127591, 0.41272996634159953]}, "actifptm": {"15-pgdh": [0.9390130808741157], "der21": [0.7523432354237691, 0.7827675746347308, 0.8580152262216768, 0.8698537230314539], "der7": [0.8449170874541263, 0.8964260572026764, 0.8995425310450242, 0.9359688965406233, 0.9366859355658813], "egfr": [0.21786141605199108, 0.2197524994972357, 0.3063281116856844, 0.390689268128115, 0.4159889440692098, 0.46661975512873044, 0.6790548123250882, 0.7429698394375206, 0.7544582083037347, 0.8678925191223408, 0.8863433526498656, 0.8908596250264028, 0.9073556964536552, 0.9094911657600625, 0.9110131995152406, 0.9146133977690681, 0.9197065245781341, 0.9197454674388983, 0.9291168795414305, 0.9312537155868041, 0.9314758530638736, 0.9434312847881673, 0.944086155744852, 0.9450601417403299], "fcrn": [0.38215710616191184, 0.4094051574895762, 0.5967642471253944, 0.618272793986528, 0.6484046471652601, 0.7436132436727492], "fgf-r1": [0.9158848860342994, 0.9176259511509517, 0.9188440635281004, 0.9237666879855982, 0.9244999182758904, 0.9259003840350005], "hnmt": [0.9545033664258126], "human-ambp": [0.6221169795629102, 0.7534394659433055, 0.8202495895723008, 0.8301652070426863, 0.9344053498575546], "human-calmodulin-1-calm1-holo-cacl2": [0.22840688092914113, 0.3180135182749883, 0.37687965531063644, 0.4229192447334039, 0.44831340835674627], "human-idi2": [0.8373949357578645, 0.9299731627321137, 0.9386511311590958, 0.9395151959818593], "human-insulin-receptor": [0.1744357217311058, 0.20488377231523597, 0.2968214526371079], "human-mzb1-perp1": [0.9318074077152879], "human-pdgfr-beta": [0.8699897385070277, 0.8785233523907066, 0.8933370847453767, 0.9010437751186775, 0.9012444940168012, 0.9066700999392139, 0.9217908048724476, 0.9253791050908453], "human-phyh": [0.31672994421532374, 0.9372336554701718], "human-vegf110": [0.2991877409538092, 0.38650668462326704, 0.4094829479438898, 0.4323586132626754, 0.46307385662781914, 0.48578981418666983, 0.5062767120030003, 0.5466322747968682, 0.5545028284722162, 0.5701421863162154, 0.6143982428514594, 0.6553609957085688, 0.6718772707377009, 0.672922779038617, 0.686826960036262, 0.7140262573606283, 0.715197381988235, 0.7171065977138078, 0.7628574748684708, 0.7635080116994803, 0.7638383203342893, 0.7698152142256173, 0.7795907431147155, 0.8015238061706325, 0.8324415164192726], "ifnar2": [0.2215901898636457, 0.39857933538096985, 0.4453148921316714, 0.6554214036236681, 0.7348869320383326, 0.8897021242518899, 0.9199821361719432, 0.9264402723415093], "il7r": [0.22377848417599114, 0.7853364088379627, 0.8002875215585877, 0.8299509782646234, 0.860354736127988, 0.861386948864151, 0.8616267355298297, 0.873376804802068, 0.8772366075019198, 0.8807678272170151, 0.8883752297028139, 0.8909088043290918, 0.9141173404939134, 0.9143339687218369, 0.9165547263641902, 0.9200648553356207, 0.931184928618775, 0.9358695029115274, 0.9403590059929899, 0.9506207318027251, 0.9543519023049812], "mdm2": [0.048965953484200755, 0.14395642067785608, 0.21909756303410396, 0.826153519247714], "myostatin-gdf8": [0.4152061477073157, 0.6348991025257587, 0.8978937601466511, 0.9005678231076338, 0.9231230601854827, 0.9309208098524627], "pd-l1": [0.33571728160799463, 0.3818368331644727, 0.4652518865794494, 0.5665662982412624, 0.6098494861128727, 0.7061560888896403, 0.792612860256049, 0.8341807663208343, 0.8403018080988993, 0.881850232770361, 0.8955642630240837, 0.896323193997985, 0.9046891704419046, 0.9132663381958859, 0.9322807108556275, 0.9365253636952051, 0.9370947995566039, 0.9381920764590549, 0.9417605250352017, 0.9457163104184351, 0.9524672703854906, 0.9594479915622112, 0.9600744563207797, 0.9610997449127199], "rbx1": [0.1186937350647249, 0.12380819613261367, 0.17596069691163166, 0.24624482261519517, 0.7993354164748325, 0.8272107261347845, 0.8700132035144242, 0.9001366935875675, 0.9624253654405275], "spcas9": [0.4529791303025846, 0.8578181067490909, 0.8930069822728043, 0.9186994149041988], "trem2": [0.17595949196247068, 0.30644609763741276, 0.326562595366559, 0.5781060675361671, 0.5915457562171867, 0.6569218539923279, 0.6638762057195294, 0.7259316251487895, 0.7941103165085088, 0.8364567523049665, 0.86675947032288, 0.876284490999763, 0.8776116998748241, 0.8886446849024816, 0.8909591706165636, 0.8943996035357809, 0.9000007588694205, 0.9039696610372472, 0.9091967320727804, 0.9150066404782221, 0.9189293123781431, 0.926021343366693, 0.9395352860408395, 0.9566389961480665]}}, "opendde": {"iptm": {"15-pgdh": [0.9480777382850647], "der21": [0.5389322638511658, 0.7251704335212708, 0.7569423317909241, 0.8391890525817871], "der7": [0.7656484246253967, 0.83867347240448, 0.8409266471862793, 0.8854271173477173, 0.9073126316070557], "egfr": [0.2753937840461731, 0.3533988296985626, 0.355536550283432, 0.40055328607559204, 0.40124890208244324, 0.4505611062049866, 0.5487218499183655, 0.5567421913146973, 0.5728999972343445, 0.595030665397644, 0.6036839485168457, 0.6047351360321045, 0.6250959634780884, 0.634675920009613, 0.73088139295578, 0.7393200993537903, 0.8749021291732788, 0.8953923583030701, 0.9267972111701965, 0.9282987713813782, 0.9389196634292603, 0.9490640759468079, 0.9576218128204346, 0.9588277339935303], "fcrn": [0.58857262134552, 0.62800133228302, 0.7071853876113892, 0.7109612226486206, 0.7400169372558594, 0.8090757131576538], "fgf-r1": [0.24358923733234406, 0.2781227231025696, 0.3174518048763275, 0.3639439642429352, 0.3700077533721924, 0.3809957802295685], "hnmt": [0.9286847114562988], "human-ambp": [0.6111422181129456, 0.6218115091323853, 0.782187819480896, 0.7970499396324158, 0.9110263586044312], "human-calmodulin-1-calm1-holo-cacl2": [0.4887793958187103, 0.6494055390357971, 0.6525881886482239, 0.6961922645568848, 0.7074286937713623], "human-idi2": [0.9456430077552795, 0.954227864742279, 0.9546118378639221, 0.9772261381149292], "human-insulin-receptor": [0.1693546026945114, 0.27458348870277405, 0.2814098596572876], "human-mzb1-perp1": [0.9343925714492798], "human-pdgfr-beta": [0.7467512488365173, 0.8330686688423157, 0.8563613891601562, 0.8722879886627197, 0.8936711549758911, 0.9197763204574585, 0.9331784248352051, 0.9573944211006165], "human-phyh": [0.21318215131759644, 0.9392480254173279], "human-vegf110": [0.19000406563282013, 0.2161257416009903, 0.4737955331802368, 0.5912597179412842, 0.6471306085586548, 0.6574202179908752, 0.6685465574264526, 0.6737204790115356, 0.6830452680587769, 0.691623866558075, 0.6947991847991943, 0.7332410216331482, 0.761708676815033, 0.7664967775344849, 0.7684615254402161, 0.8003392815589905, 0.804959237575531, 0.8088757395744324, 0.8262041211128235, 0.8281883001327515, 0.8289396166801453, 0.8392902612686157, 0.8493270874023438, 0.8516743779182434, 0.8756148815155029], "ifnar2": [0.2434568554162979, 0.31899920105934143, 0.3380581736564636, 0.4797765910625458, 0.5264607071876526, 0.7048774361610413, 0.8501163125038147, 0.9137139916419983], "il7r": [0.7732419967651367, 0.782230794429779, 0.8012262582778931, 0.815675675868988, 0.8218978643417358, 0.8325387239456177, 0.8447789549827576, 0.8627473711967468, 0.8739138245582581, 0.8945390582084656, 0.8969868421554565, 0.8974701762199402, 0.9008969664573669, 0.9094147682189941, 0.9142305254936218, 0.9164173603057861, 0.9240962266921997, 0.93220055103302, 0.9339721202850342, 0.9447658658027649, 0.9646950960159302], "mdm2": [0.5819348692893982, 0.6440386772155762, 0.807251513004303, 0.8479658961296082], "myostatin-gdf8": [0.24538931250572205, 0.6839497089385986, 0.8873175978660583, 0.9105265140533447, 0.9259687662124634, 0.936439037322998], "pd-l1": [0.3303285241127014, 0.4717414975166321, 0.4954351484775543, 0.5911693572998047, 0.7114684581756592, 0.7786371111869812, 0.78190678358078, 0.7824512720108032, 0.7844328880310059, 0.7900202870368958, 0.7918948531150818, 0.8036282658576965, 0.8129981756210327, 0.8447927832603455, 0.8527486324310303, 0.8702071905136108, 0.8712776899337769, 0.8909090161323547, 0.9056272506713867, 0.9299705624580383, 0.9353295564651489, 0.9505859017372131, 0.9688830971717834, 0.9695730209350586], "rbx1": [0.19396035373210907, 0.3927568197250366, 0.3963784873485565, 0.6772715449333191, 0.7047101855278015, 0.7248245477676392, 0.8505341410636902, 0.912615180015564, 0.9251074194908142], "spcas9": [0.22548145055770874, 0.7149220108985901, 0.795561671257019, 0.8751302361488342], "trem2": [0.2964932918548584, 0.4771979749202728, 0.48310598731040955, 0.5069749355316162, 0.6025654673576355, 0.640104353427887, 0.7587363123893738, 0.7683220505714417, 0.8113309144973755, 0.8407289981842041, 0.8438112139701843, 0.869927704334259, 0.8797439932823181, 0.8888465762138367, 0.8952751755714417, 0.897375762462616, 0.9065521359443665, 0.9276063442230225, 0.9309000968933105, 0.9400808811187744, 0.9418232440948486, 0.9422726631164551, 0.9447385668754578, 0.9551271200180054]}, "ipsae": {"15-pgdh": [0.9153370420377077], "der21": [0.13758050613480563, 0.4243320564222547, 0.5610590919848821, 0.6591021366053473], "der7": [0.5120126623207556, 0.622581904577845, 0.656034981907702, 0.7253117755575483, 0.8124548367172915], "egfr": [0.0, 0.014376409802014437, 0.015234241842745066, 0.017664619489507715, 0.01881246305812771, 0.019561336858349317, 0.019690207036550073, 0.020037067160774545, 0.020416361008126552, 0.023532176418206034, 0.02606989297809718, 0.029627237175868164, 0.03367038539889221, 0.03412689532830102, 0.33527931345002654, 0.40880329959792244, 0.6313293914732314, 0.6652141803169532, 0.72151189763424, 0.7218974853104941, 0.7432477812419442, 0.7711077700176886, 0.8437033526928106, 0.8464956070783065], "fcrn": [0.017059657918238897, 0.028485482351317627, 0.04593873018057785, 0.07030058498099064, 0.08508376810885002, 0.35465072056198826], "fgf-r1": [0.0, 0.0, 0.010602700202901104, 0.011880096856033924, 0.028581348629442328, 0.06606067043240853], "hnmt": [0.8398575070941688], "human-ambp": [0.21715527256302156, 0.341911112085456, 0.5930104946225321, 0.6340758391006668, 0.8372291891156143], "human-calmodulin-1-calm1-holo-cacl2": [0.02241516489208304, 0.036145051545430754, 0.04006309941807997, 0.1505970121062334, 0.31367426698009787], "human-idi2": [0.885964398306717, 0.8936772486938492, 0.9064033018795489, 0.9326326389751213], "human-insulin-receptor": [0.0, 0.0, 0.010690111043931532], "human-mzb1-perp1": [0.8389667444750464], "human-pdgfr-beta": [0.24444189045253456, 0.49452220113879963, 0.5258336333631332, 0.6082025018723154, 0.630629908918034, 0.6841158048133448, 0.7297541990257368, 0.8360621128918909], "human-phyh": [0.0, 0.8374604244136662], "human-vegf110": [0.010079822418575963, 0.01318869843434127, 0.0434154740891883, 0.16813391860475102, 0.197101027923298, 0.21858018087747474, 0.22137391447625338, 0.23530415353716508, 0.2357749266973018, 0.2531447996068758, 0.26452908738748276, 0.3162170147436711, 0.32800424411419216, 0.3952509607172836, 0.4063623103868439, 0.4300609596167282, 0.45416308593695726, 0.4542163892446081, 0.4678881052217568, 0.5055269600320587, 0.5160301466924836, 0.523478353672568, 0.5370934546839916, 0.5503948681095272, 0.666599785115916], "ifnar2": [0.014007150921536428, 0.01675726828339504, 0.021322767397411783, 0.04078707533791869, 0.16097520415537928, 0.40584989088706874, 0.7012659361036229, 0.7579114192858079], "il7r": [0.4280368115418131, 0.45501967179210207, 0.5260455691804237, 0.5271986882550584, 0.5376153712119097, 0.5978926629080175, 0.5997845782195874, 0.6430839090777163, 0.6940826931777915, 0.7155845687046843, 0.7350065742900092, 0.7393224408308625, 0.7427094737138448, 0.7533191829219, 0.7847775102476525, 0.7959613219889774, 0.8235297999033195, 0.8305036209145866, 0.8354901889910901, 0.844875444600691, 0.9039563910799944], "mdm2": [0.26372035426691504, 0.3270057555288166, 0.7108105061180037, 0.8105050231761873], "myostatin-gdf8": [0.0, 0.17646405013978206, 0.6248225194258675, 0.6751227313219005, 0.7332368944719002, 0.7467468627803063], "pd-l1": [0.02074822307295456, 0.02227694643448168, 0.14313810617156544, 0.16333922121060512, 0.30813458271619576, 0.4237040167904223, 0.45312422815231346, 0.46728665012046094, 0.4740554598401724, 0.4964797731646421, 0.5203104114413369, 0.5332740717023042, 0.5431745979945003, 0.575632415336581, 0.6162299779577479, 0.6177889770793068, 0.6201138796839237, 0.7122428497319864, 0.7342423488223296, 0.7857728787225983, 0.8177153346470446, 0.8804704538260727, 0.9113778014220243, 0.920922595258112], "rbx1": [0.0, 0.0549579136763636, 0.0819664488818397, 0.3009818599359333, 0.47921155743725213, 0.6008679842819218, 0.7279203922755014, 0.816609119145649, 0.8901531696603076], "spcas9": [0.0, 0.1400992659137402, 0.44154604732603553, 0.48501644383064], "trem2": [0.019256166282557344, 0.027804001457303152, 0.051433338327904685, 0.09843887052125527, 0.14331892823410505, 0.20043936897355338, 0.3049195673566343, 0.49455536707305076, 0.5666920489188692, 0.5716220989007208, 0.604396985422408, 0.6187107708045367, 0.6629455483418323, 0.7064283542662838, 0.7386177315730269, 0.7908150046349087, 0.7915203276829725, 0.8225251530258587, 0.8476440076275268, 0.85247569137764, 0.8551409757936186, 0.8677402081078152, 0.8922500886484127, 0.8960277604373511]}, "ipdae": {"15-pgdh": [0.7677543108886149], "der21": [0.11170189469381944, 0.3790175969375673, 0.4295362848244941, 0.6213671301981668], "der7": [0.3790175969375673, 0.43057051608649394, 0.544353989224152, 0.6503825850901819, 0.7352941021787254], "egfr": [0.0032329221094195043, 0.005040258933158492, 0.005471281522320314, 0.005504263398227414, 0.006070001798285395, 0.006570029432570829, 0.006908591304867293, 0.006930924021772055, 0.00965093510444341, 0.011363040883528623, 0.01241212196531134, 0.012777250470761536, 0.013158657011405811, 0.017376951074265068, 0.16867957266256187, 0.31054935908025966, 0.49060750436809125, 0.5691843561258446, 0.5942423015046462, 0.6018952459225436, 0.6038282698314021, 0.7223345822948526, 0.7260878930515339, 0.7288500454115716], "fcrn": [0.009182449515213261, 0.009448549259111347, 0.01003966611232661, 0.059946287040139226, 0.06606067043240853, 0.3020783039860822], "fgf-r1": [0.002496214463029028, 0.0025264204629489233, 0.003267248307789244, 0.010602700202901104, 0.017934002415481844, 0.12464941564303862], "hnmt": [0.775966526176647], "human-ambp": [0.14181178362135172, 0.16494845360824742, 0.28090035942284053, 0.4708985011882928, 0.6256067261916819], "human-calmodulin-1-calm1-holo-cacl2": [0.01032587434559853, 0.021983384465796348, 0.026508605180418483, 0.09592418529849583, 0.1003751435707708], "human-idi2": [0.7912434923725408, 0.819067901480728, 0.847069415890331, 0.8715128935313459], "human-insulin-receptor": [0.0019696666983206063, 0.00231049104884643, 0.002845449083397904], "human-mzb1-perp1": [0.8228193084718164], "human-pdgfr-beta": [0.15898251746773207, 0.4524886770434498, 0.4708985011882928, 0.5100999698686011, 0.520399678101481, 0.6901787456450239, 0.726535117174524, 0.7612667463209489], "human-phyh": [0.004679901972574192, 0.6625124921549094], "human-vegf110": [0.0044721438644809205, 0.004661604026439564, 0.0639751842938867, 0.12350969616699975, 0.1458491258012015, 0.15751424846486226, 0.17516508294068664, 0.1965918332855002, 0.23786304618920648, 0.24024336839046423, 0.2783886827139361, 0.3410524926036118, 0.361154234090254, 0.3717472290291268, 0.37536129910392035, 0.3967705848235308, 0.520399678101481, 0.5580045835659443, 0.6277858110616225, 0.6338742451752952, 0.64, 0.64, 0.64, 0.664849425076718, 0.7223345822948526], "ifnar2": [0.011478632032828052, 0.018271882322439117, 0.04382954717658864, 0.05666878430046006, 0.2834386695146678, 0.4138901302241209, 0.5635707871560383, 0.6038282698314021], "il7r": [0.27345566032728225, 0.31976752257995833, 0.3472708434287195, 0.4570174885722599, 0.5308982811529142, 0.5580045835659443, 0.5635707871560383, 0.5863039551081187, 0.5921013774668232, 0.6080209803845168, 0.6411400615559235, 0.664849425076718, 0.668327633263208, 0.6838074352764484, 0.7068125048173308, 0.7547739498485644, 0.7742334949347254, 0.8, 0.8127438299504254, 0.8147031336942302, 0.826611608241439], "mdm2": [0.10696456070971394, 0.1548329884449263, 0.7029877130324729, 0.7807010880444991], "myostatin-gdf8": [0.0038959001049033034, 0.13462145473211726, 0.4877302690182708, 0.6277858110616225, 0.6461617930213407, 0.6585879694856501], "pd-l1": [0.019986582615877782, 0.031225700635656946, 0.047662401965584356, 0.05130835894499127, 0.14374287006637199, 0.29397311774178353, 0.3020783039860822, 0.3681871816808862, 0.3864584979469778, 0.39209515811638873, 0.4425674963186659, 0.44800859539108107, 0.4980700007298761, 0.5, 0.5253696580868314, 0.5308982811529142, 0.591287061409647, 0.6217359038481038, 0.7417847510984446, 0.7612667463209489, 0.7903707406506149, 0.8063185901409186, 0.8127272385653395, 0.8408773409397434], "rbx1": [0.004976619599527252, 0.014212438424494205, 0.037870467723277824, 0.23385796600661468, 0.4640564782665568, 0.5805515102871259, 0.7547739498485644, 0.7612667463209489, 0.8677667810727039], "spcas9": [0.0019006755576214784, 0.132387211235082, 0.3940731363019104, 0.4756242794896638], "trem2": [0.02633090156842941, 0.028203256288398038, 0.0297020619459182, 0.057143178354788486, 0.09355703914068929, 0.13234285455812853, 0.2169197439593763, 0.2888546164631829, 0.3076923076923077, 0.4524886770434498, 0.4524886770434498, 0.5371046750236159, 0.5863039551081187, 0.6711409471146798, 0.6774608781444115, 0.6935059035455489, 0.7223345822948526, 0.72242590697977, 0.7807010880444991, 0.8127438299504254, 0.8127438299504254, 0.8127438299504254, 0.819067901480728, 0.8313974204849285]}, "actifptm": {"15-pgdh": [0.9863409130366813], "der21": [0.7631858646799231, 0.9043238095380745, 0.9531056159044217, 0.9665699961677511], "der7": [0.9389877003917493, 0.9570805642084057, 0.9612118703095163, 0.9816178371660513, 0.9837651800486901], "egfr": [0.3465964046509882, 0.37117565196267405, 0.5024621662323661, 0.5115638509755305, 0.5357656236733075, 0.5482308128621949, 0.5831207514086216, 0.588960900361927, 0.6205549932732368, 0.6328429429172101, 0.6337858177774744, 0.6400786916780753, 0.6479284764556112, 0.7394940570691713, 0.9167879232974878, 0.9223016712311043, 0.9794093222851606, 0.9806445110838277, 0.9837003363232293, 0.9849610995839574, 0.9885678284500737, 0.9894779044177734, 0.9917184458479107, 0.9920044335075141], "fcrn": [0.5950310138447548, 0.6957224636718211, 0.8235377392720735, 0.8522521543904823, 0.8646689357863028, 0.9043611903916281], "fgf-r1": [0.3156057078425937, 0.32061158280060675, 0.3725685443329835, 0.4428360279280974, 0.576278598173597, 0.8275714944252509], "hnmt": [0.9886090965786901], "human-ambp": [0.8608324242405582, 0.9494879703035918, 0.9665429058291856, 0.967483212259283, 0.9749570538511307], "human-calmodulin-1-calm1-holo-cacl2": [0.5613700369913464, 0.6762352360321224, 0.6927418480634375, 0.81244541703859, 0.8208855588970152], "human-idi2": [0.9866236603733872, 0.9909799625073171, 0.9915577067398285, 0.994244458133225], "human-insulin-receptor": [0.2897256917217215, 0.30305497519310526, 0.44720077420055176], "human-mzb1-perp1": [0.9853451548950224], "human-pdgfr-beta": [0.8832691527943244, 0.949801695123651, 0.9661220714206921, 0.9716702358268084, 0.9739440272645034, 0.9748416471074292, 0.983637234844688, 0.990396410062878], "human-phyh": [0.25537873222643476, 0.990291366207446], "human-vegf110": [0.3252100726044956, 0.3305238113704974, 0.7920077059840775, 0.8345534938976902, 0.8592636799641653, 0.8664983181856734, 0.893091939737571, 0.8968927055973059, 0.8980437773224538, 0.9062335728454828, 0.9137833915695188, 0.9205873254149282, 0.9216041235014395, 0.9260208083134185, 0.9279317747793118, 0.9376393265718823, 0.9417128981455996, 0.9491219841910818, 0.9529748181103521, 0.9570130193026312, 0.957531129563746, 0.9621110028408906, 0.9629687091600195, 0.9639461170550707, 0.9726670139383948], "ifnar2": [0.4403375475450788, 0.503182037539787, 0.6082219640517783, 0.7527485707158292, 0.7628233851611755, 0.9283771197148726, 0.959978657616476, 0.9764206598292575], "il7r": [0.9284789325848418, 0.9364051897928771, 0.9421151278491202, 0.943142430067841, 0.9531423998886075, 0.9579119256390976, 0.9583692949307554, 0.9599093302838424, 0.9675152896682965, 0.9729180349422912, 0.9734384810707674, 0.9773522846523582, 0.9794987721281924, 0.981164012382965, 0.9852975948198206, 0.9855228342014791, 0.9857331125926592, 0.988665551155969, 0.989720676358852, 0.9900353268302517, 0.9909810859765609], "mdm2": [0.7610336946666156, 0.8989734727986611, 0.9685368933233038, 0.9747882210503387], "myostatin-gdf8": [0.3239740572289336, 0.8559892855223399, 0.968569182091314, 0.97813518594533, 0.982007679460389, 0.9827027117951623], "pd-l1": [0.5731197915855233, 0.6790977820387563, 0.7964227387169938, 0.8097656645690615, 0.880523129011333, 0.9275830734041405, 0.9299457669092497, 0.9350805651831114, 0.9368472310841653, 0.9401041568946382, 0.9412099331059272, 0.9472707495797869, 0.949872623371979, 0.9508900941212622, 0.9572528770302605, 0.9605662026421518, 0.9648140858744849, 0.9785149351128423, 0.9807561045745721, 0.9878633768006677, 0.987986271204336, 0.9906408698334065, 0.9910138045230912, 0.9913126824949148], "rbx1": [0.2691454288240184, 0.6242023844808771, 0.6585865626314982, 0.8825448309753103, 0.9505807001994427, 0.9556868430113337, 0.9740523617122399, 0.9836930712690811, 0.9903892234934024], "spcas9": [0.23462158062185087, 0.8969661564595877, 0.926503554217849, 0.9716345712243831], "trem2": [0.42292913981556596, 0.5910172894983515, 0.655798554195587, 0.7081902115643663, 0.7860026098477344, 0.792950642035681, 0.867728071985477, 0.9157362498215113, 0.9263244906849649, 0.9505440384200597, 0.9526936886533225, 0.9539273426579663, 0.9569704559228901, 0.9727353700452765, 0.9738800316123668, 0.9750474121466162, 0.9782417861134906, 0.981597962588836, 0.9820542531896137, 0.9825484174987918, 0.9863971125108266, 0.9873220116570451, 0.9900251094580774, 0.9904239170660265]}}, "mol1": {"iptm": {"15-pgdh": [0.9157935976982117], "der21": [0.6100349426269531, 0.742516040802002, 0.7487025856971741, 0.9089864492416382], "der7": [0.16884584724903107, 0.7420479655265808, 0.7755333185195923, 0.9021165370941162, 0.9260873794555664], "egfr": [0.12941117584705353, 0.14308276772499084, 0.1491856873035431, 0.14930371940135956, 0.17933449149131775, 0.18680042028427124, 0.20604291558265686, 0.24141395092010498, 0.24586032330989838, 0.24855755269527435, 0.2569362223148346, 0.2808167636394501, 0.29737207293510437, 0.42768728733062744, 0.48638609051704407, 0.605333149433136, 0.6243802309036255, 0.7304665446281433, 0.7963192462921143, 0.8363891839981079, 0.8673756122589111, 0.9225693941116333, 0.9292845129966736, 0.9476684331893921], "fcrn": [0.5506709814071655, 0.6606577634811401, 0.7098397016525269, 0.7365361452102661, 0.7882226705551147, 0.8033267855644226], "fgf-r1": [0.249017134308815, 0.29044854640960693, 0.30756881833076477, 0.3657066226005554, 0.3782076835632324, 0.42462050914764404], "hnmt": [0.8819519281387329], "human-ambp": [0.5100571513175964, 0.5696467757225037, 0.6366201043128967, 0.8417044878005981, 0.872676432132721], "human-calmodulin-1-calm1-holo-cacl2": [0.4594224989414215, 0.4825168251991272, 0.5076850652694702, 0.5396926403045654, 0.8796719908714294], "human-idi2": [0.7334811091423035, 0.9230836629867554, 0.9333794713020325, 0.9606106281280518], "human-insulin-receptor": [0.23712092638015747, 0.5363118648529053, 0.6568778157234192], "human-mzb1-perp1": [0.8751449584960938], "human-pdgfr-beta": [0.8270916938781738, 0.8730558156967163, 0.8911271691322327, 0.8923375606536865, 0.9088242053985596, 0.9195905327796936, 0.9334577322006226, 0.9499350190162659], "human-phyh": [0.9248218536376953, 0.9548214673995972], "human-vegf110": [0.15574784576892853, 0.2832898795604706, 0.2898220717906952, 0.3171449899673462, 0.4422086179256439, 0.4662523567676544, 0.5186269283294678, 0.576400101184845, 0.6221456527709961, 0.6851304769515991, 0.7051497101783752, 0.7431726455688477, 0.7677961587905884, 0.7874197363853455, 0.7994434833526611, 0.8239178657531738, 0.8309556245803833, 0.8620790243148804, 0.8692230582237244, 0.8861308097839355, 0.8907514214515686, 0.9032468199729919, 0.9157604575157166, 0.9402812719345093, 0.9434030055999756], "ifnar2": [0.27117496728897095, 0.30062875151634216, 0.35315749049186707, 0.6306174397468567, 0.8751334547996521, 0.8762522339820862, 0.9042031764984131, 0.9249244928359985], "il7r": [0.5714682340621948, 0.5841127038002014, 0.6505549550056458, 0.7048724293708801, 0.7180435061454773, 0.7637757658958435, 0.806395411491394, 0.8071292638778687, 0.8446446061134338, 0.8647416830062866, 0.8798161149024963, 0.8879919648170471, 0.8950701951980591, 0.9133417010307312, 0.9233735203742981, 0.9308133721351624, 0.9372484087944031, 0.937643826007843, 0.9451189041137695, 0.9464600682258606, 0.9534136056900024], "mdm2": [0.10271885246038437, 0.2352636605501175, 0.5147603154182434, 0.8219320178031921], "myostatin-gdf8": [0.1964689940214157, 0.4076990485191345, 0.8374958634376526, 0.8623265027999878, 0.8686687350273132, 0.8836721777915955], "pd-l1": [0.2875192165374756, 0.31072893738746643, 0.4836411476135254, 0.5531042814254761, 0.6198177933692932, 0.6270698308944702, 0.6292569041252136, 0.6299963593482971, 0.736340343952179, 0.8354685306549072, 0.8450868129730225, 0.8463953733444214, 0.8734449744224548, 0.8797261118888855, 0.907645583152771, 0.9187473654747009, 0.9289703369140625, 0.9366903305053711, 0.9497439861297607, 0.9518658518791199, 0.9529038667678833, 0.9548546671867371, 0.9574074149131775, 0.9605470299720764], "rbx1": [0.1263299435377121, 0.15288680791854858, 0.19945281744003296, 0.23626966774463654, 0.3697251081466675, 0.6309831738471985, 0.7200411558151245, 0.9184035658836365, 0.9202694892883301], "spcas9": [0.22563162446022034, 0.8270437121391296, 0.8600661158561707, 0.9381951093673706], "trem2": [0.19798579812049866, 0.23993131518363953, 0.27607959508895874, 0.3468099534511566, 0.3582000732421875, 0.3670682907104492, 0.38557887077331543, 0.6044842600822449, 0.6149270534515381, 0.6919304132461548, 0.6997565031051636, 0.755466103553772, 0.7947266101837158, 0.8008865118026733, 0.8139746785163879, 0.8448964357376099, 0.8695694804191589, 0.900698184967041, 0.9208245873451233, 0.9289021492004395, 0.9390727877616882, 0.9395312666893005, 0.9519480466842651, 0.9635157585144043]}, "ipsae": {"15-pgdh": [0.7739836066506346], "der21": [0.2884742326845834, 0.468797584356166, 0.4835991114771799, 0.8273337076772068], "der7": [0.0, 0.4169263222346397, 0.5539888564791382, 0.7548801585656519, 0.8763193477906088], "egfr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.013651877133105802, 0.016950343898120068, 0.019833044263279608, 0.08740527434984739, 0.24303131240544834, 0.47561772231274474, 0.5401976180173668, 0.5672090579001113, 0.7028874965817563, 0.7417122939782803, 0.7676250931375184], "fcrn": [0.058281913779999514, 0.11494973008515268, 0.27214008792747024, 0.303833063804571, 0.3942031039045923, 0.4609022439637481], "fgf-r1": [0.0153468141351538, 0.015624738150205916, 0.018736890480092583, 0.026407393515863138, 0.12859125615052794, 0.15806694448523761], "hnmt": [0.726779043635368], "human-ambp": [0.07927359288018533, 0.3283468783285186, 0.3583699276868802, 0.7992974229225169, 0.8314065342938797], "human-calmodulin-1-calm1-holo-cacl2": [0.01690154655653421, 0.030534814845945304, 0.0357518167353213, 0.05851051828197033, 0.6279922849900598], "human-idi2": [0.418254338397943, 0.8535082429669874, 0.8703021524816693, 0.9382454856243967], "human-insulin-receptor": [0.0, 0.013700036004636621, 0.07030561099520184], "human-mzb1-perp1": [0.6912273352917039], "human-pdgfr-beta": [0.5276963567508338, 0.5690691502112816, 0.5946145161392172, 0.6423294325373747, 0.6591769472612992, 0.6804961194372987, 0.7439244133837869, 0.8147011035222778], "human-phyh": [0.7638287788213878, 0.8526798209280957], "human-vegf110": [0.0, 0.0, 0.009979880485106143, 0.012522791161761828, 0.015865261781892428, 0.01713337845800002, 0.04146419098268026, 0.08976542096971771, 0.15817518673779452, 0.17545945538344826, 0.2841198040003541, 0.2895441989880402, 0.3404550302739064, 0.36599200332240855, 0.4327003008187646, 0.4407845028939446, 0.5474535543790957, 0.6199925837288316, 0.6389679044693848, 0.673866790420817, 0.759677610968158, 0.8215548481082279, 0.854429913269306, 0.8603428674120948, 0.9048521709797558], "ifnar2": [0.011097105075319328, 0.011905882511276044, 0.0169249513076175, 0.23410130400812051, 0.7138053673840643, 0.7423380202169232, 0.7720235704054954, 0.8332843192846017], "il7r": [0.06781842924210689, 0.175974079954817, 0.2616230307579037, 0.3384095578149935, 0.3428084886045326, 0.3605217163253629, 0.4614816307956361, 0.4809957102391305, 0.575336392572607, 0.5922388926837815, 0.6496278148161928, 0.6916516329932562, 0.7225998087345352, 0.7591383902846848, 0.8128920468135181, 0.826327040662569, 0.8293469050418255, 0.8416584373985976, 0.8484775009983277, 0.8523295156422208, 0.8800462005788549], "mdm2": [0.0, 0.0, 0.1940588487984396, 0.8061107064898456], "myostatin-gdf8": [0.0, 0.014139770369005092, 0.46430234290949013, 0.5445693258940764, 0.6168572652995943, 0.640826721101738], "pd-l1": [0.0102635269326837, 0.016800369196290757, 0.06002123871057472, 0.1596804188525453, 0.20331863477521156, 0.20438337972403128, 0.22300873643407004, 0.29528684931728094, 0.3748441751711234, 0.5512007268931791, 0.552729441402765, 0.5822055456203102, 0.6477685920949658, 0.6553475736988017, 0.7475090673823561, 0.7744425281481856, 0.7792621811380133, 0.8274231158616012, 0.849753072711058, 0.8627214163815066, 0.8728675543433825, 0.9008581912494925, 0.9019739656546426, 0.9040806991537331], "rbx1": [0.0, 0.0, 0.0, 0.017544890431568542, 0.022845863788374814, 0.4312814119058266, 0.5259696958095684, 0.8555893472665664, 0.8832279217528398], "spcas9": [0.0, 0.3543677633792599, 0.4717628101295177, 0.7210492135430878], "trem2": [0.0, 0.0, 0.010868509261401257, 0.01169343400612404, 0.012211753006607961, 0.013098879922612825, 0.013394678445553703, 0.24273537430253744, 0.24654825637134004, 0.2656094272086331, 0.31379151773185293, 0.4564549340550036, 0.4852043315615145, 0.5131006002509677, 0.6156994623054899, 0.6967985344963805, 0.7570172167599183, 0.7710050768634437, 0.8168253661911645, 0.8206203516203834, 0.8865677786382983, 0.8878915821835562, 0.8923132207192608, 0.8974928630891637]}, "ipdae": {"15-pgdh": [0.715870859976418], "der21": [0.15112818419191376, 0.15554761394667965, 0.31853080060419203, 0.6745747530269068], "der7": [0.0045877265562314605, 0.17030449260932323, 0.3717472290291268, 0.6902810343511425, 0.7611672911474263], "egfr": [0.0014306704111824542, 0.0016459272218846263, 0.0016823415200113523, 0.0016972764632520814, 0.001895977687353722, 0.0019368958006369776, 0.001972523575616209, 0.001980188082567339, 0.002099781769375684, 0.0022906519625720536, 0.0023761994693180313, 0.002528034088811787, 0.0025401023762866627, 0.009958462597602802, 0.012771945559063658, 0.021209061886670377, 0.02783336859449715, 0.1131757952633956, 0.12221352613777452, 0.2419608546873215, 0.24828681888963408, 0.46622217710277414, 0.5487915995653682, 0.6128644455373851], "fcrn": [0.03558320287709589, 0.05253054110873882, 0.11607422657211933, 0.11678832116788321, 0.17785681228358996, 0.19683489787565625], "fgf-r1": [0.009341180114029582, 0.016636549404521698, 0.016903371634554992, 0.04219982289182052, 0.11783344539803947, 0.13863280355566893], "hnmt": [0.5285181638923082], "human-ambp": [0.05746300308705036, 0.06911375165330243, 0.132387211235082, 0.4708985011882928, 0.64], "human-calmodulin-1-calm1-holo-cacl2": [0.009999750385793255, 0.010220197730806556, 0.015791508430958466, 0.02397795458291203, 0.334660831799752], "human-idi2": [0.22237541329231725, 0.6901787456450239, 0.7525875302713474, 0.8826671466609498], "human-insulin-receptor": [0.0015697395274314873, 0.008226363870927051, 0.060519497731240574], "human-mzb1-perp1": [0.7994074459226092], "human-pdgfr-beta": [0.17922430273318912, 0.4570174885722599, 0.5313462670950934, 0.5545401083124054, 0.5635707871560383, 0.5979430800131478, 0.6020433877434931, 0.6523582633137126], "human-phyh": [0.7251831838696498, 0.8492571504963357], "human-vegf110": [0.0020769018461442136, 0.005040258933158492, 0.007545726107547966, 0.008377005280355699, 0.013006270497314782, 0.019081170105552932, 0.020169058225829227, 0.028250061569045064, 0.09652075362942826, 0.11384205829082462, 0.12322251104862911, 0.16015118719759705, 0.24023978507408472, 0.2510842310245805, 0.2591602790899311, 0.33783784545713147, 0.4098360463516335, 0.5050249939213871, 0.5308982811529142, 0.6020831882936425, 0.6414383999160604, 0.6838074352764484, 0.715870859976418, 0.819067901480728, 0.8324346530483251], "ifnar2": [0.0054409802288472526, 0.0057496876924183205, 0.02080749656435885, 0.09050674331576543, 0.6348828973709072, 0.6585879694856501, 0.7352941021787254, 0.7673314922820105], "il7r": [0.034828396842473894, 0.07545762648360284, 0.10961908025084273, 0.185275021468315, 0.21596844133188414, 0.2902061372024428, 0.33152103182788845, 0.3530741947405532, 0.37729831633565175, 0.4642041531234931, 0.520399678101481, 0.5246697643566957, 0.5534804097331973, 0.6523582633137126, 0.664849425076718, 0.6972953426744216, 0.7644692078911284, 0.767716316959675, 0.806150963532482, 0.8063865756499511, 0.8084848257102264], "mdm2": [0.002899180843281645, 0.005240044845600054, 0.09163288702131149, 0.6579228062329256], "myostatin-gdf8": [0.003910269675329509, 0.009871353068741325, 0.2570693980264055, 0.3974768217727622, 0.4467734612695134, 0.4950250034831301], "pd-l1": [0.0052386632113474, 0.00806528991656326, 0.015888930478316077, 0.031410396452935085, 0.03268657430477918, 0.057466584852339624, 0.06139451248409878, 0.08136961799912153, 0.12301786833167223, 0.27590773527683393, 0.37536129910392035, 0.40582767036067713, 0.4263301641432068, 0.5635707871560383, 0.5835637180694961, 0.6494633790881511, 0.7251016679559035, 0.7417847510984446, 0.7742334949347254, 0.7787591390359374, 0.7807010880444991, 0.8118381612963986, 0.8251698237388402, 0.8316008390204648], "rbx1": [0.003695267510548261, 0.0041082644133768895, 0.006652102707054373, 0.007440337547426989, 0.016170762481961188, 0.2698729533191655, 0.5413650102812424, 0.7839273763641557, 0.836467153206171], "spcas9": [0.0015916262559849154, 0.19064332949894458, 0.2710320936462045, 0.4535623076431034], "trem2": [0.003046250004110875, 0.0063999344257343795, 0.006902636515718426, 0.008389236948291664, 0.008438419552196841, 0.011478632032828052, 0.011905882511276044, 0.08640680838553558, 0.10626991936437845, 0.10895459778663778, 0.2016088369887073, 0.46622217710277414, 0.5, 0.5050249939213871, 0.5500413186393793, 0.5805515102871259, 0.6217359038481038, 0.748284038614789, 0.7570603572148537, 0.7871536646838122, 0.822172334485445, 0.8222113994295708, 0.8738203453046942, 0.8795848332274618]}, "actifptm": {"15-pgdh": [0.9844159183144117], "der21": [0.8541528181997975, 0.8979770576301245, 0.9047365967466697, 0.9798903957767263], "der7": [0.2351873806261604, 0.8962062884511461, 0.9476183259703936, 0.9834134101398249, 0.9873916849144347], "egfr": [0.13507718344624745, 0.1442900646855934, 0.14689106289997947, 0.15451695469565058, 0.16861819653655358, 0.19577447388765978, 0.20685926811360417, 0.25255360913776737, 0.26111374580459434, 0.285039412226873, 0.2976445086062617, 0.3036445391302982, 0.3192112694180453, 0.5430508751865566, 0.6106993506315725, 0.7377289748511345, 0.7515928828612535, 0.8590909337079375, 0.8975503209638789, 0.9347563021972385, 0.9416688595498189, 0.9809702522103916, 0.9822222008975473, 0.982271374059201], "fcrn": [0.728031273246444, 0.7690842829133643, 0.8777618281476218, 0.9339784048178524, 0.935983057859058, 0.9541057796155259], "fgf-r1": [0.49950349168897656, 0.6321627537382313, 0.6438665520846459, 0.7099171986942828, 0.84990111385424, 0.8642904826023619], "hnmt": [0.973108918920832], "human-ambp": [0.8086690061036441, 0.8097451765839255, 0.9538104808827115, 0.9628115720962619, 0.9738775918594015], "human-calmodulin-1-calm1-holo-cacl2": [0.4752964511411728, 0.5767303767354325, 0.5785385667830449, 0.6525663795931289, 0.9465757418906957], "human-idi2": [0.8771525137989131, 0.9830039919675032, 0.9878765697677799, 0.9941235841960321], "human-insulin-receptor": [0.25637640832909103, 0.6041285962868612, 0.8090250684359983], "human-mzb1-perp1": [0.9755103999607359], "human-pdgfr-beta": [0.9372797426859251, 0.9666334979460813, 0.9685397795045779, 0.9705015289172898, 0.9779094044902411, 0.9779818344731163, 0.9795839261591024, 0.9842617235275165], "human-phyh": [0.9847726334837744, 0.9954377724180477], "human-vegf110": [0.19529547152303897, 0.38250149200956163, 0.3852145543420169, 0.38541004310559623, 0.5534969113462771, 0.557142002184509, 0.6059744662777291, 0.7243518974231733, 0.7612724945350542, 0.7686367423981103, 0.8082913847931308, 0.8181590947030678, 0.8777693887775099, 0.8871809169802963, 0.9075158020543536, 0.9118902712153781, 0.919553301462092, 0.9414285156358105, 0.9464888347759889, 0.9516129559876014, 0.9602024077885744, 0.96450568380923, 0.9749867672738584, 0.9816943079472906, 0.9855302386743121], "ifnar2": [0.3819783012758504, 0.40166822357422394, 0.5738767641374733, 0.8224181313922095, 0.9686455468240703, 0.9765269360389363, 0.9819925729241603, 0.9876223274241342], "il7r": [0.6838870692769426, 0.7836189650818829, 0.8058777329313835, 0.8574247316217279, 0.9074186803861838, 0.9324559607137916, 0.9400194015011444, 0.9443020355537898, 0.950649367671909, 0.9584122230437015, 0.9591569443145854, 0.9658680800450042, 0.9708460406772532, 0.9782923650434855, 0.9808231382772741, 0.9866382712334062, 0.9880691112574188, 0.9885026381237786, 0.9907508290384494, 0.9907761276735672, 0.991158051632275], "mdm2": [0.11971504968379618, 0.35252331923093466, 0.7848005104398288, 0.9818586589937331], "myostatin-gdf8": [0.31679708322155176, 0.5133148507800216, 0.9276050410650222, 0.9495121437353053, 0.9587310125706416, 0.9627321818339982], "pd-l1": [0.33935533990102956, 0.40507600265431826, 0.6380566363638815, 0.7154089075722866, 0.7239985208827195, 0.752851353151716, 0.7806969468264818, 0.8488511560441807, 0.8639428098497587, 0.9072675665027805, 0.9323555554474683, 0.9509987518463365, 0.9566784725538788, 0.965791942033978, 0.9702708515046874, 0.9724256795987283, 0.9845119392326516, 0.9850412482492065, 0.9877343452446452, 0.9890125695559479, 0.9893671153900976, 0.9896223602406495, 0.9896338220990956, 0.990292819293445], "rbx1": [0.14138948861850914, 0.24059300290618407, 0.3150751904869768, 0.424318889926949, 0.5354896229984798, 0.8999093584708361, 0.9599050151191064, 0.9858778170309828, 0.989807061452297], "spcas9": [0.23758130232052083, 0.9173342110559763, 0.9437095937975211, 0.9792248169809282], "trem2": [0.22956512166785062, 0.2678339766377179, 0.30801013518736564, 0.38714027040230775, 0.39088486623357405, 0.4042323057133847, 0.40643424254216876, 0.7649527338688016, 0.7959464322628166, 0.8121915047715187, 0.830893437280576, 0.9115688512711799, 0.920735298992891, 0.9550522468431255, 0.9571238704038526, 0.9654821773472433, 0.9750854879378272, 0.9800818552586945, 0.9808808122062496, 0.9847918874108426, 0.9861804730463797, 0.9900642986756045, 0.9905821265083289, 0.992367278167987]}}}}'
)


if __name__ == "__main__":
    main()
