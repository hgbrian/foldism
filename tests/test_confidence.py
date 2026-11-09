"""CPU-only tests for interface confidence, percentiles, native adapters and result plumbing."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import foldism
import numpy as np
import pytest
from backends import confidence as package_confidence
from backends.common import get_cache_key
from backends.confidence import (
    METRICS,
    REFERENCE,
    SCORE_VERSION,
    annotate_percentiles,
    cache_supports,
    callbacks,
    target_percentile,
)


# Interface confidence: reference equations, pair enumeration and missing data.

PATH = Path(__file__).resolve().parents[1] / "backends/confidence.py"
spec = importlib.util.spec_from_file_location("confidence", PATH)
assert spec is not None and spec.loader is not None
confidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confidence)


def test_standalone_copy_in_modal_images_loads_on_its_own():
    assert confidence.REFERENCE == REFERENCE


@pytest.mark.parametrize("partners", [1, 10, 26])
def test_ipsae_reference_small_interface_floor(partners):
    assert confidence.ipsae_direction(np.ones((2, partners))) == 0.5


def test_ipsae_cutoff_is_strict():
    assert confidence.ipsae_direction(np.full((2, 2), 10.0)) == 0


def test_all_three_pairs_and_native_directional_maxima():
    result = confidence.calculate(
        np.ones((6, 6)),
        np.zeros((6, 3)),
        np.array(list("AABBCC")),
        np.array([[1, 0.1, 0.2], [0.3, 1, 0.4], [0.5, 0.6, 1]]),
    )
    assert result["pairs"] == [["A", "B"], ["A", "C"], ["B", "C"]]
    assert result["iptm"] == [0.3, 0.5, 0.6]
    assert result["ipsae"] == [0.5, 0.5, 0.5]
    assert result["ipdae"] == [0.5, 0.5, 0.5]
    assert result["actifptm"] == [None] * 3
    assert result["status"]["actifptm"] == ["not_requested"] * 3


def test_ipdae_uses_forward_transpose_not_native_reverse():
    pae = np.array([[0, 9], [1, 0.0]])
    result = confidence.calculate(pae, np.zeros((2, 3)), np.array(["A", "B"]))
    assert result["ipsae"] == [0.5]
    assert result["ipdae"] == [1 / 82]


def test_missing_native_iptm_and_absent_contacts_are_not_zero():
    result = confidence.calculate(
        np.ones((2, 2)), np.array([[0, 0, 0], [100, 0, 0]]), np.array(["A", "B"])
    )
    assert result["iptm"] == [None]
    assert result["ipdae"] == [None]
    assert result["status"]["ipdae"] == ["no_ca_contacts_within_8_angstrom"]


def test_nonfinite_native_score_rejected():
    with pytest.raises(ValueError, match="Invalid iptm"):
        confidence.calculate(
            np.ones((2, 2)), np.zeros((2, 3)), np.array(["A", "B"]), [float("nan")]
        )


def test_actifptm_multichain_uses_full_complex_length():
    chains = np.repeat([1, 2, 3], 10)
    logits = np.zeros((30, 30, 2))
    centers = np.array([1.0, 5.0])
    contacts = np.ones((30, 30))
    values = confidence.actifptm_pairs(logits, centers, contacts, chains)
    scale = 1.24 * (30 - 15) ** (1 / 3) - 1.8
    expected = np.mean(1 / (1 + (centers / scale) ** 2)) * 10 / (10 + 1e-8)
    assert values == pytest.approx({"1:2": expected, "1:3": expected, "2:3": expected})
    cropped = confidence.actifptm_pairs(
        logits[:20, :20], centers, contacts[:20, :20], chains[:20]
    )
    assert values["1:2"] != pytest.approx(cropped["1:2"])


def test_download_filters_large_distributions():
    assert confidence.downloadable(Path("sample.cif"))
    assert confidence.downloadable(Path("sample_summary_confidence_sample_0.json"))
    for name in [
        "sample_full_data_sample_0.json",
        "sample.raw.npz",
        "pae_logits.npz",
        "distogram.npy",
    ]:
        assert not confidence.downloadable(Path(name))


def test_native_dictionary_order_does_not_change_chain_pairs():
    native = {"9": {"5": 0.2, "9": 1}, "5": {"9": 0.8, "5": 1}}
    assert confidence.native_pair_value(native, 0, 1, 2, [5, 9]) == 0.8
    assert confidence.native_pair_value(native, 0, 1, 2, [1, 2]) is None


def test_requested_actifptm_must_not_silently_be_missing():
    with pytest.raises(ValueError, match="native confidence capture is missing"):
        confidence.calculate(
            np.ones((2, 2)),
            np.zeros((2, 3)),
            np.array(["A", "B"]),
            calculate_actifptm=True,
        )


def test_actifptm_accepts_only_roundoff_outside_probability_bounds():
    logits = np.zeros((2, 2, 2))
    centers = np.array([0.5, 1.5])
    chains = np.array([0, 1])
    exact = confidence.actifptm_pairs(logits, centers, np.ones((2, 2)), chains)
    rounded = confidence.actifptm_pairs(
        logits, centers, np.full((2, 2), 1 + 1e-7), chains
    )
    assert exact == rounded
    with pytest.raises(ValueError, match="Invalid contact probabilities"):
        confidence.actifptm_pairs(logits, centers, np.full((2, 2), 1.01), chains)


# Percentiles against the ProteinBase non-binder reference.

# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///


def scores() -> dict:
    """Create a supported two-chain score bundle."""
    return {
        "version": REFERENCE["metadata"]["score_version"],
        "pairs": [["A", "B"]],
        **{metric: [0.5] for metric in METRICS},
        "status": {metric: ["complete"] for metric in METRICS},
    }


def test_targets_have_equal_weight_and_ties_have_half_weight():
    groups = {"large": [0.1] * 99, "small": [0.9]}
    assert target_percentile(0.5, groups) == 50
    assert target_percentile(0.1, groups) == 25
    assert target_percentile(0, groups) == 0
    assert target_percentile(1, groups) == 100
    assert target_percentile(0, {"ties": [0, 0, 0]}) == 50


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_query_scores_rejected(value):
    with pytest.raises(ValueError):
        target_percentile(value, {"target": [0.5]})


def test_reference_has_all_models_metrics_and_targets():
    reference = REFERENCE["metadata"]
    assert len(set(reference["case_ids"])) == reference["nonbinders"] == 200
    assert len(reference["target_counts"]) == reference["targets"] == 23
    assert set(REFERENCE["distributions"]) == {"esmfold2", "opendde", "mol1"}
    for metrics in REFERENCE["distributions"].values():
        assert set(metrics) == set(METRICS)
        for groups in metrics.values():
            assert {
                target: len(values) for target, values in groups.items()
            } == reference["target_counts"]
            assert all(values == sorted(values) for values in groups.values())


@pytest.mark.parametrize("method", ["esmfold2", "opendde", "mol1"])
def test_annotation_preserves_raw_scores_and_is_idempotent(method):
    raw = scores()
    before = deepcopy(raw)
    result = annotate_percentiles(method, raw, use_msa=True)
    assert raw == before
    assert result["percentile_reference"]["status"] == "available"
    for metric in METRICS:
        assert result[metric] == raw[metric]
        assert 0 <= result["percentiles"][metric][0] <= 100
    assert annotate_percentiles(method, result, use_msa=True) == result


@pytest.mark.parametrize(
    ("method", "msa", "version", "pairs", "reason"),
    [
        ("alphafold2", True, None, None, "no_model_reference"),
        ("esmfold2", False, None, None, "requires_msa"),
        ("mol1", True, "old", None, "score_version_mismatch"),
        ("opendde", True, None, [], "requires_two_protein_chains"),
        (
            "mol1",
            True,
            None,
            [["A", "B"], ["A", "C"], ["B", "C"]],
            "requires_two_protein_chains",
        ),
    ],
)
def test_unsupported_context_has_no_percentile(method, msa, version, pairs, reason):
    raw = scores()
    if version is not None:
        raw["version"] = version
    if pairs is not None:
        raw["pairs"] = pairs
    result = annotate_percentiles(method, raw, use_msa=msa)
    assert result["percentile_reference"]["status"] == reason
    assert all(
        value is None for values in result["percentiles"].values() for value in values
    )


def test_no_contacts_rank_as_zero_but_missing_native_scores_remain_unavailable():
    raw = scores()
    raw["ipdae"] = [None]
    raw["status"]["ipdae"] = ["no_ca_contacts_within_8_angstrom"]
    raw["iptm"] = [None]
    raw["status"]["iptm"] = ["missing_native_input"]
    result = annotate_percentiles("esmfold2", raw, use_msa=True)
    assert result["ipdae"] == [None]
    assert result["percentiles"]["ipdae"][0] is not None
    assert result["percentile_status"]["ipdae"] == ["no_contacts_ranked_as_zero"]
    assert result["percentiles"]["iptm"] == [None]


def test_disabling_msa_clears_previously_attached_percentiles():
    annotated = annotate_percentiles("mol1", scores(), use_msa=True)
    result = annotate_percentiles("mol1", annotated, use_msa=False)
    assert all(value == [None] for value in result["percentiles"].values())


# Adapters, cache capabilities and result plumbing.

@pytest.mark.parametrize(
    "model", ["esmfold2", "opendde", "mol1", "alphafold2"]
)
def test_actifptm_and_array_retention_default_cache_identity(model):
    off = foldism._build_method_params(model, ">A\nAAA\n>B\nGGG", False, download_confidence_data=False)
    on = foldism._build_method_params(
        model, ">A\nAAA\n>B\nGGG", False
    )
    assert off["calculate_actifptm"] is True
    assert off["download_confidence_data"] is False
    assert on["download_confidence_data"] is True
    assert on["calculate_actifptm"] is True
    assert get_cache_key(model, off) != get_cache_key(model, on)


def test_cache_rejects_missing_scores_and_incomplete_opt_in(tmp_path):
    path = tmp_path / "sample_scores.json"
    path.write_text("{}")
    assert not cache_supports(tmp_path, False)
    scores = {
        "version": SCORE_VERSION,
        "actifptm_requested": False,
        "status": {"actifptm": ["not_requested"]},
    }
    path.write_text(json.dumps({"interface_confidence": scores}))
    assert not cache_supports(tmp_path, False)
    scores.update(
        pairs=[["A", "B"]],
        iptm=[0.6],
        ipsae=[0.5],
        ipdae=[0.4],
        actifptm=[None],
        status={
            "iptm": ["complete"],
            "ipsae": ["complete"],
            "ipdae": ["complete"],
            "actifptm": ["not_requested"],
        },
    )
    path.write_text(json.dumps({"interface_confidence": scores}))
    assert cache_supports(tmp_path, False)
    assert not cache_supports(tmp_path, True)
    scores.update(
        actifptm_requested=True, status={"actifptm": ["missing_native_input"]}
    )
    path.write_text(json.dumps({"interface_confidence": scores}))
    assert not cache_supports(tmp_path, True)


@pytest.mark.parametrize("model", ["opendde", "mol1"])
def test_native_best_sample_preserves_interface_schema(model):
    scores = {
        "ranking_score": 0.5,
        "interface_confidence": {"pairs": [["A", "B"], ["A", "C"], ["B", "C"]]},
    }
    outputs = [
        (Path("sample_sample_0.cif"), b"data_fixture"),
        (Path("sample_summary_confidence_sample_0.json"), json.dumps(scores).encode()),
    ]
    result = foldism._process_outputs_to_files(model, outputs)
    assert result is not None
    enriched = json.loads(result["scores"])
    assert enriched["ranking_score"] == scores["ranking_score"]
    assert enriched["interface_confidence"]["pairs"] == scores["interface_confidence"]["pairs"]
    assert json.loads(result["unified"])["interface_confidence"] == enriched["interface_confidence"]
    assert all(value is None for values in enriched["interface_confidence"]["percentiles"].values() for value in values)


def test_native_confidence_hook_preserves_outputs_and_chain_keys(monkeypatch):
    calls = []

    def original(
        configs,
        token_asym_id,
        atom_to_token_idx,
        atom_is_polymer,
        pae_logits,
        contact_probs,
    ):
        calls.append(True)
        return [{"native": 7}], [{"native_full": 8}]

    module = SimpleNamespace(
        compute_full_data_and_summary=original,
        get_bin_params=lambda bins: {},
        get_bin_centers=lambda: np.array([1.0, 5.0]),
    )
    package_confidence.install_native(module, "mol1")
    values = {
        "configs": SimpleNamespace(loss=SimpleNamespace(pae=None)),
        "token_asym_id": np.array([1, 1, 2, 2, 3, 3]),
        "atom_to_token_idx": np.arange(6),
        "atom_is_polymer": np.ones(6),
        "pae_logits": np.zeros((1, 6, 6, 2)),
        "contact_probs": np.ones((6, 6)),
    }
    monkeypatch.setenv("FOLDISM_ACTIFPTM", "0")
    summaries, full = module.compute_full_data_and_summary(**values)
    assert summaries == [{"native": 7}] and full == [{"native_full": 8}]
    monkeypatch.setenv("FOLDISM_ACTIFPTM", "1")
    summaries, full = module.compute_full_data_and_summary(**values)
    assert set(summaries[0]["foldism_actifptm"]) == {"1:2", "1:3", "2:3"}
    assert summaries[0]["native"] == 7 and full == [{"native_full": 8}]
    assert len(calls) == 2


def test_native_hook_replaces_preexisting_from_import_alias(monkeypatch):
    import sys
    from types import ModuleType

    def original():
        return [], []

    module = ModuleType("mol.model.sample_confidence")
    vars(module)["compute_full_data_and_summary"] = original
    consumer = ModuleType("mol.model.consumer")
    vars(consumer)["confidence_alias"] = original
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    package_confidence.install_native(module, "mol1")
    assert consumer.confidence_alias is module.compute_full_data_and_summary
    assert consumer.confidence_alias is not original


def test_af2_default_trimer_iptm_does_not_calculate_actifptm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("actifpTM/contact calculation must stay off")

    expected = {(0, 1, 2, 3): 0.1, (0, 1, 4, 5): 0.2, (2, 3, 4, 5): 0.3}
    native = SimpleNamespace(
        get_chain_and_interface_metrics=forbidden,
        get_contact_map=forbidden,
        get_actifptm_probs=forbidden,
        get_pairwise_iptm=lambda data, asym, *bounds: np.array([expected[bounds]]),
    )
    monkeypatch.setitem(sys.modules, "colabfold.alphafold.extra_ptm", native)
    _, _, callback, compact = callbacks(False)
    protein = SimpleNamespace(
        chain_index=np.repeat([0, 1, 2], 2), atom_positions=np.zeros((6, 37, 3))
    )
    result = {
        "pae_matrix_with_logits": {"logits": np.zeros((6, 6, 2))},
        "predicted_aligned_error": np.ones((6, 6)),
    }
    callback(
        protein,
        [2, 2, 2],
        result,
        {"asym_id": np.repeat([1, 2, 3], 2)},
        ("model", False),
    )
    scores = compact["model"]
    assert scores["pairs"] == [["A", "B"], ["A", "C"], ["B", "C"]]
    assert scores["iptm"] == [0.1, 0.2, 0.3]
    assert scores["actifptm"] == [None, None, None]
    assert "pae_matrix_with_logits" not in result


@pytest.mark.parametrize(
    "key", ["missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys"]
)
def test_esmfold2_rejects_incomplete_checkpoints(key):
    from backends.esmfold2 import _require_loaded_weights

    with pytest.raises(RuntimeError, match="Incomplete pretrained weight loading"):
        _require_loaded_weights({key: ["encoder.embedding"]}, encoder=True)


def test_esmfold2_allows_only_unused_encoder_heads():
    from backends.esmfold2 import _require_loaded_weights

    _require_loaded_weights(
        {"unexpected_keys": ["lm_head.weight", "layer._extra_state"]}, encoder=True
    )
    with pytest.raises(RuntimeError, match="Incomplete pretrained weight loading"):
        _require_loaded_weights({"unexpected_keys": ["lm_head.weight"]})


def test_af2_unpaired_only_input_keeps_query_rows_for_every_chain(
    monkeypatch, tmp_path
):
    from backends.alphafold2 import _build_af2_queries

    captured = {}

    def serialize(unpaired, paired, sequences, cardinality):
        captured.update(paired=paired, sequences=sequences, cardinality=cardinality)
        return ">query\n" + "".join(sequences)

    monkeypatch.setitem(
        sys.modules, "colabfold.input", SimpleNamespace(msa_to_str=serialize)
    )
    msa = {}
    for sequence in ("AAA", "GGG"):
        directory = tmp_path / sequence
        directory.mkdir()
        (directory / "merged.a3m").write_text(f">query\n{sequence}\n")
        msa[sequence] = str(directory)
    _, is_complex = _build_af2_queries(
        {"unpaired": msa, "paired_dir": None}, "trimer", ["AAA", "GGG", "AAA"]
    )
    assert is_complex
    assert captured["paired"] == [">101\nAAA\n", ">102\nGGG\n", ">103\nAAA\n"]
    assert captured["sequences"] == ["AAA", "GGG", "AAA"]
    assert captured["cardinality"] == [1, 1, 1]


def test_native_arrays_roundtrip_matches_scalar(monkeypatch, tmp_path):
    def original(configs, token_asym_id, atom_to_token_idx, atom_is_polymer, pae_logits, contact_probs):
        return [{}], [{}]

    module = SimpleNamespace(
        compute_full_data_and_summary=original,
        get_bin_params=lambda bins: {},
        get_bin_centers=lambda: np.array([1.0, 5.0]),
    )
    package_confidence.install_native(module, "mol1")
    values = dict(configs=SimpleNamespace(loss=SimpleNamespace(pae=None)),
                  token_asym_id=np.array([1, 1, 2, 2, 3, 3]),
                  atom_to_token_idx=np.arange(6), atom_is_polymer=np.ones(6),
                  pae_logits=np.arange(72).reshape(1, 6, 6, 2) / 100,
                  contact_probs=np.full((6, 6), 0.4))
    monkeypatch.setenv("FOLDISM_ACTIFPTM", "1")
    monkeypatch.delenv("FOLDISM_CONFIDENCE_DIR", raising=False)
    small, _ = module.compute_full_data_and_summary(**values)
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("FOLDISM_CONFIDENCE_DIR", str(tmp_path))
    full, _ = module.compute_full_data_and_summary(**values)
    path = tmp_path / full[0]["confidence_arrays_file"]
    with np.load(path, allow_pickle=False) as arrays:
        recalculated = package_confidence.actifptm_pairs(
            arrays["pae_logits"], arrays["pae_bin_centers"], arrays["contact_probs"], arrays["chain_ids"])
    assert recalculated == small[0]["foldism_actifptm"] == full[0]["foldism_actifptm"]
    assert len(recalculated) == 3


def test_additional_arrays_survive_result_zip():
    payload = b'npz-test-payload'
    outputs = [(Path('sample_sample_0.cif'), b'data_fixture'),
               (Path('sample_summary_confidence_sample_0.json'), b'{}'),
               (Path('sample_confidence_arrays.npz'), payload)]
    files = foldism._process_outputs_to_files('mol1', outputs)
    assert ('sample_confidence_arrays.npz', payload) in files['all_files']


def test_scalar_cache_cannot_satisfy_array_download(tmp_path):
    scores = dict(version=SCORE_VERSION, actifptm_requested=True, pairs=[["A", "B"]],
                  iptm=[0.5], ipsae=[0.5], ipdae=[0.5], actifptm=[0.5],
                  status={name:["complete"] for name in ("iptm", "ipsae", "ipdae", "actifptm")})
    (tmp_path/'sample_scores.json').write_text(json.dumps({'interface_confidence': scores}))
    assert cache_supports(tmp_path, True)
    assert not cache_supports(tmp_path, True, True)
    np.savez_compressed(tmp_path/'sample_confidence_arrays.npz', pae_logits=np.zeros((2,2,2)),
                        pae_bin_centers=np.array([1,5]),contact_probs=np.ones((2,2)),chain_ids=np.array([1,2]))
    assert cache_supports(tmp_path, True, True)


def test_array_download_does_not_enter_job_state(monkeypatch, tmp_path):
    import base64
    import io
    import zipfile
    from types import SimpleNamespace
    original = foldism._store_confidence_download
    monkeypatch.setattr(foldism, 'CACHE_VOLUME', SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(foldism, '_store_confidence_download', lambda files: original(files, tmp_path))
    arrays = b'large-array-payload' * 100000
    files = {'all_files': [('sample.cif', b'data_test'), ('sample_confidence_arrays.npz', arrays)]}
    payload = foldism._build_result_payload(b'data_test', 'cif', files)
    assert len(json.dumps(payload)) < 10000
    token = payload['confidence_arrays_url'].rsplit('/', 1)[1]
    with zipfile.ZipFile(tmp_path / f'{token}.zip') as archive:
        assert archive.read('sample_confidence_arrays.npz') == arrays
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload['zip']))) as archive:
        assert archive.namelist() == ['sample.cif']
    with pytest.raises(ValueError):
        original([('../escape_confidence_arrays.npz', arrays)], tmp_path)


def test_array_download_rejects_invalid_token():
    app = foldism.web.local()
    with app.test_client() as client:
        assert client.get('/confidence-arrays/not-a-valid-token').status_code == 404


def test_af2_monomer_confidence_arrays(monkeypatch, tmp_path):
    native = SimpleNamespace(
        get_chain_and_interface_metrics=lambda *a, **k: {},
        get_contact_map=lambda result, cutoff: np.ones((2, 2)),
    )
    monkeypatch.setitem(sys.modules, 'colabfold.alphafold.extra_ptm', native)
    _, _, callback, compact = callbacks(True, tmp_path)
    protein = SimpleNamespace(chain_index=np.zeros(2, dtype=int), atom_positions=np.zeros((2, 37, 3)))
    result = {'pae_matrix_with_logits':{'logits':np.zeros((2,2,3)), 'breaks':np.array([0., 2.])},
              'predicted_aligned_error':np.ones((2,2))}
    callback(protein, [2], result, {}, ('model', False))
    assert compact['model']['pairs'] == []
    with np.load(tmp_path/'model_confidence_arrays.npz', allow_pickle=False) as arrays:
        assert arrays['pae_logits'].shape == (2,2,3)
        assert arrays['pae_bin_centers'].tolist() == [1.,3.,5.]


def test_valid_confidence_archive_http_download(monkeypatch, tmp_path):
    import io
    import zipfile
    monkeypatch.setattr(foldism, 'CONFIDENCE_DOWNLOAD_DIRECTORY', tmp_path)
    monkeypatch.setattr(foldism, 'CACHE_VOLUME', SimpleNamespace(commit=lambda: None, reload=lambda: None))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, pae_logits=np.zeros((2, 2, 2)), pae_bin_centers=np.array([1., 5.]),
                        contact_probs=np.ones((2, 2)), chain_ids=np.array([1, 2]))
    content = buffer.getvalue()
    url, size = foldism._store_confidence_download([('sample_confidence_arrays.npz', content)])
    app = foldism.web.local()
    with app.test_client() as client:
        response = client.get(url)
        assert response.status_code == 200
        assert response.content_type == 'application/zip'
        assert response.headers['Content-Disposition'].startswith('attachment;')
        assert len(response.data) == size
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            assert archive.read('sample_confidence_arrays.npz') == content
        assert client.get('/confidence-arrays/' + '0' * 64).status_code == 404


def test_cached_score_processing_adds_percentiles_without_changing_native_files():
    confidence = {
        'version': SCORE_VERSION, 'pairs': [['A', 'B']],
        'iptm': [0.7], 'ipsae': [0.6], 'ipdae': [0.5], 'actifptm': [0.8],
        'status': {key: ['complete'] for key in ('iptm', 'ipsae', 'ipdae', 'actifptm')},
    }
    native = json.dumps({'interface_confidence': confidence}).encode()
    outputs = [(Path('case_sample_0.cif'), b'data_case'), (Path('case_sample_0_scores.json'), native)]
    files = foldism._process_outputs_to_files('esmfold2', outputs, use_msa=True)
    enriched = json.loads(files['scores'])['interface_confidence']
    assert all(enriched['percentiles'][key][0] is not None for key in ('iptm', 'ipsae', 'ipdae', 'actifptm'))
    assert outputs[1][1] == native
    assert json.loads(files['unified'])['interface_confidence'] == enriched
    uncalibrated = foldism._process_outputs_to_files('esmfold2', outputs, use_msa=False)
    assert all(value is None for values in json.loads(uncalibrated['scores'])['interface_confidence']['percentiles'].values() for value in values)


def test_structure_zip_contains_percentile_metadata():
    import base64
    import io
    import zipfile

    unified = json.dumps({'interface_confidence': {'percentiles': {'iptm': [95.0]}}}).encode()
    result = foldism._build_result_payload(b'data_test', 'cif', {'unified': unified, 'all_files': [('model.cif', b'data_test')]})
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result['zip']))) as archive:
        assert archive.read('foldism_unified_scores.json') == unified
