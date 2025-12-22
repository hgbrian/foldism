"""Multi-algorithm protein folding with CLI, TUI, and web interface.

Uses backends.py for Boltz-2, Chai-1, Protenix, and AlphaFold2 predictions.

## Deploy

```
uv run modal deploy foldism.py
```

## 1. CLI Mode

```
uv run modal run foldism.py --input-faa test.faa --algorithms boltz2
uv run modal run foldism.py --input-faa test.faa  # all methods
```

## 2. TUI Mode

```
uv run --with textual python foldism.py
```

## 3. Web Mode

```
uv run modal serve foldism.py  # dev
```
"""

from __future__ import annotations

import json
from pathlib import Path

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

# Job state storage for polling pattern (avoids 10-min SSE timeout)
job_store = Dict.from_name("foldism-jobs", create_if_missing=True)

# =============================================================================
# Web Image
# =============================================================================

web_image = (
    Image.micromamba(python_version="3.12")
    .micromamba_install(["maxit==11.300"], channels=["conda-forge", "bioconda"])
    .pip_install("flask==3.1.0", "polars==1.19.0", "gemmi==0.7.0", "pyyaml==6.0.2", "numpy==2.2.1")
    .add_local_python_source("backends")
)

# =============================================================================
# Helpers
# =============================================================================


def _pdb_to_cif(pdb_bytes: bytes) -> bytes:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "input.pdb"
        cif_path = Path(tmpdir) / "input.cif"
        pdb_path.write_bytes(pdb_bytes)
        subprocess.run(["maxit", "-input", str(pdb_path), "-output", str(cif_path), "-o", "1"], check=True, capture_output=True)
        return cif_path.read_bytes()


def _cif_to_pdb(cif_bytes: bytes) -> bytes:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        in_cif = Path(tmpdir) / "input.cif"
        out_pdb = Path(tmpdir) / "output.pdb"
        in_cif.write_bytes(cif_bytes)
        subprocess.run(["maxit", "-input", str(in_cif), "-output", str(out_pdb), "-o", "2"], check=True, capture_output=True)
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


def _select_best_model(algo: str, outputs: list[tuple]) -> dict[str, bytes]:
    if not outputs:
        raise ValueError(f"{algo}: No output files")

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

        result = {}
        cif_key = f"pred.model_idx_{best_idx}.cif"
        if cif_key in files:
            result["structure.cif"] = files[cif_key]
        npz_key = f"scores.model_idx_{best_idx}.npz"
        if npz_key in files:
            scores = _parse_chai1_npz(files[npz_key])
            result["scores.json"] = json.dumps(scores).encode()
        return result

    elif algo == "boltz2":
        best_idx, best_score = 0, -float("inf")
        for path, content in outputs:
            path_str = str(path)
            # Match both confidence_input_model_X.json and confidence_model_X.json
            if "confidence_" in path_str and "_model_" in path_str and path_str.endswith(".json"):
                try:
                    scores = json.loads(content)
                    score = scores.get("confidence_score", 0)
                    idx = int(path_str.split("_model_")[1].split(".")[0])
                    if score > best_score:
                        best_score, best_idx = score, idx
                except:
                    pass

        result = {}
        for path, content in outputs:
            path_str = str(path)
            # Match both input_model_X.cif and model_X.cif
            if f"_model_{best_idx}.cif" in path_str or path_str.endswith(f"model_{best_idx}.cif"):
                result["structure.cif"] = content
            elif f"confidence_" in path_str and f"_model_{best_idx}.json" in path_str:
                result["scores.json"] = content
        return result

    elif algo in ("protenix", "protenix-mini"):
        best_idx, best_score = 0, -float("inf")
        for path, content in outputs:
            path_str = str(path)
            if "summary_confidence_sample_" in path_str and path_str.endswith(".json"):
                try:
                    scores = json.loads(content)
                    score = scores.get("ranking_score", 0)
                    idx = int(path_str.split("summary_confidence_sample_")[1].split(".")[0])
                    if score > best_score:
                        best_score, best_idx = score, idx
                except:
                    pass

        result = {}
        for path, content in outputs:
            path_str = str(path)
            if f"_sample_{best_idx}.cif" in path_str:
                result["structure.cif"] = content
            elif f"summary_confidence_sample_{best_idx}.json" in path_str:
                result["scores.json"] = content
        return result

    elif algo == "alphafold2":
        import io
        import zipfile

        result = {}
        for path, content in outputs:
            if str(path).endswith(".zip") and content:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        if "ranked_0.pdb" in name or ("rank_001" in name and name.endswith(".pdb")):
                            pdb_bytes = zf.read(name)
                            result["structure.cif"] = _pdb_to_cif(pdb_bytes)
                        elif "ranking_debug.json" in name:
                            result["scores.json"] = zf.read(name)
        return result

    return {}


def _build_method_params(method: str, converted_input: str, use_msa: bool, input_name: str = "foldism") -> dict[str, Any]:
    """Build params dict for a method (centralized to avoid duplication)."""
    if method == "boltz2":
        # Boltz always uses MSA server
        return {"input_str": converted_input, "use_msa": True}
    elif method == "chai1":
        return {"input_str": converted_input, "input_name": f"{input_name}.faa", "use_msa_server": use_msa}
    elif method == "protenix":
        return {"input_str": converted_input, "input_name": input_name, "use_msa": use_msa}
    elif method == "protenix-mini":
        return {"input_str": converted_input, "input_name": input_name, "model": "protenix_mini", "use_msa": use_msa}
    elif method == "alphafold2":
        return {"input_str": converted_input, "input_name": f"{input_name}.fasta"}
    else:
        raise ValueError(f"Unknown method: {method}")


def run_algorithm(algo: str, fasta_str: str, run_name: str, use_msa: bool = True) -> list[tuple]:
    """Run a folding algorithm and return outputs."""
    converted = convert_for_app(fasta_str, algo)
    params = _build_method_params(algo, converted, use_msa, run_name)

    if algo == "boltz2":
        return boltz_predict.remote(params=params)
    elif algo == "chai1":
        return chai1_predict.remote(params=params)
    elif algo == "protenix":
        return protenix_predict.remote(params=params)
    elif algo == "protenix-mini":
        return protenix_predict.remote(params=params)
    elif algo == "alphafold2":
        return alphafold_predict.remote(params=params)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


# =============================================================================
# CLI Entry Point
# =============================================================================


@app.local_entrypoint()
def main(
    input_faa: str,
    algorithms: str | None = None,
    run_name: str | None = None,
    out_dir: str = "./out/fold",
    keep_all: bool = False,
):
    """Run multiple folding algorithms on the same input."""
    input_str = open(input_faa).read()

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
        print(f"\n{'='*60}\nRunning {app_def.name}...\n{'='*60}")

        outputs = run_algorithm(algo, input_str, run_name)

        best = _select_best_model(algo, outputs)
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
            for out_file, out_content in outputs:
                out_path = algo_dir / out_file
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(out_content or b"")

    print(f"\n{'='*60}\nComplete! Results in: {run_dir}\n{'='*60}")


# =============================================================================
# Web Interface
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Foldism - Protein Structure Prediction</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; margin-bottom: 5px; }
        .subtitle { color: #666; margin-bottom: 20px; }
        .panel { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h2 { margin-top: 0; color: #444; font-size: 1.2em; }
        textarea { width: 100%; height: 150px; font-family: monospace; font-size: 12px; border: 1px solid #ddd; border-radius: 4px; padding: 10px; }
        .methods { display: flex; gap: 15px; margin: 15px 0; flex-wrap: wrap; }
        .methods label { display: flex; align-items: center; gap: 5px; cursor: pointer; }
        button { background: #2563eb; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-size: 16px; cursor: pointer; width: 100%; }
        button:hover { background: #1d4ed8; }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        .progress-container { margin-top: 15px; display: none; }
        .progress-container.active { display: block; }
        .progress-bar { height: 24px; background: #e5e7eb; border-radius: 12px; overflow: hidden; margin-bottom: 10px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: 0%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; }
        .log { background: #1e1e1e; color: #d4d4d4; font-family: monospace; font-size: 11px; padding: 10px; border-radius: 4px; height: 150px; overflow-y: auto; margin-top: 10px; }
        .log .info { color: #4fc3f7; }
        .log .success { color: #81c784; }
        .log .error { color: #e57373; }
        #viewer-container { width: 100%; height: 500px; background: #fff; border-radius: 4px; border: 1px solid #ddd; }
        .results { margin-top: 15px; }
        .result-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; background: #f9fafb; border-radius: 4px; margin-bottom: 8px; }
        .result-item .downloads { display: flex; gap: 8px; }
        .result-item .downloads a { color: #2563eb; text-decoration: none; font-size: 13px; padding: 4px 8px; border: 1px solid #2563eb; border-radius: 4px; }
        .metrics { display: flex; gap: 8px; margin-left: 12px; }
        .metric { font-size: 12px; }
        .metric-label { color: #666; margin-right: 2px; }
        .metric-value { font-weight: 600; }
        .metric-value.good { color: #16a34a; }
        .metric-value.medium { color: #ca8a04; }
        .metric-value.poor { color: #dc2626; }
        .example-link { font-size: 13px; color: #666; margin-top: 10px; }
        .example-link a { color: #2563eb; }
    </style>
</head>
<body>
    <h1>Foldism</h1>
    <p class="subtitle">Protein structure prediction with Boltz-2, Chai-1, Protenix, and AlphaFold2</p>
    <div class="panel">
        <h2>Input Sequence</h2>
        <form id="fold-form">
            <textarea id="fasta-input" name="fasta" placeholder="Paste FASTA sequence here..."></textarea>
            <div class="methods">
                <label><input type="checkbox" name="method" value="chai1" checked> Chai-1</label>
                <label><input type="checkbox" name="method" value="boltz2"> Boltz-2</label>
                <label><input type="checkbox" name="method" value="protenix"> Protenix</label>
                <label><input type="checkbox" name="method" value="protenix-mini"> Protenix-Mini</label>
                <label><input type="checkbox" name="method" value="alphafold2"> AlphaFold2</label>
            </div>
            <div style="margin: 8px 0; font-size: 13px; color: #666;">
                <label style="cursor:pointer;"><input type="checkbox" id="use-msa" checked style="margin-right: 4px;">Use MSA</label>
                <span style="margin-left: 8px; color: #999;">(slower but higher quality; Boltz-2 and AlphaFold2 always use MSA)</span>
            </div>
            <button type="submit" id="submit-btn">Predict Structure</button>
        </form>
        <div class="progress-container" id="progress-container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div id="progress-text">Initializing...</div>
                <div id="elapsed-time" style="font-family: monospace; color: #666;">0:00</div>
            </div>
            <div class="progress-bar"><div class="progress-fill" id="progress-fill">0%</div></div>
            <div class="log" id="log"></div>
        </div>
        <div class="example-link">
            Examples: <a href="#" onclick="loadInsulin(); return false;">Insulin</a> |
            <a href="#" onclick="loadGFP(); return false;">GFP</a> |
            <a href="#" onclick="loadLysozyme(); return false;">Lysozyme</a>
        </div>
    </div>
    <div class="panel">
        <h2>Results</h2>
        <div class="results" id="results"></div>
        <div id="viewer-container"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/ngl@2.0.0-dev.37/dist/ngl.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/jszip@3/dist/jszip.min.js"></script>
    <script>
        let stage = null;
        let loadedComponents = {};
        const methodColors = {'Boltz-2': '#3B82F6', 'Chai-1': '#EF4444', 'Protenix': '#22C55E', 'Protenix-Mini': '#A855F7', 'AlphaFold2': '#F59E0B'};

        document.addEventListener("DOMContentLoaded", function () {
            stage = new NGL.Stage("viewer-container", { backgroundColor: "white" });
            window.addEventListener("resize", () => stage.handleResize(), false);
        });

        function loadInsulin() { document.getElementById('fasta-input').value = '>InsulinA\\nGIVEQCCTSICSLYQLENYCN\\n>InsulinB\\nFVNQHLCGSHLVEALYLVCGERGFFYTPKT'; }
        function loadGFP() { document.getElementById('fasta-input').value = '>GFP\\nMSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK'; }
        function loadLysozyme() { document.getElementById('fasta-input').value = '>Lysozyme\\nKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL'; }

        // Track running methods for progressive dots
        const runningMethods = {};

        // Method configurations for log sections
        const methodConfigs = {
            'boltz2': { name: 'Boltz-2', prefix: '[Boltz]' },
            'chai1': { name: 'Chai-1', prefix: '[Chai-1]' },
            'protenix': { name: 'Protenix', prefix: '[Protenix]' },
            'protenix-mini': { name: 'Protenix-Mini', prefix: '[Protenix-Mini]' },
            'alphafold2': { name: 'AlphaFold2', prefix: '[AlphaFold2]' }
        };

        function createLogSection(methodKey) {
            const config = methodConfigs[methodKey];
            if (!config) return;

            const containerId = methodKey + '-log-container';
            const detailsId = methodKey + '-details';

            // Don't create if already exists
            if (document.getElementById(detailsId)) return;

            const logEl = document.getElementById('log');
            const container = document.createElement('div');
            container.id = containerId;
            container.style.cssText = 'margin: 4px 0; padding: 6px 8px; background: rgba(255, 255, 255, 0.05); border-left: 3px solid rgba(255, 255, 255, 0.2); border-radius: 2px;';

            const header = document.createElement('div');
            header.style.cssText = 'cursor: pointer; font-size: 12px; color: rgba(255, 255, 255, 0.7); user-select: none;';
            header.innerHTML = `▶ ${config.name} detailed logs <span style="opacity: 0.5;">(click to expand)</span>`;

            const detailsEl = document.createElement('div');
            detailsEl.id = detailsId;
            detailsEl.style.cssText = 'display: none; max-height: 200px; overflow-y: auto; margin-top: 6px; padding: 6px; background: rgba(0, 0, 0, 0.3); border-radius: 2px; font-size: 11px; line-height: 1.4;';

            let isExpanded = false;
            header.onclick = () => {
                isExpanded = !isExpanded;
                detailsEl.style.display = isExpanded ? 'block' : 'none';
                header.innerHTML = `${isExpanded ? '▼' : '▶'} ${config.name} detailed logs <span style="opacity: 0.5;">(click to ${isExpanded ? 'collapse' : 'expand'})</span>`;
            };

            container.appendChild(header);
            container.appendChild(detailsEl);

            // Insert after the main method log line if it exists, otherwise append
            const methodLogLine = document.getElementById('log-' + methodKey);
            if (methodLogLine) {
                methodLogLine.parentNode.insertBefore(container, methodLogLine.nextSibling);
            } else {
                logEl.appendChild(container);
            }
        }

        function log(msg, cls = '', methodKey = null) {
            const logEl = document.getElementById('log');

            // Special handling for detailed logs from any method
            const detailedLogPrefixes = {
                '[Boltz]': { key: 'boltz2', name: 'Boltz-2' },
                '[Chai-1]': { key: 'chai1', name: 'Chai-1' },
                '[Protenix]': { key: 'protenix', name: 'Protenix' },
                '[Protenix-Mini]': { key: 'protenix-mini', name: 'Protenix-Mini' },
                '[AlphaFold2]': { key: 'alphafold2', name: 'AlphaFold2' }
            };

            for (const [prefix, config] of Object.entries(detailedLogPrefixes)) {
                if (msg.startsWith(prefix + ' ')) {
                    const containerId = config.key + '-log-container';
                    const detailsId = config.key + '-details';
                    let detailsEl = document.getElementById(detailsId);

                    if (!detailsEl) {
                        // Create expandable log section
                        const container = document.createElement('div');
                        container.id = containerId;
                        container.style.cssText = 'margin: 4px 0; padding: 6px 8px; background: rgba(255, 255, 255, 0.05); border-left: 3px solid rgba(255, 255, 255, 0.2); border-radius: 2px;';

                        const header = document.createElement('div');
                        header.style.cssText = 'cursor: pointer; font-size: 12px; color: rgba(255, 255, 255, 0.7); user-select: none;';
                        header.innerHTML = `▶ ${config.name} detailed logs <span style="opacity: 0.5;">(click to expand)</span>`;

                        detailsEl = document.createElement('div');
                        detailsEl.id = detailsId;
                        detailsEl.style.cssText = 'display: none; max-height: 200px; overflow-y: auto; margin-top: 6px; padding: 6px; background: rgba(0, 0, 0, 0.3); border-radius: 2px; font-size: 11px; line-height: 1.4;';

                        let isExpanded = false;
                        header.onclick = () => {
                            isExpanded = !isExpanded;
                            detailsEl.style.display = isExpanded ? 'block' : 'none';
                            header.innerHTML = `${isExpanded ? '▼' : '▶'} ${config.name} detailed logs <span style="opacity: 0.5;">(click to ${isExpanded ? 'collapse' : 'expand'})</span>`;
                        };

                        container.appendChild(header);
                        container.appendChild(detailsEl);

                        // Insert after the main method log line if it exists, otherwise append
                        const methodLogLine = document.getElementById('log-' + config.key);
                        if (methodLogLine) {
                            methodLogLine.parentNode.insertBefore(container, methodLogLine.nextSibling);
                        } else {
                            logEl.appendChild(container);
                        }
                    }

                    // Add log line to details
                    const logLine = document.createElement('div');
                    logLine.textContent = msg.replace(prefix + ' ', '');
                    logLine.style.cssText = 'color: rgba(255, 255, 255, 0.65); margin: 1px 0; font-family: monospace;';
                    detailsEl.appendChild(logLine);
                    detailsEl.scrollTop = detailsEl.scrollHeight;
                    return;
                }
            }

            const line = document.createElement('div');
            line.className = cls;
            line.textContent = msg;
            if (methodKey) {
                line.id = 'log-' + methodKey;
                runningMethods[methodKey] = true;
            }
            logEl.appendChild(line);
            // Create expandable log section AFTER adding line to DOM so insertion works
            if (methodKey) {
                createLogSection(methodKey);
            }
            logEl.scrollTop = logEl.scrollHeight;
        }

        function markMethodComplete(methodKey, success = true) {
            delete runningMethods[methodKey];
            const el = document.getElementById('log-' + methodKey);
            if (el) {
                // Replace "Running X..." with "Completed X" or "Failed X"
                const status = success ? 'Completed' : 'Failed';
                el.textContent = el.textContent.replace(/^Running/, status).replace(/\\.+$/, '');
                el.className = success ? 'success' : 'error';
            }
        }

        function formatMetric(value, label) {
            if (value === undefined || value === null) return '';
            // Handle array of values (e.g., multi-chain ipTM)
            if (Array.isArray(value)) {
                const formatted = value.map(v => {
                    let numVal = typeof v === 'number' ? v : parseFloat(v);
                    if (isNaN(numVal)) return null;
                    if (numVal > 1) numVal /= 100;
                    return numVal.toFixed(2);
                }).filter(v => v !== null);
                if (formatted.length === 0) return '';
                const avgVal = value.reduce((a, b) => a + (typeof b === 'number' ? b : parseFloat(b)), 0) / value.length;
                const colorClass = avgVal >= 0.8 ? 'good' : avgVal >= 0.5 ? 'medium' : 'poor';
                return `<span class="metric"><span class="metric-label">${label}:</span><span class="metric-value ${colorClass}">${formatted.join(', ')}</span></span>`;
            }
            let numVal = typeof value === 'number' ? value : parseFloat(value);
            if (isNaN(numVal)) return '';
            // Normalize to 0-1 scale (AlphaFold2 reports pLDDT as 0-100)
            if (numVal > 1) numVal /= 100;
            const displayVal = numVal.toFixed(2);
            const colorClass = numVal >= 0.8 ? 'good' : numVal >= 0.5 ? 'medium' : 'poor';
            return `<span class="metric"><span class="metric-label">${label}:</span><span class="metric-value ${colorClass}">${displayVal}</span></span>`;
        }

        function countChains(fasta) {
            // Count number of chains (headers starting with >) in FASTA
            return (fasta.match(/^>/gm) || []).length;
        }

        function extractMetrics(scores, method) {
            if (!scores) return {};
            const metrics = {};
            if (method === 'Boltz-2') {
                if (scores.confidence_score !== undefined) metrics.confidence = scores.confidence_score;
                if (scores.ptm !== undefined) metrics.ptm = scores.ptm;
                // Extract per-chain ipTM from chain 0's perspective (A vs B, A vs C, etc)
                if (scores.pair_chains_iptm && scores.pair_chains_iptm['0']) {
                    const chain0 = scores.pair_chains_iptm['0'];
                    const otherChains = Object.keys(chain0).filter(k => k !== '0').sort();
                    if (otherChains.length > 0) {
                        metrics.iptm = otherChains.map(k => chain0[k]);
                    }
                } else if (scores.iptm !== undefined) {
                    metrics.iptm = scores.iptm;
                }
            } else if (method === 'Chai-1') {
                if (scores.aggregate_score) metrics.aggregate = Array.isArray(scores.aggregate_score) ? scores.aggregate_score[0] : scores.aggregate_score;
                if (scores.ptm) metrics.ptm = Array.isArray(scores.ptm) ? scores.ptm[0] : scores.ptm;
                // Extract per-chain ipTM from chain 0's perspective (A vs B, A vs C, etc)
                // per_chain_pair_iptm is [num_chains, num_chains] - row 0 has chain 0 vs all others
                if (scores.per_chain_pair_iptm && Array.isArray(scores.per_chain_pair_iptm)) {
                    const matrix = scores.per_chain_pair_iptm;
                    // Handle nested array structure [[[]]] from npz
                    const row0 = Array.isArray(matrix[0]) && Array.isArray(matrix[0][0]) ? matrix[0][0] : matrix[0];
                    if (row0 && row0.length > 1) {
                        metrics.iptm = row0.slice(1);  // Skip diagonal (0 vs 0), get 0 vs 1, 0 vs 2, etc
                    }
                } else if (scores.iptm) {
                    metrics.iptm = Array.isArray(scores.iptm) ? scores.iptm[0] : scores.iptm;
                }
            } else if (method === 'Protenix' || method === 'Protenix-Mini') {
                if (scores.ranking_score !== undefined) metrics.ranking = scores.ranking_score;
                if (scores.ptm !== undefined) metrics.ptm = scores.ptm;
                // Protenix may have pair_iptm similar to Boltz
                if (scores.pair_iptm && typeof scores.pair_iptm === 'object') {
                    const chain0 = scores.pair_iptm['0'] || scores.pair_iptm[0];
                    if (chain0) {
                        const otherChains = Object.keys(chain0).filter(k => k !== '0' && k !== 0).sort();
                        if (otherChains.length > 0) {
                            metrics.iptm = otherChains.map(k => chain0[k]);
                        }
                    }
                } else if (scores.iptm !== undefined) {
                    metrics.iptm = scores.iptm;
                }
            } else if (method === 'AlphaFold2') {
                // ranking_debug.json format: {plddts: {model_1: 85.2}, order: [...]}
                if (scores.plddts) {
                    const plddtVals = Object.values(scores.plddts);
                    if (plddtVals.length > 0) metrics.plddt = plddtVals[0];
                }
                // Individual model scores: {plddt: 85.2} or {plddt: [per-residue array]}
                if (scores.plddt !== undefined) {
                    metrics.plddt = Array.isArray(scores.plddt)
                        ? scores.plddt.reduce((a, b) => a + b, 0) / scores.plddt.length
                        : scores.plddt;
                }
                // ptm can be dict {model_1: 0.9} or direct value
                if (scores.ptm !== undefined) {
                    if (typeof scores.ptm === 'object') {
                        const ptmVals = Object.values(scores.ptm);
                        if (ptmVals.length > 0) metrics.ptm = ptmVals[0];
                    } else metrics.ptm = scores.ptm;
                }
                if (scores.iptm !== undefined) {
                    if (typeof scores.iptm === 'object' && !Array.isArray(scores.iptm)) {
                        const iptmVals = Object.values(scores.iptm);
                        metrics.iptm = iptmVals.length > 1 ? iptmVals : iptmVals[0];
                    } else {
                        metrics.iptm = scores.iptm;
                    }
                }
            }
            return metrics;
        }

        function buildMetricsHTML(metrics, numChains) {
            if (!metrics || Object.keys(metrics).length === 0) return '';
            let html = '';
            if (metrics.plddt !== undefined) html += formatMetric(metrics.plddt, 'pLDDT');
            // ipTM only meaningful for multi-chain complexes
            if (numChains > 1) {
                if (metrics.iptm !== undefined) html += formatMetric(metrics.iptm, 'ipTM');
            }
            if (metrics.ptm !== undefined) html += formatMetric(metrics.ptm, 'pTM');
            if (metrics.confidence !== undefined) html += formatMetric(metrics.confidence, 'conf');
            if (metrics.ranking !== undefined) html += formatMetric(metrics.ranking, 'rank');
            if (metrics.aggregate !== undefined) html += formatMetric(metrics.aggregate, 'agg');
            return html ? `<span class="metrics">${html}</span>` : '';
        }

        async function loadStructure(url, format, methodName) {
            try {
                const ext = (format === 'pdb') ? 'pdb' : 'mmcif';
                console.log(`Loading ${methodName} as ${ext}...`);
                const comp = await stage.loadFile(url, { ext: ext, defaultRepresentation: false });
                console.log(`${methodName} loaded, atoms:`, comp.structure?.atomCount || 'unknown');
                comp.addRepresentation("cartoon", { color: methodColors[methodName] || '#888', opacity: 0.9 });
                if (Object.keys(loadedComponents).length === 0) {
                    comp.autoView();
                    console.log(`${methodName}: autoView called (first structure)`);
                }
                loadedComponents[methodName] = comp;
                console.log(`${methodName} added to viewport`);
                // Auto-center on all structures after each load
                stage.autoView();
            } catch (e) { console.error(`Failed to load ${methodName}:`, e); }
        }

        let downloadAllPending = null;  // Guard against concurrent calls
        async function updateDownloadAllButton(resultsData, inputHash, totalMethods) {
            if (resultsData.length === 0) return;

            // Wait for any pending update to finish
            if (downloadAllPending) await downloadAllPending;

            const updatePromise = (async () => {
                const resultsEl = document.getElementById('results');

                // Create zip from current results
                const zip = new JSZip();
                for (const r of resultsData) {
                    const methodSlug = r.method.toLowerCase().replace(/[^a-z0-9]/g, '_');
                    const structureBytes = atob(r.structure);
                    zip.file(`${methodSlug}-${inputHash}.${r.ext}`, structureBytes);

                    if (r.all_files) {
                        const allFilesBytes = atob(r.all_files);
                        const allFilesBlob = new Blob([new Uint8Array([...allFilesBytes].map(c => c.charCodeAt(0)))]);
                        const allFilesZip = await JSZip.loadAsync(allFilesBlob);
                        const folder = zip.folder(methodSlug);
                        for (const [path, fileObj] of Object.entries(allFilesZip.files)) {
                            if (!fileObj.dir) {
                                const content = await fileObj.async('uint8array');
                                folder.file(path, content);
                            }
                        }
                    }
                }

                const blob = await zip.generateAsync({type: 'blob'});
                const zipUrl = URL.createObjectURL(blob);

                // Check for existing element AFTER async work (in case another call created it)
                let downloadAllRow = document.getElementById('download-all-row');
                if (!downloadAllRow) {
                    downloadAllRow = document.createElement('div');
                    downloadAllRow.id = 'download-all-row';
                    downloadAllRow.className = 'result-item';
                    downloadAllRow.style.background = '#e0f2fe';
                    resultsEl.insertBefore(downloadAllRow, resultsEl.firstChild);
                }

                const countText = resultsData.length === totalMethods ? 'all' : `${resultsData.length}/${totalMethods}`;
                downloadAllRow.innerHTML = `<span style="font-weight:500;">Download All (${countText})</span><span class="downloads"><a href="${zipUrl}" download="foldism-${inputHash}.zip">Combined ZIP</a></span>`;
            })();

            downloadAllPending = updatePromise;
            await updatePromise;
            downloadAllPending = null;
        }

        // Re-enable submit button when textarea is edited after a run
        document.getElementById('fasta-input').addEventListener('input', () => {
            const submitBtn = document.getElementById('submit-btn');
            if (submitBtn.disabled) {
                submitBtn.disabled = false;
            }
        });

        document.getElementById('fold-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fasta = document.getElementById('fasta-input').value.trim();
            if (!fasta) { alert('Please enter a sequence'); return; }
            const methods = Array.from(document.querySelectorAll('input[name="method"]:checked')).map(el => el.value);
            if (methods.length === 0) { alert('Please select at least one method'); return; }

            const progressContainer = document.getElementById('progress-container');
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            const submitBtn = document.getElementById('submit-btn');
            const logEl = document.getElementById('log');
            const resultsEl = document.getElementById('results');

            progressContainer.classList.add('active');
            submitBtn.disabled = true;
            logEl.innerHTML = '';
            resultsEl.innerHTML = '';
            if (stage) { stage.removeAllComponents(); loadedComponents = {}; }

            // Estimate total time based on methods and protein size
            const seqLength = fasta.replace(/>[^\\n]*\\n/g, '').replace(/\\s/g, '').length;
            const useMsa = document.getElementById('use-msa').checked;
            const baseTimePerMethod = useMsa ? 180 : 60;  // seconds
            const sizeMultiplier = Math.max(1, seqLength / 200);
            const estimatedTotal = baseTimePerMethod * sizeMultiplier;  // Methods run in parallel
            const numChains = countChains(fasta);

            // Smooth progress state
            let currentProgress = 0;
            let targetProgress = 0;
            let completedMethods = 0;
            let isDone = false;

            // Start timer
            const elapsedEl = document.getElementById('elapsed-time');
            const startTime = Date.now();
            const timerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const mins = Math.floor(elapsed / 60);
                const secs = elapsed % 60;
                elapsedEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
            }, 1000);

            // Smooth progress animation with asymptotic behavior
            const progressInterval = setInterval(() => {
                if (isDone) {
                    currentProgress = 100;
                } else {
                    const elapsed = (Date.now() - startTime) / 1000;
                    // Base progress from time elapsed (asymptotic to ~80%)
                    // Time constant = estimatedTotal means ~63% at estimatedTotal seconds
                    const timeProgress = 80 * (1 - Math.exp(-elapsed / estimatedTotal));
                    // Bonus from completed methods
                    const methodProgress = (completedMethods / methods.length) * 15;
                    targetProgress = Math.min(95, timeProgress + methodProgress);
                    // Smooth interpolation toward target
                    currentProgress += (targetProgress - currentProgress) * 0.1;
                }
                const displayProgress = Math.round(currentProgress);
                progressFill.style.width = displayProgress + '%';
                progressFill.textContent = displayProgress + '%';
                if (isDone) clearInterval(progressInterval);
            }, 100);

            log('Starting prediction...', 'info');

            // Generate 6-char hash of input for filenames
            async function hashFasta(str) {
                const encoder = new TextEncoder();
                const data = encoder.encode(str);
                const hashBuffer = await crypto.subtle.digest('SHA-256', data);
                const hashArray = Array.from(new Uint8Array(hashBuffer));
                return hashArray.map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 6);
            }
            const inputHash = await hashFasta(fasta);

            function base64ToBlob(b64, mime) {
                const bytes = atob(b64);
                const arr = new Uint8Array(bytes.length);
                for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
                return URL.createObjectURL(new Blob([arr], { type: mime }));
            }

            async function createPartialZip(resultsData, inputHash) {
                const zip = new JSZip();

                for (const r of resultsData) {
                    const methodSlug = r.method.toLowerCase().replace(/[^a-z0-9]/g, '_');
                    const structureBytes = atob(r.structure);
                    zip.file(`${methodSlug}-${inputHash}.${r.ext}`, structureBytes);

                    if (r.all_files) {
                        const allFilesBytes = atob(r.all_files);
                        const allFilesBlob = new Blob([new Uint8Array([...allFilesBytes].map(c => c.charCodeAt(0)))]);
                        const allFilesZip = await JSZip.loadAsync(allFilesBlob);
                        const folder = zip.folder(methodSlug);
                        for (const [path, fileObj] of Object.entries(allFilesZip.files)) {
                            if (!fileObj.dir) {
                                const content = await fileObj.async('uint8array');
                                folder.file(path, content);
                            }
                        }
                    }
                }

                const blob = await zip.generateAsync({type: 'blob'});
                return URL.createObjectURL(blob);
            }

            const formData = new FormData();
            formData.append('fasta', fasta);
            methods.forEach(m => formData.append('method', m));
            formData.append('use_msa', useMsa ? 'true' : 'false');

            try {
                // Submit job and get job_id
                const response = await fetch('/fold', { method: 'POST', body: formData });
                const { job_id, error: submitError } = await response.json();
                if (submitError) { log('Error: ' + submitError, 'error'); return; }

                // Track displayed logs and results to avoid duplicates
                let displayedLogs = 0;
                let displayedResults = new Set();
                let completedResultsData = [];
                let pollCount = 0;

                // Poll for status
                function getPollDelay() {
                    if (pollCount < 3) return 1000;   // First 3 polls: 1s
                    return 3000;                       // After that: 3s
                }

                async function poll() {
                    pollCount++;
                    try {
                        const statusRes = await fetch('/status/' + job_id);
                        const data = await statusRes.json();

                        if (data.error) {
                            log('Error: ' + data.error, 'error');
                            isDone = true;
                            clearInterval(timerInterval);
                            clearInterval(progressInterval);
                            submitBtn.disabled = false;
                            return;
                        }

                        // Update progress and status
                        if (data.status) progressText.textContent = data.status;
                        if (data.progress) targetProgress = data.progress;

                        // Display new logs
                        if (data.logs && data.logs.length > displayedLogs) {
                            for (let i = displayedLogs; i < data.logs.length; i++) {
                                const l = data.logs[i];
                                if (l.msg) log(l.msg, l.cls || '', l.method_key || null);
                                if (l.method_complete) markMethodComplete(l.method_complete, l.cls !== 'error');
                            }
                            displayedLogs = data.logs.length;
                        }

                        // Handle method completion/error from top-level data
                        if (data.method_complete) {
                            markMethodComplete(data.method_complete, !data.method_error);
                        }

                        // Display new results
                        if (data.results) {
                            for (const result of data.results) {
                                if (displayedResults.has(result.method_key)) continue;
                                displayedResults.add(result.method_key);
                                completedMethods++;

                                const resultData = result.data;
                                const colorStr = methodColors[result.method] || '#888';

                                const structureMime = resultData.ext === 'pdb' ? 'chemical/x-pdb' : 'chemical/x-cif';
                                const structureUrl = base64ToBlob(resultData.structure, structureMime);
                                let downloadUrl = structureUrl, downloadExt = resultData.ext;
                                if (resultData.original_cif) {
                                    downloadUrl = base64ToBlob(resultData.original_cif, 'chemical/x-cif');
                                    downloadExt = 'cif';
                                }

                                const methodSlug = result.method.toLowerCase().replace(/[^a-z0-9]/g, '');
                                let downloads = `<a href="${downloadUrl}" download="${methodSlug}-${inputHash}.${downloadExt}">Structure</a>`;
                                if (resultData.zip) {
                                    const zipUrl = base64ToBlob(resultData.zip, 'application/zip');
                                    downloads += `<a href="${zipUrl}" download="${methodSlug}-${inputHash}_all.zip">All Files</a>`;
                                }

                                // Parse scores and build metrics HTML
                                let metricsHTML = '';
                                if (resultData.scores) {
                                    try {
                                        const scoresBytes = atob(resultData.scores);
                                        const scores = JSON.parse(scoresBytes);
                                        const metrics = extractMetrics(scores, result.method);
                                        metricsHTML = buildMetricsHTML(metrics, numChains);
                                    } catch (e) { console.error('Failed to parse scores:', e); }
                                }

                                const item = document.createElement('div');
                                item.className = 'result-item';
                                item.innerHTML = `<span style="display:flex;align-items:center;"><span style="display:inline-block;width:12px;height:12px;background:${colorStr};border-radius:2px;margin-right:6px;"></span>${result.method}${metricsHTML}</span><span class="downloads">${downloads}</span>`;
                                resultsEl.appendChild(item);

                                // Load structure asynchronously (don't block on it)
                                loadStructure(structureUrl, result.format, result.method).catch(e =>
                                    console.error(`Failed to load structure for ${result.method}:`, e)
                                );

                                // Track completed results for partial download
                                completedResultsData.push({
                                    method: result.method,
                                    method_key: result.method_key,
                                    structure: resultData.original_cif || resultData.structure,
                                    ext: resultData.original_cif ? 'cif' : resultData.ext,
                                    all_files: resultData.zip
                                });

                                // Update "Download All" button after each result
                                updateDownloadAllButton(completedResultsData, inputHash, methods.length);
                            }
                        }

                        // Check if done
                        if (data.done) {
                            isDone = true;
                            currentProgress = 100;
                            progressFill.style.width = '100%';
                            progressFill.textContent = '100%';
                            progressText.textContent = 'Complete!';
                            log('All predictions complete!', 'success');

                            clearInterval(timerInterval);
                            clearInterval(progressInterval);
                            submitBtn.disabled = false;
                        }
                    } catch (pollError) {
                        console.error('Poll error:', pollError);
                    }
                    if (!isDone) setTimeout(poll, getPollDelay());
                }
                setTimeout(poll, 500);  // First poll after 500ms
            } catch (error) { log('Error: ' + error.message, 'error'); clearInterval(timerInterval); clearInterval(progressInterval); submitBtn.disabled = false; }
        });
    </script>
</body>
</html>
"""


def _superpose_structures(structures: dict[str, bytes], reference_key: str | None = None) -> dict[str, bytes]:
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
                if polymer and polymer.check_polymer_type() == gemmi.PolymerType.PeptideL:
                    target_chain = chain
                    break

            if not target_chain:
                result[key] = cif_bytes
                continue

            target_polymer = target_chain.get_polymer()
            sup = gemmi.calculate_superposition(ref_polymer, target_polymer, ptype, gemmi.SupSelect.CaP, trim_cycles=3)
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


@app.function(image=web_image, timeout=60 * 60, volumes={"/cache": CACHE_VOLUME})
def fold_structure_web(fasta_str: str, method: str, use_msa: bool = True, job_id: str | None = None) -> tuple[str, dict | None, str | None]:
    """Run a single folding method for web interface."""
    try:
        converted = convert_for_app(fasta_str, method)
        params = _build_method_params(method, converted, use_msa, "foldism")

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
        best_idx, best_score = 0, -float("inf")
        for path, content in outputs:
            path_str = str(path)
            if "confidence_" in path_str and "_model_" in path_str and path_str.endswith(".json"):
                try:
                    scores = json.loads(content)
                    score = scores.get("confidence_score", 0)
                    idx = int(path_str.split("_model_")[1].split(".")[0])
                    if score > best_score:
                        best_score, best_idx = score, idx
                except:
                    pass
        for path, content in outputs:
            path_str = str(path)
            if f"_model_{best_idx}.cif" in path_str or path_str.endswith(f"model_{best_idx}.cif"):
                files["structure"] = (content, "cif")
            elif "confidence_" in path_str and f"_model_{best_idx}.json" in path_str:
                files["scores"] = content

    elif method == "chai1":
        best_idx, best_score = 0, -float("inf")
        for i in range(5):
            key = f"scores.model_idx_{i}.npz"
            if key in outputs_dict:
                try:
                    scores = _parse_chai1_npz(outputs_dict[key])
                    score = scores.get("aggregate_score", [0])[0]
                    if score > best_score:
                        best_score, best_idx = score, i
                except:
                    pass
        cif_key = f"pred.model_idx_{best_idx}.cif"
        if cif_key in outputs_dict:
            files["structure"] = (outputs_dict[cif_key], "cif")
        npz_key = f"scores.model_idx_{best_idx}.npz"
        if npz_key in outputs_dict:
            scores = _parse_chai1_npz(outputs_dict[npz_key])
            files["scores"] = json.dumps(scores).encode()

    elif method in ("protenix", "protenix-mini"):
        best_idx, best_score = 0, -float("inf")
        for path, content in outputs:
            path_str = str(path)
            if "summary_confidence_sample_" in path_str and path_str.endswith(".json"):
                try:
                    scores = json.loads(content)
                    score = scores.get("ranking_score", 0)
                    idx = int(path_str.split("summary_confidence_sample_")[1].split(".")[0])
                    if score > best_score:
                        best_score, best_idx = score, idx
                except:
                    pass
        for path, content in outputs:
            path_str = str(path)
            if f"_sample_{best_idx}.cif" in path_str:
                files["structure"] = (content, "cif")
            elif f"summary_confidence_sample_{best_idx}.json" in path_str:
                files["scores"] = content

    elif method == "alphafold2":
        extracted_files = []
        for path, content in outputs:
            if str(path).endswith(".zip") and content:
                with zipfile.ZipFile(BytesIO(content), "r") as zf:
                    print(f"[alphafold2] Zip contains: {zf.namelist()}")
                    for name in zf.namelist():
                        if name.endswith("/"):  # Skip directories
                            continue
                        file_content = zf.read(name)
                        extracted_files.append((name, file_content))
                        if "ranked_0.pdb" in name or ("rank_001" in name and name.endswith(".pdb")):
                            files["structure"] = (_pdb_to_cif(file_content), "cif")
                            print(f"[alphafold2] Found structure: {name}")
                        elif "ranking_debug.json" in name:
                            files["scores"] = file_content
                            print(f"[alphafold2] Found ranking_debug.json")
                        elif "_scores_rank_001" in name and name.endswith(".json"):
                            if "scores" not in files:
                                files["scores"] = file_content
                                print(f"[alphafold2] Found scores file: {name}")
        # Use extracted files instead of the zip
        if extracted_files:
            files["all_files"] = extracted_files
            return files if "structure" in files else None

    if "structure" not in files:
        return None

    files["all_files"] = [(str(f), c) for f, c in outputs if c]
    return files


@app.function(image=web_image, timeout=60 * 60, volumes={"/cache": CACHE_VOLUME})
def run_folding_job(job_id: str, fasta: str, methods: list[str], use_msa: bool):
    """Background orchestrator for folding jobs (avoids 10-min SSE timeout)."""
    import base64
    import hashlib
    import io
    import time
    import zipfile

    def update_job(progress: int, status: str, logs: list[dict] | None = None, results: list[dict] | None = None, done: bool = False, combined_zip: str | None = None):
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
        if combined_zip is not None:
            state["combined_zip"] = combined_zip
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
            logs.append({"msg": f"{FOLDING_APPS[method].name}: cached", "cls": "success"})
        else:
            methods_to_run.append(method)
            # Boltz always uses MSA, others respect checkbox
            if method == "boltz2":
                msa_note = " (with MSA)"
            else:
                msa_note = " (with MSA)" if use_msa else ""
            logs.append({"msg": f"Running {FOLDING_APPS[method].name}{msa_note}", "cls": "dim", "method_key": method})
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
                    logs.append({"msg": f"{FOLDING_APPS[method].name}: Using as reference for alignment", "cls": "info"})
                else:
                    try:
                        ref_key = list(structures_to_superpose.keys())[0]
                        ref_structure = structures_to_superpose[ref_key]
                        to_align = {ref_key: ref_structure, method: structure_bytes}
                        aligned = _superpose_structures(to_align, ref_key)
                        structure_bytes = aligned[method]
                        structures_to_superpose[method] = structure_bytes
                        logs.append({"msg": f"{FOLDING_APPS[method].name}: Aligned to {FOLDING_APPS[ref_key].name}", "cls": "info"})
                    except Exception as e:
                        logs.append({"msg": f"{FOLDING_APPS[method].name}: Alignment failed: {e}", "cls": "warning"})
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
                    logs.append({"msg": f"{method}: CIF to PDB failed: {e}", "cls": "error"})

            ext = "pdb" if display_fmt == "pdb" else "cif"
            data_payload = {"structure": base64.b64encode(display_bytes).decode("ascii"), "ext": ext}
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

            result_to_send = [{"method": FOLDING_APPS[method].name, "method_key": method, "data": data_payload, "format": display_fmt}]
            logs.append({"method_complete": method})
            progress = int(10 + (len(all_results) / total) * 70)
            update_job(progress, f"{len(all_results)}/{total} complete", logs, result_to_send)

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
                logs.append({"msg": f"{FOLDING_APPS[method].name}: Timeout after {int(elapsed/60)} minutes", "cls": "error", "method_complete": method})
                all_results[method] = (None, f"Timeout after {int(elapsed/60)} minutes")
                completed += 1
                progress = int(10 + (completed / total) * 70)
                running = len(still_pending)
                status = f"{completed}/{total} complete" + (f" ({running} running)" if running > 0 else "")
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
                            logs.append({"msg": f"{FOLDING_APPS[method].name}: Using as reference for alignment", "cls": "info"})
                        else:
                            # Superpose this structure to the reference
                            try:
                                ref_key = list(structures_to_superpose.keys())[0]
                                ref_structure = structures_to_superpose[ref_key]
                                to_align = {ref_key: ref_structure, method: structure_bytes}
                                aligned = _superpose_structures(to_align, ref_key)
                                structure_bytes = aligned[method]
                                structures_to_superpose[method] = structure_bytes
                                logs.append({"msg": f"{FOLDING_APPS[method].name}: Aligned to {FOLDING_APPS[ref_key].name}", "cls": "info"})
                            except Exception as e:
                                logs.append({"msg": f"{FOLDING_APPS[method].name}: Alignment failed: {e}", "cls": "warning"})
                                structures_to_superpose[method] = structure_bytes

                    # Convert to PDB for viewer
                    original_cif_bytes = structure_bytes if fmt == "cif" else None
                    if fmt == "cif":
                        try:
                            structure_bytes = _cif_to_pdb(structure_bytes)
                            fmt = "pdb"
                        except Exception as e:
                            logs.append({"msg": f"{method}: CIF to PDB failed: {e}", "cls": "error"})

                    # Build result payload
                    ext = "pdb" if fmt == "pdb" else "cif"
                    data_payload = {"structure": base64.b64encode(structure_bytes).decode("ascii"), "ext": ext}
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

                    result_to_send = [{"method": FOLDING_APPS[method].name, "method_key": method, "data": data_payload, "format": fmt}]

                if error:
                    logs.append({"msg": f"{FOLDING_APPS[method].name}: {error}", "cls": "error", "method_complete": method})
                else:
                    logs.append({"method_complete": method})

                running = len(still_pending)
                status = f"{completed}/{total} complete" + (f" ({running} running)" if running > 0 else "")
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
                            logs_to_send = [{"msg": f"[{display_name}] {line}", "cls": "dim", "method_key": method_key} for line in new_logs[last_count:]]
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
                logs.append({"msg": f"{method}: CIF to PDB failed: {e}", "cls": "error"})

        ext = "pdb" if fmt == "pdb" else "cif"
        data_payload = {"structure": base64.b64encode(structure_bytes).decode("ascii"), "ext": ext}

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

        results.append({"method": FOLDING_APPS[method].name, "method_key": method, "data": data_payload, "format": fmt})

    # Create combined zip
    combined_zip_b64 = None
    try:
        input_hash = hashlib.sha256(fasta.encode()).hexdigest()[:6]
        combined_buffer = io.BytesIO()
        with zipfile.ZipFile(combined_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for method, (files, error) in all_results.items():
                if error or not files:
                    continue
                method_slug = FOLDING_APPS[method].name.lower().replace("-", "_")
                if "structure" in files:
                    structure_bytes, fmt = files["structure"]
                    # Use aligned structure if available
                    if method in structures_to_superpose:
                        structure_bytes = structures_to_superpose[method]
                    zf.writestr(f"{method_slug}-{input_hash}.{fmt}", structure_bytes)
                if "all_files" in files and files["all_files"]:
                    for name, content in files["all_files"]:
                        zf.writestr(f"{method_slug}/{name}", content)
        combined_zip_b64 = base64.b64encode(combined_buffer.getvalue()).decode("ascii")
    except Exception as e:
        print(f"[job] Failed to create combined zip: {e}")

    update_job(100, "Complete", logs, results, done=True, combined_zip=combined_zip_b64)


def _check_cache_inline_for_job(method: str, fasta_str: str, use_msa: bool) -> list | None:
    """Check cache inline for job orchestrator."""
    converted = convert_for_app(fasta_str, method)
    params = _build_method_params(method, converted, use_msa, "foldism")

    cache_key = get_cache_key(method, params)
    cache_subdir = get_cache_subdir(method)
    cache_path = Path(f"/cache/{cache_subdir}/{cache_key}")
    cache_marker = cache_path / "_COMPLETE"

    if not cache_marker.exists():
        return None

    print(f"[job-cache] {method} HIT: {cache_key}")
    CACHE_VOLUME.reload()

    return [
        (str(f.relative_to(cache_path)), f.read_bytes())
        for f in cache_path.glob("**/*")
        if f.is_file() and f.name != "_COMPLETE"
    ]


@app.function(image=web_image, timeout=60 * 60, volumes={"/cache": CACHE_VOLUME})
@wsgi_app()
def web():
    from io import BytesIO
    from flask import Flask, Response, render_template_string, request, send_file, abort

    flask_app = Flask(__name__)
    session_files: dict[str, bytes] = {}

    def check_cache_in_server(method: str, fasta_str: str, use_msa: bool) -> list | None:
        """Check cache directly in the web server - no container spawn needed."""
        from pathlib import Path

        converted = convert_for_app(fasta_str, method)

        # Build params dict for cache key
        if method == "boltz2":
            params = {"input_str": converted, "use_msa": True}
        elif method == "chai1":
            params = {"input_str": converted, "input_name": "foldism.faa", "use_msa_server": use_msa}
        elif method == "protenix":
            params = {"input_str": converted, "input_name": "foldism", "use_msa": use_msa}
        elif method == "protenix-mini":
            params = {"input_str": converted, "input_name": "foldism", "model": "protenix_mini", "use_msa": use_msa}
        elif method == "alphafold2":
            params = {"input_str": converted, "input_name": "foldism.fasta"}
        else:
            return None

        cache_key = get_cache_key(method, params)
        cache_subdir = get_cache_subdir(method)
        cache_path = Path(f"/cache/{cache_subdir}/{cache_key}")
        cache_marker = cache_path / "_COMPLETE"

        if not cache_marker.exists():
            return None

        print(f"[server-cache] {method} HIT: {cache_key}")
        CACHE_VOLUME.reload()

        return [
            (str(f.relative_to(cache_path)), f.read_bytes())
            for f in cache_path.glob("**/*")
            if f.is_file() and f.name != "_COMPLETE"
        ]

    @flask_app.route("/", methods=["GET"])
    def home():
        return render_template_string(HTML_TEMPLATE)

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
            return Response("data: " + json.dumps({"error": "No sequence"}) + "\n\n", mimetype="text/event-stream")

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
                    pending_names = ', '.join(FOLDING_APPS[m].name for m, _ in pending)
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
                structure_b64 = base64.b64encode(structure_bytes).decode('ascii')
                data_payload = {"structure": structure_b64, "ext": ext}

                if original_cif_bytes:
                    data_payload["original_cif"] = base64.b64encode(original_cif_bytes).decode('ascii')

                session_files[f"{result_id}.{ext}"] = structure_bytes

                if "scores" in files and files["scores"]:
                    data_payload["scores"] = base64.b64encode(files["scores"]).decode('ascii')

                if "all_files" in files and files["all_files"]:
                    import io
                    import zipfile
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for name, content in files["all_files"]:
                            zf.writestr(name, content)
                    data_payload["zip"] = base64.b64encode(zip_buffer.getvalue()).decode('ascii')

                yield f"data: {json.dumps({'progress': 100, 'status': 'Complete', 'result': {'method': FOLDING_APPS[method].name, 'method_key': method, 'data': data_payload, 'format': fmt}})}\n\n"

            # Create combined zip with all files from all methods
            # Root level: best structure from each method (boltz-{hash}.cif, chai1-{hash}.cif, etc)
            # Subdirectories: all files from each method
            combined_zip_b64 = None
            try:
                import io
                import zipfile
                import hashlib
                input_hash = hashlib.sha256(fasta.encode()).hexdigest()[:6]
                combined_buffer = io.BytesIO()
                with zipfile.ZipFile(combined_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for method, (files, error) in all_results.items():
                        if error or not files:
                            continue
                        method_slug = FOLDING_APPS[method].name.lower().replace("-", "_")
                        # Add best structure at root level
                        if "structure" in files:
                            structure_bytes, fmt = files["structure"]
                            zf.writestr(f"{method_slug}-{input_hash}.{fmt}", structure_bytes)
                        # Add all files under method subdirectory
                        if "all_files" in files and files["all_files"]:
                            for name, content in files["all_files"]:
                                zf.writestr(f"{method_slug}/{name}", content)
                combined_zip_b64 = base64.b64encode(combined_buffer.getvalue()).decode('ascii')
            except Exception as e:
                print(f"[web] Failed to create combined zip: {e}")

            done_payload = {'progress': 100, 'done': True}
            if combined_zip_b64:
                done_payload['combined_zip'] = combined_zip_b64
            yield f"data: {json.dumps(done_payload)}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    @flask_app.route("/result/<filename>")
    def get_result(filename):
        if filename not in session_files:
            abort(404)
        mimetype = "chemical/x-pdb" if filename.endswith(".pdb") else "application/zip" if filename.endswith(".zip") else "chemical/x-mmcif"
        return send_file(BytesIO(session_files[filename]), mimetype=mimetype, download_name=filename)

    return flask_app


# =============================================================================
# TUI Mode
# =============================================================================

if __name__ == "__main__":
    try:
        from textual.app import App as TextualApp
        from textual.app import ComposeResult
        from textual.binding import Binding
        from textual.widgets import Button, Footer, Header, Label, RichLog, TextArea

        class FoldismTUI(TextualApp):
            CSS = """
            Screen { layout: vertical; }
            #input-area { height: 12; margin: 1; }
            #input-label { height: 1; }
            #output-log { height: 1fr; margin: 1; border: solid $accent; }
            #run-button { margin: 1; }
            """
            BINDINGS = [Binding("q", "quit", "Quit"), Binding("r", "run", "Run")]

            def compose(self) -> ComposeResult:
                yield Header()
                yield Label("Paste FASTA sequence:", id="input-label")
                yield TextArea(id="input-area")
                yield Button("Run Boltz-2", id="run-button", variant="primary")
                yield RichLog(id="output-log", wrap=True, markup=True)
                yield Footer()

            def action_quit(self) -> None:
                raise SystemExit(0)

            def on_button_pressed(self, event: Button.Pressed) -> None:
                if event.button.id == "run-button":
                    self.action_run()

            def action_run(self) -> None:
                log = self.query_one("#output-log", RichLog)
                input_area = self.query_one("#input-area", TextArea)
                fasta = input_area.text.strip()

                if not fasta:
                    log.write("[red]Error: No sequence entered[/]")
                    return

                log.write("[cyan]Starting Boltz-2 prediction...[/]")
                log.write("[dim]This will use the deployed foldism app.[/]")
                log.write("[dim]Run: uv run modal run foldism.py --input-faa <file>[/]")
                log.write("")
                log.write("[yellow]TUI is for quick preview. Use CLI for actual runs.[/]")

        tui = FoldismTUI()
        tui.run()

    except ImportError:
        print("TUI requires textual. Install with: pip install textual")
        print("")
        print("For CLI mode, run:")
        print("  uv run modal run foldism.py --input-faa test.faa")
