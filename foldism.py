"""Multi-algorithm protein folding orchestrator with TUI and web interface.

Runs Boltz, Chai-1, Protenix, and AlphaFold2 on the same input.

## 1. TUI Mode (local terminal)

```
uv run --with textual python foldism.py
```

Keys: / search, r run, c clear log, q quit

## 2. CLI Mode (Modal remote)

```
# Single method
uv run modal run foldism.py --input-faa test.faa --algorithms boltz

# Multiple methods
uv run modal run foldism.py --input-faa test.faa --algorithms boltz,chai1,alphafold2

# All methods
uv run modal run foldism.py --input-faa test.faa

# Custom output
uv run modal run foldism.py --input-faa test.faa --run-name myrun --out-dir ./out
```

## 3. Web Mode (deploy)

```
uv run modal deploy foldism.py::web_app
```

## Example FASTA input:

```
>A
GIVEQCCTSICSLYQLENYCN
>B
FVNQHLCGSHLVEALYLVCGERGFFYTPKT
```

## Output structure:

```
out/fold/{run_name}/
├── {run_name}.boltz.cif
├── {run_name}.boltz.scores.json
├── {run_name}.chai1.cif
├── {run_name}.chai1.scores.json
├── {run_name}.protenix.cif
├── {run_name}.protenix.scores.json
├── {run_name}.alphafold2.cif
└── {run_name}.alphafold2.scores.json
```

## Notes:
- Only best model saved by default (use --keep-all for all models)
- Caching is handled by Modal Volumes (automatic, no external dependencies)
- Run same input twice = instant cache hit
- All structures saved as CIF (AlphaFold2 PDB converted via MAXIT)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from modal import App, Function, Image, wsgi_app

app = App("foldism")
web_app = App("foldism-web")


# =============================================================================
# Folding App Definitions
# =============================================================================


@dataclass
class FoldingApp:
    name: str
    key: str
    script: str
    modal_func: str  # "app_name::func_name"
    description: str
    input_flag: str
    input_format: str  # "boltz_yaml", "chai_fasta", "protenix_fasta"
    gpu_options: tuple[str, ...] = ("L40S", "A100", "A10G", "H100")
    default_gpu: str = "L40S"


FOLDING_APPS: dict[str, FoldingApp] = {
    "boltz": FoldingApp(
        name="Boltz",
        key="boltz",
        script="modal_boltz.py",
        modal_func="boltz::boltz_from_file",
        description="Boltz-2 structure prediction",
        input_flag="--input-yaml",
        input_format="boltz_yaml",
    ),
    "chai1": FoldingApp(
        name="Chai-1",
        key="chai1",
        script="modal_chai1.py",
        modal_func="chai1::chai1_from_file",
        description="Chai-1 structure prediction",
        input_flag="--input-faa",
        input_format="chai_fasta",
    ),
    "protenix": FoldingApp(
        name="Protenix",
        key="protenix",
        script="modal_protenix.py",
        modal_func="protenix::protenix_from_file",
        description="Protenix (AlphaFold3-style)",
        input_flag="--input-faa",
        input_format="protenix_fasta",
    ),
    "alphafold2": FoldingApp(
        name="AlphaFold2",
        key="alphafold2",
        script="modal_alphafold.py",
        modal_func="alphafold::alphafold_from_file",
        description="AlphaFold2/ColabFold",
        input_flag="--input-fasta",
        input_format="af2_fasta",
    ),
}


# =============================================================================
# Format Converters
# =============================================================================


def parse_fasta(fasta_str: str) -> list[tuple[str, str]]:
    """Parse FASTA string into list of (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq: list[str] = []

    for line in fasta_str.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_header is not None:
                sequences.append((current_header, "".join(current_seq)))
            current_header = line[1:].strip()
            current_seq = []
        elif line:
            current_seq.append(line)

    if current_header is not None:
        sequences.append((current_header, "".join(current_seq)))

    return sequences


def get_fasta_name(fasta_str: str) -> str:
    """Extract a name from FASTA for job naming."""
    sequences = parse_fasta(fasta_str)
    if sequences:
        return sequences[0][0].split()[0].split("|")[0]
    return "protein"


def _parse_header(header: str) -> tuple[str, str]:
    """Parse header to extract entity type and ID."""
    header_lower = header.lower()

    for entity in ("protein", "dna", "rna", "ligand", "smiles"):
        if header_lower.startswith(entity):
            parts = header.split("|")
            if len(parts) >= 2:
                if "name=" in parts[1]:
                    seq_id = parts[1].split("name=")[-1].split("|")[0].split()[0]
                else:
                    seq_id = parts[1].split()[0]
                return (entity if entity != "smiles" else "ligand", seq_id)
            else:
                seq_id = header[len(entity):].strip("_|- ") or "A"
                return (entity if entity != "smiles" else "ligand", seq_id)

    seq_id = header.split()[0].split("|")[0] if header else "A"
    return ("protein", seq_id)


def convert_to_boltz_yaml(fasta_str: str) -> str:
    """Convert FASTA to Boltz YAML format."""
    sequences = parse_fasta(fasta_str)
    lines = ["sequences:"]
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i, (header, seq) in enumerate(sequences):
        entity_type, _ = _parse_header(header)
        chain_id = chain_ids[i] if i < len(chain_ids) else f"Z{i}"
        lines.append(f"  - {entity_type}:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {seq}")

    return "\n".join(lines)


def convert_to_chai_fasta(fasta_str: str) -> str:
    """Convert FASTA to Chai-1 format."""
    sequences = parse_fasta(fasta_str)
    lines = []
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i, (header, seq) in enumerate(sequences):
        entity_type, _ = _parse_header(header)
        chain_id = chain_ids[i] if i < len(chain_ids) else f"Z{i}"
        lines.append(f">{entity_type}|name={chain_id}")
        lines.append(seq)

    return "\n".join(lines)


def convert_to_protenix_fasta(fasta_str: str) -> str:
    """Convert FASTA to Protenix format."""
    sequences = parse_fasta(fasta_str)
    lines = []
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i, (header, seq) in enumerate(sequences):
        entity_type, _ = _parse_header(header)
        chain_id = chain_ids[i] if i < len(chain_ids) else f"Z{i}"
        lines.append(f">{entity_type}|{chain_id}")
        lines.append(seq)

    return "\n".join(lines)


def convert_to_af2_fasta(fasta_str: str) -> str:
    """Convert FASTA to AlphaFold2/ColabFold format."""
    sequences = parse_fasta(fasta_str)

    if sequences:
        _, name = _parse_header(sequences[0][0])
    else:
        name = "input"

    seqs = [seq for header, seq in sequences if _parse_header(header)[0] == "protein"]
    joined = ":".join(seqs)

    return f">{name}\n{joined}"


CONVERTERS = {
    "boltz_yaml": convert_to_boltz_yaml,
    "chai_fasta": convert_to_chai_fasta,
    "protenix_fasta": convert_to_protenix_fasta,
    "af2_fasta": convert_to_af2_fasta,
}


def convert_for_app(fasta_str: str, app_key: str) -> str:
    """Convert FASTA to format required by specified app."""
    app_def = FOLDING_APPS[app_key]
    converter = CONVERTERS[app_def.input_format]
    return converter(fasta_str)


# =============================================================================
# CLI Mode (Modal local_entrypoint)
# =============================================================================


def _pdb_to_cif(pdb_bytes: bytes) -> bytes:
    """Convert PDB to CIF using MAXIT."""
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
    """Convert CIF to PDB using MAXIT."""
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


def _select_best_model(algo: str, outputs: list[tuple]) -> dict[str, bytes]:
    """Select best model from algorithm outputs and return standardized files."""
    import json

    if not outputs:
        raise ValueError(f"{algo}: No output files received from Modal function")

    files = {str(path): content for path, content in outputs}

    if algo == "chai1":
        best_idx, best_score = 0, -float("inf")
        for i in range(5):
            score_file = f"scores.model_idx_{i}.json"
            if score_file in files:
                scores = json.loads(files[score_file])
                score = scores.get("aggregate_score", [0])[0]
                if score > best_score:
                    best_score, best_idx = score, i

        result = {}
        cif_key = f"pred.model_idx_{best_idx}.cif"
        if cif_key in files:
            result["structure.cif"] = files[cif_key]
        json_key = f"scores.model_idx_{best_idx}.json"
        if json_key in files:
            result["scores.json"] = files[json_key]
        npz_key = f"scores.model_idx_{best_idx}.npz"
        if npz_key in files:
            result["scores.npz"] = files[npz_key]
        return result

    elif algo == "boltz":
        best_idx, best_score = 0, -float("inf")
        for path, content in outputs:
            path_str = str(path)
            if "confidence_fixed_model_" in path_str and path_str.endswith(".json"):
                try:
                    scores = json.loads(content)
                    score = scores.get("confidence_score", 0)
                    idx = int(path_str.split("confidence_fixed_model_")[1].split(".")[0])
                    if score > best_score:
                        best_score, best_idx = score, idx
                except (json.JSONDecodeError, ValueError, IndexError):
                    pass

        result = {}
        for path, content in outputs:
            path_str = str(path)
            if f"fixed_model_{best_idx}.cif" in path_str:
                result["structure.cif"] = content
            elif f"confidence_fixed_model_{best_idx}.json" in path_str:
                result["scores.json"] = content
            elif f"pae_fixed_model_{best_idx}.npz" in path_str:
                result["pae.npz"] = content
            elif f"pde_fixed_model_{best_idx}.npz" in path_str:
                result["pde.npz"] = content

        if not result:
            sample_paths = [str(p) for p, _ in outputs[:5]]
            raise ValueError(
                f"boltz: No matching model files found (best_idx={best_idx}). "
                f"Got {len(outputs)} files, first 5: {sample_paths}"
            )

        return result

    elif algo == "protenix":
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
                except (json.JSONDecodeError, ValueError, IndexError):
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
                        if "ranked_0.pdb" in name or "rank_001" in name and name.endswith(".pdb"):
                            pdb_bytes = zf.read(name)
                            result["structure.cif"] = _pdb_to_cif(pdb_bytes)
                        elif "ranking_debug.json" in name:
                            result["scores.json"] = zf.read(name)
                        elif "_scores_rank_001" in name and name.endswith(".json"):
                            if "scores.json" not in result:
                                result["scores.json"] = zf.read(name)
                        elif "predicted_aligned_error" in name and name.endswith(".json"):
                            result["pae.json"] = zf.read(name)
        return result

    return {}


@app.local_entrypoint()
def main(
    input_faa: str,
    algorithms: str | None = None,
    run_name: str | None = None,
    out_dir: str = "./out/fold",
    keep_all: bool = False,
):
    """Run multiple folding algorithms on the same input.

    Args:
        input_faa: Path to input FASTA file (simple format: >A\\nSEQUENCE)
        algorithms: Comma-separated list of algorithms to run (default: all)
                   Options: chai1, boltz, protenix, alphafold2
        run_name: Name for this folding run (default: from FASTA header)
        out_dir: Base output directory (default: ./out/fold)
        keep_all: If True, save all models to {algo}/ subdirectory
    """
    input_str = open(input_faa).read()

    if run_name is None:
        run_name = Path(input_faa).stem

    if algorithms is None:
        algos_to_run = list(FOLDING_APPS.keys())
    else:
        algos_to_run = [a.strip() for a in algorithms.split(",")]
        for algo in algos_to_run:
            if algo not in FOLDING_APPS:
                raise ValueError(f"Unknown algorithm: {algo}. Choose from {list(FOLDING_APPS.keys())}")

    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running algorithms: {', '.join(algos_to_run)}")
    print(f"Output directory: {run_dir}")

    for algo in algos_to_run:
        app_def = FOLDING_APPS[algo]
        print(f"\n{'='*60}")
        print(f"Running {app_def.name}...")
        print(f"{'='*60}")

        converted = convert_for_app(input_str, algo)
        print(f"Converted input for {algo}:")
        print(converted[:300] + ("..." if len(converted) > 300 else ""))

        func = Function.from_name(*app_def.modal_func.split("::"))

        if algo == "chai1":
            outputs = func.remote(
                params={"input_str": converted, "input_name": f"{run_name}.faa"},
            )
        elif algo == "boltz":
            outputs = func.remote(
                params={"input_str": converted},
            )
        elif algo == "protenix":
            outputs = func.remote(
                params={"input_str": converted, "input_name": run_name},
            )
        elif algo == "alphafold2":
            outputs = func.remote(
                params={"input_str": converted, "input_name": f"{run_name}.fasta"},
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

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
            print(f"  + All files saved to: {algo_dir}")

    print(f"\n{'='*60}")
    print("All algorithms complete!")
    print(f"Results in: {run_dir}")
    print(f"{'='*60}")


# =============================================================================
# TUI Mode (Textual)
# =============================================================================

COMMON_EXTENSIONS = (".yaml", ".yml", ".faa", ".fasta", ".fa")
MAX_RECENT_FILES = 50


def run_tui():
    """Launch the Textual TUI."""
    import asyncio

    from textual import on, work
    from textual.app import App as TextualApp
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal
    from textual.widgets import (
        Button,
        Checkbox,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        RichLog,
        Select,
        SelectionList,
        Static,
        TabbedContent,
        TabPane,
        TextArea,
    )

    class FoldismApp(TextualApp):
        """TUI for running Modal structure prediction apps."""

        CSS = """
        Screen {
            layout: grid;
            grid-size: 2;
            grid-columns: 1fr 2fr;
        }

        #sidebar {
            width: 100%;
            height: 100%;
            border: solid $primary;
            padding: 1;
        }

        #main {
            width: 100%;
            height: 100%;
        }

        #folding-methods {
            height: 6;
            margin-bottom: 1;
        }

        #config-section {
            height: auto;
            margin-bottom: 1;
        }

        .config-row {
            height: 3;
            margin-bottom: 1;
        }

        .config-label {
            width: 12;
            height: 3;
        }

        .config-input {
            width: 1fr;
        }

        #output-label {
            height: 1;
            color: $text-muted;
            margin-bottom: 1;
        }

        #output-log {
            height: 1fr;
            border: solid $accent;
            scrollbar-gutter: stable;
        }

        #run-button {
            dock: bottom;
            margin-top: 1;
        }

        .selected-file {
            background: $success 20%;
            padding: 0 1;
            margin-bottom: 1;
            height: 2;
        }

        TabbedContent {
            height: 100%;
        }

        TabPane {
            padding: 1;
        }

        #file-search {
            margin-bottom: 1;
        }

        #file-list {
            height: 1fr;
            max-height: 50%;
        }

        #file-preview-label {
            height: 1;
            margin-top: 1;
            color: $text-muted;
        }

        #file-preview {
            height: 1fr;
            min-height: 10;
        }

        #file-count {
            height: 1;
            color: $text-muted;
            margin-bottom: 1;
        }

        #progress-section {
            height: 0;
            overflow: hidden;
            margin-bottom: 0;
        }

        #progress-section.running {
            height: auto;
            margin-bottom: 1;
        }

        #progress-status {
            color: $warning;
            text-style: bold;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("r", "run_modal", "Run"),
            Binding("c", "clear_log", "Clear Log"),
            Binding("/", "focus_search", "Search"),
        ]

        def __init__(self):
            super().__init__()
            self.selected_file: Path | None = None
            self.apps_dir = Path.cwd()
            self.all_files: list[Path] = []
            self.filtered_files: list[Path] = []
            self._running_tasks: list[asyncio.Task] = []

        def action_quit(self) -> None:
            raise SystemExit(0)

        def compose(self) -> ComposeResult:
            yield Header()

            with Container(id="sidebar"):
                yield Label("Folding Methods")
                yield SelectionList[str](
                    *[(a.name, k, k == "boltz") for k, a in FOLDING_APPS.items()],
                    id="folding-methods",
                )

                with Container(id="config-section"):
                    with Horizontal(classes="config-row"):
                        yield Label("GPU:", classes="config-label")
                        yield Select(
                            [(g, g) for g in ("L40S", "A100", "A10G", "H100")],
                            value="L40S",
                            id="gpu-select",
                            classes="config-input",
                        )
                    yield Checkbox("Keep all files", id="keep-all-checkbox")

                with Container(id="progress-section"):
                    yield Static("", id="progress-status")
                    yield Static("", id="progress-bar-text")
                    yield Static("", id="output-paths")

                yield Static("No file selected", id="selected-file", classes="selected-file")
                yield Button("Run Selected", id="run-button", variant="primary")

            with Container(id="main"):
                with TabbedContent():
                    with TabPane("Files", id="files-tab"):
                        yield Input(placeholder="Search files...", id="file-search")
                        yield Static("", id="file-count")
                        yield ListView(id="file-list")
                        yield Static("Select a file to preview", id="file-preview-label")
                        yield TextArea(id="file-preview", read_only=True)
                    with TabPane("Format Preview", id="format-tab"):
                        yield Static("", id="original-file-label")
                        with Horizontal(id="preview-container"):
                            yield TextArea(id="original-preview", read_only=True)
                            yield TextArea(id="format-preview", read_only=True)
                        yield Select(
                            [(a.name, k) for k, a in FOLDING_APPS.items()],
                            value="boltz",
                            id="format-select",
                        )
                    with TabPane("Output", id="output-tab"):
                        yield Static("Command output will appear here when you run a job", id="output-label")
                        yield RichLog(id="output-log", wrap=True, markup=True)

            yield Footer()

        def on_mount(self) -> None:
            self._load_files_progressive()

        def _scan_current_dir(self) -> list[Path]:
            files = []
            for ext in COMMON_EXTENSIONS:
                files.extend(self.apps_dir.glob(f"*{ext}"))
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return files

        @work(exclusive=True, thread=True)
        def _load_files_progressive(self) -> None:
            self.all_files = self._scan_current_dir()
            self.call_from_thread(self._render_file_list)

            seen = {f.resolve() for f in self.all_files}
            for ext in COMMON_EXTENSIONS:
                for f in self.apps_dir.rglob(f"*{ext}"):
                    if f.resolve() not in seen:
                        seen.add(f.resolve())
                        self.all_files.append(f)

            self.all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            self.call_from_thread(self._render_file_list)

        def _render_file_list(self, search: str = "") -> None:
            import time

            search_lower = search.lower()
            if search:
                self.filtered_files = [
                    f for f in self.all_files if search_lower in f.name.lower()
                ][:100]
                count_label = f"{len(self.filtered_files)} matches"
            else:
                self.filtered_files = self.all_files[:MAX_RECENT_FILES]
                count_label = f"{len(self.filtered_files)} recent files"

            file_list = self.query_one("#file-list", ListView)
            file_list.clear()

            now = time.time()
            for f in self.filtered_files:
                mtime = f.stat().st_mtime
                age_secs = now - mtime
                if age_secs < 3600:
                    age_str = f"{int(age_secs / 60)}m ago"
                elif age_secs < 86400:
                    age_str = f"{int(age_secs / 3600)}h ago"
                else:
                    age_str = f"{int(age_secs / 86400)}d ago"

                item = ListItem(
                    Static(f"{f.name}  [dim]{age_str}[/]", classes="file-name"),
                    classes="file-item",
                )
                item.data = f  # type: ignore
                file_list.append(item)

            self.query_one("#file-count", Static).update(count_label)

        def _update_format_preview(self) -> None:
            if not self.selected_file:
                self.query_one("#original-file-label", Static).update("No file selected")
                self.query_one("#original-preview", TextArea).text = ""
                self.query_one("#format-preview", TextArea).text = ""
                return

            try:
                content = self.selected_file.read_text()
                format_select = self.query_one("#format-select", Select)
                app_key = str(format_select.value)
                app_name = FOLDING_APPS[app_key].name

                self.query_one("#original-file-label", Static).update(
                    f"[bold]{self.selected_file.name}[/] -> [cyan]{app_name}[/]"
                )

                self.query_one("#original-preview", TextArea).text = content

                if self.selected_file.suffix.lower() in (".yaml", ".yml"):
                    preview = content
                else:
                    preview = convert_for_app(content, app_key)

                self.query_one("#format-preview", TextArea).text = preview
            except Exception as e:
                self.query_one("#format-preview", TextArea).text = f"Error: {e}"

        def action_focus_search(self) -> None:
            self.query_one("#file-search", Input).focus()

        @on(Input.Changed, "#file-search")
        def on_search_changed(self, event: Input.Changed) -> None:
            self._render_file_list(event.value)

        @on(ListView.Selected, "#file-list")
        def on_file_selected(self, event: ListView.Selected) -> None:
            if event.item and hasattr(event.item, "data"):
                self.selected_file = event.item.data  # type: ignore
                self.query_one("#selected-file", Static).update(
                    f"Selected: {self.selected_file.name}"
                )
                self._update_format_preview()
                self._update_file_preview()

        def _update_file_preview(self) -> None:
            if not self.selected_file:
                self.query_one("#file-preview-label", Static).update("Select a file to preview")
                self.query_one("#file-preview", TextArea).text = ""
                return

            try:
                content = self.selected_file.read_text()
                self.query_one("#file-preview-label", Static).update(
                    f"[bold]{self.selected_file.name}[/]"
                )
                self.query_one("#file-preview", TextArea).text = content
            except Exception as e:
                self.query_one("#file-preview-label", Static).update(f"[red]Error: {e}[/]")
                self.query_one("#file-preview", TextArea).text = ""

        @on(Select.Changed, "#format-select")
        def on_format_changed(self, event: Select.Changed) -> None:
            self._update_format_preview()

        @on(Button.Pressed, "#run-button")
        def on_run_pressed(self) -> None:
            self.action_run_modal()

        def action_clear_log(self) -> None:
            self.query_one("#output-log", RichLog).clear()

        def _get_selected_apps(self) -> list[str]:
            selection_list = self.query_one("#folding-methods", SelectionList)
            return list(selection_list.selected)

        def action_run_modal(self) -> None:
            if not self.selected_file:
                log = self.query_one("#output-log", RichLog)
                log.write("[red]Error: No file selected[/]")
                return

            selected_apps = self._get_selected_apps()
            if not selected_apps:
                log = self.query_one("#output-log", RichLog)
                log.write("[red]Error: No folding methods selected[/]")
                return

            self.run_modal_apps(selected_apps)

        def _show_progress(self, status: str, percent: int) -> None:
            section = self.query_one("#progress-section", Container)
            section.add_class("running")
            self.query_one("#progress-status", Static).update(f"... {status}")

            filled = int(percent / 100 * 16)
            empty = 16 - filled
            bar = f"[{'#' * filled}{'-' * empty}] {percent}%"
            self.query_one("#progress-bar-text", Static).update(bar)

            self.query_one("#output-paths", Static).update("")
            self.query_one("#run-button", Button).disabled = True

        def _show_complete(self, output_paths: list[tuple[str, str]]) -> None:
            section = self.query_one("#progress-section", Container)
            section.add_class("running")
            self.query_one("#progress-status", Static).update("Complete!")
            self.query_one("#progress-bar-text", Static).update("[################] 100%")

            if output_paths:
                paths_str = "\n".join(f"  {name}: {path}" for name, path in output_paths)
                self.query_one("#output-paths", Static).update(paths_str)
            else:
                self.query_one("#output-paths", Static).update("")

            self.query_one("#run-button", Button).disabled = False

        def _hide_progress(self) -> None:
            section = self.query_one("#progress-section", Container)
            section.remove_class("running")
            self.query_one("#run-button", Button).disabled = False

        @work(exclusive=True, exit_on_error=False)
        async def run_modal_apps(self, app_keys: list[str]) -> None:
            import concurrent.futures
            import time

            log = self.query_one("#output-log", RichLog)
            log.clear()
            self.query_one("#output-label", Static).update("Running...")
            self.query_one(TabbedContent).active = "output-tab"

            input_content = self.selected_file.read_text()  # type: ignore
            input_suffix = self.selected_file.suffix.lower()  # type: ignore
            run_name = self.selected_file.stem  # type: ignore
            keep_all = self.query_one("#keep-all-checkbox", Checkbox).value

            run_dir = self.apps_dir / "out" / "fold" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            log.write("[bold yellow]--- Original Input ---[/]")
            log.write(f"[dim]File: {self.selected_file}[/]")
            preview = input_content[:300] + ("..." if len(input_content) > 300 else "")
            for line in preview.split("\n"):
                log.write(f"[dim]{line}[/]")
            log.write("")

            output_paths: list[tuple[str, str]] = []
            completed = 0
            start_time = time.time()

            jobs: list[tuple[str, str]] = []
            for app_key in app_keys:
                app_def = FOLDING_APPS[app_key]
                if input_suffix in (".yaml", ".yml") and app_def.input_format != "boltz_yaml":
                    log.write(f"[yellow]Skipping {app_def.name}: YAML not supported[/]")
                    continue
                jobs.append((app_key, app_def.name))

            log.write(f"[bold cyan]Starting {len(jobs)} algorithms in parallel...[/]")
            log.write(f"[dim]Output: {run_dir}[/]")
            if keep_all:
                log.write("[dim]Keep all files: enabled[/]")
            for _, name in jobs:
                log.write(f"[dim]  * {name}[/]")
            log.write("")

            def run_one_sync(app_key: str, name: str) -> tuple[str, list[Path] | None, str | None]:
                try:
                    app_def = FOLDING_APPS[app_key]

                    if input_suffix in (".yaml", ".yml"):
                        converted = input_content
                    else:
                        converted = convert_for_app(input_content, app_key)

                    func = Function.from_name(*app_def.modal_func.split("::"))

                    if app_key == "chai1":
                        outputs = func.remote(params={"input_str": converted, "input_name": f"{run_name}.faa"})
                    elif app_key == "boltz":
                        outputs = func.remote(params={"input_str": converted})
                    elif app_key == "protenix":
                        outputs = func.remote(params={"input_str": converted, "input_name": run_name})
                    elif app_key == "alphafold2":
                        outputs = func.remote(params={"input_str": converted, "input_name": f"{run_name}.fasta"})
                    else:
                        return (name, None, f"Unknown algorithm: {app_key}")

                    best = _select_best_model(app_key, outputs)
                    saved_files: list[Path] = []

                    for key, content in best.items():
                        ext = key.split(".")[-1]
                        if key.startswith("structure"):
                            out_path = run_dir / f"{run_name}.{app_key}.{ext}"
                        else:
                            out_path = run_dir / f"{run_name}.{app_key}.{key}"
                        out_path.write_bytes(content)
                        saved_files.append(out_path)

                    if keep_all:
                        algo_dir = run_dir / app_key
                        algo_dir.mkdir(parents=True, exist_ok=True)
                        for out_file, out_content in outputs:
                            out_path = algo_dir / Path(out_file).name
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_bytes(out_content or b"")

                    return (name, saved_files, None)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return (name, None, str(e))

            async def run_one(app_key: str, name: str) -> tuple[str, list[Path] | None]:
                nonlocal completed

                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result_name, saved_files, error = await loop.run_in_executor(
                        pool, run_one_sync, app_key, name
                    )

                completed += 1
                elapsed = int(time.time() - start_time)
                pct = int((completed / len(jobs)) * 100)
                self._show_progress(f"{completed}/{len(jobs)} complete ({elapsed}s)", pct)

                if error:
                    log.write(f"[red]X {name} failed: {error}[/]")
                    return (name, None)
                else:
                    log.write(f"[green]OK {name} completed[/]")
                    if saved_files:
                        for f in saved_files:
                            log.write(f"[dim]  -> {f.name}[/]")
                    return (name, saved_files)

            running = True

            async def update_timer():
                while running:
                    elapsed = int(time.time() - start_time)
                    pct = max(5, int((completed / len(jobs)) * 100)) if jobs else 5
                    self._show_progress(f"{completed}/{len(jobs)} running ({elapsed}s)", pct)
                    await asyncio.sleep(1)

            self._show_progress(f"Running {len(jobs)} algorithms...", 5)
            timer_task = asyncio.create_task(update_timer())
            self._running_tasks.append(timer_task)

            job_tasks = [asyncio.create_task(run_one(app_key, name)) for app_key, name in jobs]
            self._running_tasks.extend(job_tasks)

            try:
                results = await asyncio.gather(*job_tasks, return_exceptions=True)
                results = [(name, files) for r in results if isinstance(r, tuple) for name, files in [r]]
            except asyncio.CancelledError:
                results = []
            finally:
                running = False
                timer_task.cancel()
                try:
                    await timer_task
                except asyncio.CancelledError:
                    pass
                self._running_tasks = [t for t in self._running_tasks if not t.done()]

            for name, saved_files in results:
                if saved_files:
                    output_paths.append((name, str(run_dir)))

            log.write("\n[bold green]--- Complete ---[/]")
            log.write(f"[green]OK All {len(jobs)} methods finished[/]")
            log.write("")
            log.write("[bold yellow]Output directory:[/]")
            log.write(f"  {run_dir}")
            log.write("")
            log.write("[bold yellow]Output files:[/]")
            for f in sorted(run_dir.glob(f"{run_name}.*")):
                log.write(f"  [cyan]{f.name}[/]")

            self.query_one("#output-label", Static).update(f"Done! Output: {run_dir}")
            self._show_complete([(n, str(run_dir)) for n, _ in results if _])

    tui_app = FoldismApp()
    try:
        tui_app.run()
    except (KeyboardInterrupt, SystemExit):
        pass


# =============================================================================
# Web Interface (Flask + NGL)
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Foldism - Protein Structure Prediction</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; margin-bottom: 5px; }
        .subtitle { color: #666; margin-bottom: 20px; }
        .panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        h2 { margin-top: 0; color: #444; font-size: 1.2em; }
        textarea {
            width: 100%;
            height: 150px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            resize: vertical;
        }
        .methods {
            display: flex;
            gap: 15px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        .methods label {
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
        }
        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
        }
        button:hover { background: #1d4ed8; }
        button:disabled { background: #9ca3af; cursor: not-allowed; }
        .progress-container { margin-top: 15px; display: none; }
        .progress-container.active { display: block; }
        .progress-bar {
            height: 24px;
            background: #e5e7eb;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            width: 0%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
        }
        .log {
            background: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 11px;
            padding: 10px;
            border-radius: 4px;
            height: 150px;
            overflow-y: auto;
            margin-top: 10px;
        }
        .log .info { color: #4fc3f7; }
        .log .success { color: #81c784; }
        .log .error { color: #e57373; }
        #viewer-container {
            width: 100%;
            height: 500px;
            background: #fff;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .results { margin-top: 15px; }
        .result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #f9fafb;
            border-radius: 4px;
            margin-bottom: 8px;
        }
        .result-item .downloads { display: flex; gap: 8px; }
        .result-item .downloads a {
            color: #2563eb;
            text-decoration: none;
            font-size: 13px;
            padding: 4px 8px;
            border: 1px solid #2563eb;
            border-radius: 4px;
        }
        .example-link { font-size: 13px; color: #666; margin-top: 10px; }
        .example-link a { color: #2563eb; }
    </style>
</head>
<body>
    <h1>Foldism</h1>
    <p class="subtitle">Protein structure prediction with Boltz, Chai-1, Protenix, and AlphaFold2</p>

    <div class="panel">
        <h2>Input Sequence</h2>
        <form id="fold-form">
            <textarea id="fasta-input" name="fasta" placeholder="Paste FASTA sequence here...">{{ fasta or '' }}</textarea>

            <div class="methods">
                <label><input type="checkbox" name="method" value="boltz" checked> Boltz-2</label>
                <label><input type="checkbox" name="method" value="chai1"> Chai-1</label>
                <label><input type="checkbox" name="method" value="protenix"> Protenix</label>
                <label><input type="checkbox" name="method" value="alphafold2"> AlphaFold2</label>
            </div>

            <button type="submit" id="submit-btn">Predict Structure</button>
        </form>

        <div class="progress-container" id="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" id="progress-fill">0%</div>
            </div>
            <div id="progress-text">Initializing...</div>
            <div class="log" id="log"></div>
        </div>

        <div class="example-link">
            Examples: <a href="#" onclick="loadInsulin(); return false;">Insulin (51 aa)</a> |
            <a href="#" onclick="loadGFP(); return false;">GFP (238 aa)</a> |
            <a href="#" onclick="loadLysozyme(); return false;">Lysozyme (129 aa)</a>
        </div>
    </div>

    <div class="panel">
        <h2>Results</h2>
        <div class="results" id="results"></div>
        <div id="viewer-container"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/ngl@2.0.0-dev.37/dist/ngl.js"></script>
    <script>
        let stage = null;
        let loadedComponents = {};

        const methodColors = {
            'Boltz': '#3B82F6',
            'Chai-1': '#EF4444',
            'Protenix': '#22C55E',
            'AlphaFold2': '#F59E0B'
        };

        document.addEventListener("DOMContentLoaded", function () {
            stage = new NGL.Stage("viewer-container", { backgroundColor: "white" });
            window.addEventListener("resize", () => stage.handleResize(), false);
        });

        function loadInsulin() {
            document.getElementById('fasta-input').value = `>A|InsulinA
GIVEQCCTSICSLYQLENYCN
>B|InsulinB
FVNQHLCGSHLVEALYLVCGERGFFYTPKT`;
        }

        function loadGFP() {
            document.getElementById('fasta-input').value = `>GFP
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK`;
        }

        function loadLysozyme() {
            document.getElementById('fasta-input').value = `>Lysozyme
KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL`;
        }

        function log(msg, className = '') {
            const logEl = document.getElementById('log');
            const line = document.createElement('div');
            line.className = className;
            line.textContent = msg;
            logEl.appendChild(line);
            logEl.scrollTop = logEl.scrollHeight;
        }

        async function loadStructure(url, format = 'mmcif', methodName = null) {
            try {
                const ext = (format === 'pdb') ? 'pdb' : 'mmcif';
                const component = await stage.loadFile(url, { ext: ext, defaultRepresentation: false });

                const colorHex = methodColors[methodName] || '#888888';
                component.addRepresentation("cartoon", { color: colorHex, opacity: 0.9 });

                if (Object.keys(loadedComponents).length === 0) {
                    component.autoView();
                }

                loadedComponents[methodName] = component;
            } catch (e) {
                console.error(`Failed to load ${methodName}:`, e);
            }
        }

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

            if (stage) {
                stage.removeAllComponents();
                loadedComponents = {};
            }

            log('Starting prediction...', 'info');

            const formData = new FormData();
            formData.append('fasta', fasta);
            methods.forEach(m => formData.append('method', m));

            try {
                const response = await fetch('/fold', { method: 'POST', body: formData });
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, {stream: true});
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));

                                if (data.progress !== undefined) {
                                    progressFill.style.width = data.progress + '%';
                                    progressFill.textContent = data.progress + '%';
                                }
                                if (data.status) progressText.textContent = data.status;
                                if (data.log) log(data.log, data.log_class || '');

                                if (data.result) {
                                    const result = data.result;
                                    const resultData = result.data;
                                    const colorStr = methodColors[result.method] || '#888888';

                                    function base64ToBlob(b64, mime) {
                                        const bytes = atob(b64);
                                        const arr = new Uint8Array(bytes.length);
                                        for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
                                        return URL.createObjectURL(new Blob([arr], { type: mime }));
                                    }

                                    const structureMime = resultData.ext === 'pdb' ? 'chemical/x-pdb' : 'chemical/x-cif';
                                    const structureUrl = base64ToBlob(resultData.structure, structureMime);

                                    let downloadUrl = structureUrl;
                                    let downloadExt = resultData.ext;
                                    if (resultData.original_cif) {
                                        downloadUrl = base64ToBlob(resultData.original_cif, 'chemical/x-cif');
                                        downloadExt = 'cif';
                                    }

                                    let downloads = `<a href="${downloadUrl}" download="${result.method.toLowerCase()}.${downloadExt}">Structure (CIF)</a>`;
                                    if (resultData.zip) {
                                        const zipUrl = base64ToBlob(resultData.zip, 'application/zip');
                                        downloads += `<a href="${zipUrl}" download="${result.method.toLowerCase()}_all.zip">All Files (ZIP)</a>`;
                                    }

                                    const item = document.createElement('div');
                                    item.className = 'result-item';
                                    item.innerHTML = `
                                        <span style="display:flex;align-items:center;">
                                            <span style="display:inline-block;width:12px;height:12px;background:${colorStr};border-radius:2px;margin-right:6px;"></span>
                                            ${result.method}
                                        </span>
                                        <span class="downloads">${downloads}</span>
                                    `;
                                    resultsEl.appendChild(item);

                                    await loadStructure(structureUrl, result.format, result.method);
                                }

                                if (data.done) {
                                    progressFill.style.width = '100%';
                                    progressFill.textContent = '100%';
                                    progressText.textContent = 'Complete!';
                                    log('All predictions complete!', 'success');
                                }
                            } catch (e) {
                                console.error('Parse error:', e);
                            }
                        }
                    }
                }
            } catch (error) {
                log('Error: ' + error.message, 'error');
            } finally {
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

web_image = (
    Image.micromamba(python_version="3.12")
    .micromamba_install(["maxit==11.300"], channels=["conda-forge", "bioconda"])
    .pip_install("flask==3.1.0", "polars==1.19.0", "gemmi==0.7.0")
)


def _superpose_structures(structures: dict[str, bytes], reference_key: str | None = None) -> dict[str, bytes]:
    """Superpose all structures to a reference structure using gemmi."""
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
        print(f"No protein chain found in reference {ref_key}")
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
                print(f"No protein chain found in {key}")
                result[key] = cif_bytes
                continue

            target_polymer = target_chain.get_polymer()

            sup = gemmi.calculate_superposition(
                ref_polymer, target_polymer, ptype,
                gemmi.SupSelect.CaP,
                trim_cycles=3
            )

            print(f"Superposed {key} to {ref_key}: RMSD={sup.rmsd:.2f}A ({sup.count} atoms)")

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


@web_app.function(image=web_image, timeout=60 * 60)
def fold_structure(fasta_str: str, method: str) -> tuple[str, dict | None, str | None]:
    """Run a single folding method and return (method, files_dict, error)."""
    import json
    import zipfile
    from io import BytesIO

    try:
        converted = convert_for_app(fasta_str, method)
        func = Function.from_name(*FOLDING_APPS[method].modal_func.split("::"))

        if method == "chai1":
            outputs = func.remote(params={"input_str": converted, "input_name": "web.faa"})
        elif method == "boltz":
            outputs = func.remote(params={"input_str": converted})
        elif method == "protenix":
            outputs = func.remote(params={"input_str": converted, "input_name": "web"})
        elif method == "alphafold2":
            outputs = func.remote(params={"input_str": converted, "input_name": "web.fasta"})
        else:
            return (method, None, f"Unknown method: {method}")

        print(f"[{method}] Got {len(outputs)} output files")

        files = {}
        outputs_dict = {str(f): c for f, c in outputs}

        if method == "boltz":
            best_idx, best_score = 0, -float("inf")
            for path, content in outputs:
                path_str = str(path)
                if "confidence_fixed_model_" in path_str and path_str.endswith(".json"):
                    try:
                        scores = json.loads(content)
                        score = scores.get("confidence_score", 0)
                        idx = int(path_str.split("confidence_fixed_model_")[1].split(".")[0])
                        if score > best_score:
                            best_score, best_idx = score, idx
                    except:
                        pass

            for path, content in outputs:
                path_str = str(path)
                if f"fixed_model_{best_idx}.cif" in path_str:
                    files["structure"] = (content, "cif")
                elif f"confidence_fixed_model_{best_idx}.json" in path_str:
                    files["scores"] = content

        elif method == "chai1":
            best_idx, best_score = 0, -float("inf")
            for i in range(5):
                key = f"scores.model_idx_{i}.json"
                if key in outputs_dict:
                    try:
                        scores = json.loads(outputs_dict[key])
                        score = scores.get("aggregate_score", [0])[0]
                        if score > best_score:
                            best_score, best_idx = score, i
                    except:
                        pass

            cif_key = f"pred.model_idx_{best_idx}.cif"
            if cif_key in outputs_dict:
                files["structure"] = (outputs_dict[cif_key], "cif")
            json_key = f"scores.model_idx_{best_idx}.json"
            if json_key in outputs_dict:
                files["scores"] = outputs_dict[json_key]

        elif method == "protenix":
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
            for path, content in outputs:
                if str(path).endswith(".zip") and content:
                    with zipfile.ZipFile(BytesIO(content), "r") as zf:
                        for name in zf.namelist():
                            if "ranked_0.pdb" in name or ("rank_001" in name and name.endswith(".pdb")):
                                pdb_bytes = zf.read(name)
                                files["structure"] = (_pdb_to_cif(pdb_bytes), "cif")
                            elif "ranking_debug.json" in name:
                                files["scores"] = zf.read(name)

        if "structure" not in files:
            all_files_list = [str(f) for f, _ in outputs]
            return (method, None, f"No structure in {len(outputs)} files: {all_files_list[:10]}")

        files["all_files"] = [(str(f), c) for f, c in outputs if c]

        return (method, files, None)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (method, None, str(e))


@web_app.function(image=web_image, timeout=60 * 60)
@wsgi_app()
def flask_app():
    import json
    import uuid
    from io import BytesIO

    from flask import Flask, Response, render_template_string, request, send_file

    flask = Flask(__name__)
    session_files: dict[str, bytes] = {}

    @flask.route("/", methods=["GET"])
    def home():
        return render_template_string(HTML_TEMPLATE, fasta="")

    @flask.route("/fold", methods=["POST"])
    def fold():
        fasta = request.form.get("fasta", "").strip()
        methods = request.form.getlist("method")

        if not fasta:
            return Response("data: " + json.dumps({"error": "No sequence provided"}) + "\n\n",
                          mimetype="text/event-stream")

        if not methods:
            methods = ["boltz"]

        def generate():
            import time

            total_methods = len(methods)

            yield f"data: {json.dumps({'progress': 5, 'status': f'Starting {total_methods} methods in parallel...', 'log': f'Spawning {total_methods} jobs', 'log_class': 'info'})}\n\n"

            handles = []
            for method in methods:
                handle = fold_structure.spawn(fasta, method)
                handles.append((method, handle))
                yield f"data: {json.dumps({'log': f'Started {FOLDING_APPS[method].name}', 'log_class': 'dim'})}\n\n"

            all_results = {}
            structures_to_superpose = {}
            pending = list(handles)
            completed = 0
            last_heartbeat = time.time()
            heartbeat_interval = 15

            while pending:
                still_pending = []
                for method, handle in pending:
                    try:
                        method_name, files, error = handle.get(timeout=0.5)
                        completed += 1
                        progress = int((completed / total_methods) * 80)
                        all_results[method] = (files, error)

                        if error:
                            yield f"data: {json.dumps({'progress': progress, 'log': f'{method}: {error}', 'log_class': 'error'})}\n\n"
                        elif files and "structure" in files:
                            structure_bytes, fmt = files["structure"]
                            if fmt == "cif":
                                structures_to_superpose[method] = structure_bytes
                            yield f"data: {json.dumps({'progress': progress, 'status': f'{completed}/{total_methods} folded', 'log': f'{method}: Folding complete', 'log_class': 'info'})}\n\n"
                    except TimeoutError:
                        still_pending.append((method, handle))

                pending = still_pending

                now = time.time()
                if pending and (now - last_heartbeat) >= heartbeat_interval:
                    yield f"data: {json.dumps({'log': f'Waiting for {len(pending)} job(s)...', 'log_class': 'dim'})}\n\n"
                    last_heartbeat = now

                if pending:
                    time.sleep(1)

            if len(structures_to_superpose) > 1:
                yield f"data: {json.dumps({'progress': 85, 'status': 'Aligning structures...', 'log': 'Superposing structures with gemmi', 'log_class': 'info'})}\n\n"
                try:
                    ref_key = "boltz" if "boltz" in structures_to_superpose else None
                    superposed = _superpose_structures(structures_to_superpose, ref_key)
                    for method, sup_bytes in superposed.items():
                        if method in all_results and all_results[method][0]:
                            all_results[method][0]["structure"] = (sup_bytes, "cif")
                    yield f"data: {json.dumps({'progress': 95, 'log': f'Aligned {len(superposed)} structures', 'log_class': 'success'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'log': f'Superposition failed: {e}', 'log_class': 'error'})}\n\n"

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
                        import traceback
                        traceback.print_exc()
                        yield f"data: {json.dumps({'log': f'{method}: CIF to PDB failed: {e}', 'log_class': 'error'})}\n\n"

                ext = "pdb" if fmt == "pdb" else "cif"

                import base64
                structure_b64 = base64.b64encode(structure_bytes).decode('ascii')
                data_payload = {"structure": structure_b64, "ext": ext}

                if original_cif_bytes:
                    data_payload["original_cif"] = base64.b64encode(original_cif_bytes).decode('ascii')

                session_files[f"{result_id}.{ext}"] = structure_bytes

                if "scores" in files and files["scores"]:
                    scores_b64 = base64.b64encode(files["scores"]).decode('ascii')
                    data_payload["scores"] = scores_b64
                    session_files[f"{result_id}.scores.json"] = files["scores"]

                if "all_files" in files and files["all_files"]:
                    import io
                    import zipfile
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for name, content in files["all_files"]:
                            zf.writestr(name, content)
                    zip_b64 = base64.b64encode(zip_buffer.getvalue()).decode('ascii')
                    data_payload["zip"] = zip_b64
                    session_files[f"{result_id}.all.zip"] = zip_buffer.getvalue()

                yield f"data: {json.dumps({'progress': 100, 'status': 'Complete', 'log': f'{method}: Ready', 'log_class': 'success', 'result': {'method': FOLDING_APPS[method].name, 'data': data_payload, 'format': fmt}})}\n\n"

            yield f"data: {json.dumps({'progress': 100, 'done': True})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    @flask.route("/result/<filename>")
    def get_result(filename):
        from flask import abort
        if filename not in session_files:
            abort(404)

        if filename.endswith(".pdb"):
            mimetype = "chemical/x-pdb"
        elif filename.endswith(".zip"):
            mimetype = "application/zip"
        elif filename.endswith(".json"):
            mimetype = "application/json"
        else:
            mimetype = "chemical/x-mmcif"

        return send_file(
            BytesIO(session_files[filename]),
            mimetype=mimetype,
            as_attachment=filename.endswith(".zip"),
            download_name=filename
        )

    return flask


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_tui()
