# Foldism

Multi-algorithm protein structure prediction using Modal serverless infrastructure.

Runs **Boltz**, **Chai-1**, **Protenix**, and **AlphaFold2** on the same input, with automatic caching via Modal Volumes.

## Quick Start

```bash
# Install dependencies
pip install modal
modal setup  # Login to Modal

# Deploy all apps (one-time)
./deploy.sh

# Run structure prediction
uv run modal run foldism.py --input-faa test.faa
```

## Three Ways to Use

### 1. CLI Mode

```bash
# Single algorithm
uv run modal run foldism.py --input-faa test.faa --algorithms boltz

# Multiple algorithms
uv run modal run foldism.py --input-faa test.faa --algorithms boltz,chai1,alphafold2

# All algorithms
uv run modal run foldism.py --input-faa test.faa

# Custom output
uv run modal run foldism.py --input-faa test.faa --run-name myrun --out-dir ./out
```

### 2. TUI Mode (Terminal UI)

```bash
uv run --with textual python foldism.py
```

Keys:
- `/` - Search files
- `r` - Run selected
- `c` - Clear log
- `q` - Quit

### 3. Web Mode

```bash
# Start local dev server
uv run modal serve foldism.py::web_app

# Or deploy
uv run modal deploy foldism.py::web_app
```

## Input Format

Simple FASTA format:

```
>A|InsulinA
GIVEQCCTSICSLYQLENYCN
>B|InsulinB
FVNQHLCGSHLVEALYLVCGERGFFYTPKT
```

Supported entity types: `protein`, `dna`, `rna`, `ligand`

## Output Structure

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

## Deployment

Deploy all apps:

```bash
./deploy.sh
```

Or deploy individually:

```bash
uv run modal deploy modal_boltz.py
uv run modal deploy modal_chai1.py
uv run modal deploy modal_protenix.py
uv run modal deploy modal_alphafold.py
uv run modal deploy foldism.py::web_app
```

## Caching

Results are cached in Modal Volumes:
- `foldism-boltz-models` - Boltz model weights
- `foldism-boltz-cache` - Boltz prediction cache
- `foldism-chai1-cache` - Chai-1 prediction cache
- `foldism-protenix-cache` - Protenix prediction cache
- `foldism-alphafold-cache` - AlphaFold2 prediction cache

Run the same input twice = instant cache hit.

## Testing

```bash
# Test CLI
./test_cli.sh

# Test TUI
./test_tui.sh

# Test web (local dev)
./test_web.sh
```

## License

MIT
