# Foldism

Multi-algorithm protein structure prediction using Modal serverless infrastructure.

Runs **Boltz-2**, **Chai-1**, **Protenix**, **Protenix-Mini**, and **AlphaFold2** on the same input, with automatic caching.

## Quick Start

Set up uv and modal (mac and linux)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run modal setup
```

Run dev server or deploy
```bash
uv run modal serve foldism.py
uv run modal deploy foldism.py
```

## Command line interface

Examples
```bash
uv run modal run foldism.py --input-faa input.faa
uv run modal run foldism.py --input-faa input.faa --algorithms chai1,boltz2
uv run modal run foldism.py --input-faa input.faa --no-use-msa  # skip MSA (faster)
```

Test
```bash
python test_cli.py
```

## Input Format

Standard FASTA. Only protein is supported for now:

```
>Protein1
MKTAYIAKQRQISFVKSH...
>Protein2
GIVEQCCTSICSLYQLEN...
```

## License

MIT
