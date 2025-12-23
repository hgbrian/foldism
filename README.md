# Foldism

Multi-algorithm protein structure prediction using Modal serverless infrastructure.

Runs **Boltz-2**, **Chai-1**, **Protenix**, **Protenix-Mini**, and **AlphaFold2** on the same input, with automatic caching.

## Quick Start

```bash
# Deploy
uv run modal deploy foldism.py

# Dev server
uv run modal serve foldism.py
```

## CLI

```bash
uv run modal run foldism.py --input-faa input.faa
uv run modal run foldism.py --input-faa input.faa --algorithms chai1,boltz2
uv run modal run foldism.py --input-faa input.faa --no-use-msa  # skip MSA (faster)
```

## Testing

```bash
python test_cli.py
```

## Input Format

Standard FASTA:

```
>Protein1
MKTAYIAKQRQISFVKSH...
>Protein2
GIVEQCCTSICSLYQLEN...
```

## License

MIT
