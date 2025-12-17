#!/bin/bash
# Test CLI mode
# Usage: ./test_cli.sh

set -e

echo "Testing Foldism CLI..."
echo ""

# Test with single algorithm
echo "Running Boltz only..."
uv run modal run foldism.py --input-faa test.faa --algorithms boltz --run-name test_boltz

echo ""
echo "CLI test complete! Check ./out/fold/test_boltz/ for results."
