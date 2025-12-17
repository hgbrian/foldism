#!/bin/bash
# Deploy all foldism Modal apps
# Usage: ./deploy.sh

set -e

echo "======================================"
echo "Deploying Foldism Modal Apps"
echo "======================================"

# Deploy individual folding backends
echo ""
echo "Deploying Boltz..."
uv run modal deploy modal_boltz.py

echo ""
echo "Deploying Chai-1..."
uv run modal deploy modal_chai1.py

echo ""
echo "Deploying Protenix..."
uv run modal deploy modal_protenix.py

echo ""
echo "Deploying AlphaFold2..."
uv run modal deploy modal_alphafold.py

# Deploy web interface
echo ""
echo "Deploying Web Interface..."
uv run modal deploy foldism.py::web_app

echo ""
echo "======================================"
echo "All apps deployed!"
echo ""
echo "Web interface: https://<your-modal-workspace>--foldism-web-flask-app.modal.run"
echo ""
echo "To run CLI:"
echo "  uv run modal run foldism.py --input-faa test.faa"
echo ""
echo "To run TUI:"
echo "  uv run --with textual python foldism.py"
echo "======================================"
