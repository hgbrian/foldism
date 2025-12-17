#!/bin/bash
# Test web interface locally
# Usage: ./test_web.sh

echo "Starting Foldism web interface (local development)..."
echo ""
echo "This will start a local Modal dev server."
echo "Visit the URL printed below to test the web interface."
echo ""

uv run modal serve foldism.py::web_app
