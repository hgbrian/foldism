#!/bin/bash
# Test TUI mode
# Usage: ./test_tui.sh

echo "Launching Foldism TUI..."
echo ""
echo "Keys:"
echo "  / - Search files"
echo "  r - Run selected"
echo "  c - Clear log"
echo "  q - Quit"
echo ""

uv run --with textual python foldism.py
