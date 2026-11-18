#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=============================================================================="
echo "                     PathSnap System Verification"
echo "              (Unit Tests, Contract Tests, & 10-Trip E2E)"
echo "=============================================================================="

if [ ! -f ".venv/bin/python" ]; then
    echo "[!] Virtual environment .venv not found. Please setup .venv first."
    exit 1
fi

echo ""
echo "[1/2] Running Unit & Contract Tests (pytest)..."
.venv/bin/python -m pytest -q

echo ""
echo "[2/2] Running End-to-End Verification Across All 10 Student Test Trips..."
.venv/bin/python frontend/verify_evaluation_trips.py

echo ""
echo "=============================================================================="
echo "[SUCCESS] All unit, contract, and E2E evaluation tests passed!"
echo "=============================================================================="
