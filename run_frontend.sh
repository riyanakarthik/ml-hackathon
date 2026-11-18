#!/usr/bin/env bash
set -e

# Change directory to repo root
cd "$(dirname "$0")"

echo "=============================================================================="
echo "                  PathSnap Evaluator Dashboard Launcher"
echo "                        (Streamlit + Folium UI)"
echo "=============================================================================="

# 1. Check or create virtual environment
if [ ! -f ".venv/bin/python" ]; then
    echo "[*] Creating virtual environment in .venv..."
    python3 -m venv .venv
    echo "[*] Installing dependencies from requirements.txt..."
    .venv/bin/pip install -r requirements.txt
fi

# 2. Launch Streamlit
echo ""
echo "[*] Starting Streamlit dashboard on port 8501..."
echo "[*] Open your browser at: http://localhost:8501"
echo "[*] Press Ctrl+C to stop the dashboard."
echo ""

.venv/bin/streamlit run frontend/app.py --server.port 8501
