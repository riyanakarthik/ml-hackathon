#!/usr/bin/env bash
set -e

# Change directory to repo root
cd "$(dirname "$0")"

echo "=============================================================================="
echo "                     PathSnap Backend Server Launcher"
echo "                    (FastAPI + ML Model + HMM Decoder)"
echo "=============================================================================="

# 1. Check or create virtual environment
if [ ! -f ".venv/bin/python" ]; then
    echo "[*] Creating virtual environment in .venv..."
    python3 -m venv .venv
    echo "[*] Installing dependencies from requirements.txt..."
    .venv/bin/pip install -r requirements.txt
fi

# 2. Display Local IP Addresses
echo ""
echo "=============================================================================="
echo "[INFO] Detected Local Network IP Addresses on this Laptop:"
if command -v ip >/dev/null 2>&1; then
    ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | while read -r ip_addr; do
        echo "   -> IP: $ip_addr"
    done
elif command -v ifconfig >/dev/null 2>&1; then
    ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print "   -> IP: "$2}'
fi
echo ""
echo "[ACTION FOR LAPTOP 2]:"
echo "   In the Streamlit Dashboard on Laptop 2, enter:"
echo "   Backend URL: http://<YOUR_IP_ABOVE>:8000"
echo "=============================================================================="
echo ""
echo "[*] Starting FastAPI server on 0.0.0.0:8000..."
echo "[*] Press Ctrl+C to stop the server."
echo ""

.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
