# PathSnap Two-Laptop LAN Deployment Guide

**Team ID:** PT-056 (UE24CS352A — PES University, RR Campus)  
**Members:** RIYANA KARTHIK VISHALAKSHI (PES1UG24AM226), ROHANTH GANESHA C (PES1UG24AM227), ROHIT CHAKI (PES1UG24AM228), ROHITH KARNING (PES1UG24AM229)  

This guide details the physical deployment and network configuration for running the PathSnap distributed map-matching system across two physical machines over a shared local area network (LAN / Wi-Fi).

```
+------------------------------------+           +------------------------------------+
|             LAPTOP 2               |           |             LAPTOP 1               |
|      (Evaluator UI / Presenter)    |           |       (Backend Inference Server)   |
|                                    |           |                                    |
|  Streamlit Dashboard               |  HTTP GET |  FastAPI Server (Port 8000)        |
|  (frontend/app.py on Port 8501)    | --------> |  Local Tangent Plane Projector     |
|                                    |  /health  |  k-NN Candidate Generator          |
|  Folium Map Renderer               |  /trips   |  HMM + Viterbi Decoder (Optuna)    |
|  (Blue GPS points, Green Route)    |           |  Server-Side Metric Calculator     |
|                                    |  HTTP POST|                                    |
|  Evaluation Metrics & KPIs         | --------> |                                    |
|  (arc_f1, continuity, runtime_ms)  |  /predict |                                    |
+------------------------------------+           +------------------------------------+
                   \                               /
                    \                             /
                     \                           /
                      Shared Local Network (Wi-Fi)
```

---

## 1. Prerequisites

1. Both laptops must be connected to the **same Wi-Fi network or mobile hotspot**.
2. Both laptops must have cloned the team repository: `https://github.com/riyanakarthik/ml-hackathon`.
3. Python 3.10+ installed with virtual environments activated.

---

## 2. Laptop 1 Instructions (Backend & ML Server)

### Step 2.1: Find Laptop 1's IPv4 Address
Open PowerShell or Command Prompt on Laptop 1 and run:
```powershell
ipconfig
```
Look for the active adapter (e.g. `Wireless LAN adapter Wi-Fi`):
```text
IPv4 Address. . . . . . . . . . . : 192.168.1.45   <--- NOTE THIS ADDRESS
```

### Step 2.2: Configure Windows Firewall on Laptop 1
If Laptop 2 cannot connect, Windows Defender Firewall may be blocking inbound traffic on port 8000.
To allow inbound traffic on port 8000 (Private network only), run PowerShell as Administrator:
```powershell
New-NetFirewallRule -DisplayName "PathSnap FastAPI Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow -Profile Private
```
*(To remove this rule after the hackathon, run: `Remove-NetFirewallRule -DisplayName "PathSnap FastAPI Port 8000"`)*

### Step 2.3: Start the FastAPI Backend

#### Option A: 1-Click Launcher (Recommended)
* **Windows:** Run `.\run_backend.bat`
  *(Automatically detects IP addresses, checks virtualenv, and starts server)*
* **macOS / Linux:** Run `chmod +x run_backend.sh && ./run_backend.sh`

#### Option B: Manual Command
In the project root directory on Laptop 1, run:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
*Note:* Binding to `0.0.0.0` is essential so that external machines on the LAN can reach the API.

Expected terminal output:
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 3. Laptop 2 Instructions (Frontend Dashboard & Presenter)

### Step 3.1: Verify Connectivity to Laptop 1
Before launching the UI, test that Laptop 1 is reachable over the network:
```powershell
# Test network reachability
ping <LAPTOP_1_IP>

# Test FastAPI HTTP health endpoint
curl http://<LAPTOP_1_IP>:8000/health
```
Expected response:
```json
{"status":"ok","data_dir":"...","default_config":"tuned_v2","configs":["baseline_v1","tuned_candidate","tuned_v2"]}
```

### Step 3.2: Launch the Streamlit Evaluator Dashboard

#### Option A: 1-Click Launcher (Recommended)
* **Windows:** Run `.\run_frontend.bat`
* **macOS / Linux:** Run `chmod +x run_frontend.sh && ./run_frontend.sh`

#### Option B: Manual Command
In the project root directory on Laptop 2, run:
```bash
streamlit run frontend/app.py
```
Open a browser to `http://localhost:8501`.

### Step 3.3: Configure Backend Connection in the UI
1. Look at the left sidebar under **Backend Connection**.
2. Replace `http://localhost:8000` with:
   ```text
   http://<LAPTOP_1_IP>:8000
   ```
   *(e.g., `http://192.168.1.45:8000`)*
3. Click **Check Backend Status**.
4. The status badge will change to: `Backend online — {'status': 'ok', ...}`.
5. Click **Refresh Trips** to load the trip list dynamically.
6. Select any trip from the dropdown and click **Run Route Reconstruction**.

---

## 4. Troubleshooting Checklist

| Symptom | Cause | Solution |
|---|---|---|
| `Connection refused` | FastAPI is not running or listening only on `127.0.0.1` | On Laptop 1, ensure the command includes `--host 0.0.0.0`. |
| `Request timed out` | Windows Firewall or router AP client isolation | Run the PowerShell firewall command on Laptop 1. If using university Wi-Fi with client isolation, switch both laptops to a shared mobile hotspot. |
| Dropdown shows raw dicts | Old unpatched frontend | Ensure `frontend/app.py` has the contract patch applied (`_extract_id` helper). |
| 404 Not Found on `/predict` | Trip ID not found in dataset | Verify that `Student folder` on Laptop 1 contains the selected trip. |
