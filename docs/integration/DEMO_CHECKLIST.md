# PathSnap Evaluator Demonstration Checklist

Use this checklist during the live evaluation to ensure a seamless, error-free demonstration.

---

## 1. Pre-Demo Setup (10 Minutes Before Evaluation)

- [ ] **Network Connection:** Both Laptop 1 and Laptop 2 are connected to the exact same Wi-Fi network or mobile hotspot.
- [ ] **Git Synchronization:** Both laptops have pulled the latest `main` branch: `git pull origin main`.
- [ ] **Python Environments:** Virtual environments are active on both laptops (`.venv\Scripts\activate`).
- [ ] **Laptop 1 IP Recorded:** Person 1 ran `ipconfig` and gave the IPv4 address (e.g. `192.168.1.45`) to Person 4.

---

## 2. Launching the System

### On Laptop 1 (Backend Server — Person 1)
- [ ] Open terminal in project root.
- [ ] Run command:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```
- [ ] Confirm terminal shows:
  ```text
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  ```
- [ ] **KEEP THIS TERMINAL VISIBLE TO EVALUATORS:** The official guidelines require showing incoming API request logs on Laptop 1.

### On Laptop 2 (Dashboard — Person 4 & Person 3)
- [ ] Open terminal in project root.
- [ ] Run command:
  ```bash
  streamlit run frontend/app.py --server.address 0.0.0.0
  ```
- [ ] Open browser to `http://localhost:8501`.
- [ ] In sidebar, enter Laptop 1's backend URL: `http://<LAPTOP_1_IP>:8000`.
- [ ] Click **🔍 Check Backend Status**.
- [ ] Verify green status badge appears: `✅ Backend online`.
- [ ] Click **🔄 Refresh Trips** to dynamically populate evaluation trips.

---

## 3. Demonstration Flow for Evaluators

1. **Explain Two-Laptop Topology (30 seconds):**
   * Point to Laptop 1: *"This is our backend server hosting the road network graph, spatial indexer, and Bayesian-tuned HMM Viterbi decoder."*
   * Point to Laptop 2: *"This is our evaluator client dashboard communicating over the local network via REST API."*

2. **Select an Evaluator Trip:**
   * Ask the evaluator to choose any trip from the dropdown (e.g. `00000000`, `00000040`, or any student_test / evaluation trip).
   * Note to evaluator: *"Notice that trip IDs are loaded dynamically from `/trips` rather than hardcoded."*

3. **Execute Route Reconstruction:**
   * Click **🚀 Run Route Reconstruction**.
   * Point to Laptop 1 terminal: Show the incoming `POST /predict` HTTP log appearing live with timestamp and runtime in ms.
   * Point back to Laptop 2: The UI immediately updates with:
     * **KPI Cards:** Trip ID, Method (`hmm_viterbi`), Runtime (typically 50–200 ms), Point/Arc counts.
     * **Evaluation Metrics:** Arc Precision, Arc Recall, Arc F1, Length-weighted metrics, Route Continuity (1.0).
     * **Folium Interactive Map:** Blue circles (raw noisy GPS observations) overlaid with the continuous green polyline (reconstructed vehicle path).

4. **Demonstrate Map Interactivity:**
   * Zoom in to show GPS points that drifted off the road and how the HMM correctly snapped them to the true roadway.
   * Toggle layers using the Folium Layer Control (top right).
   * Show the Route Start (green marker) and Route End (red marker).

5. **Show Comparison / Model Rollback (Bonus):**
   * Demonstrate baseline vs. tuned HMM if asked: explain how `tuned_v2` was selected via Optuna Bayesian optimization over `baseline_v1`.

---

## 4. Emergency Backup Protocol

If Wi-Fi fails or university network blocks peer-to-peer traffic:
1. **Instant Hotspot:** Enable phone Wi-Fi hotspot, connect both laptops, check new `ipconfig` IP.
2. **Local Fallback:** On Laptop 2, run local backend in background (`uvicorn main:app --port 8000`) and point UI to `http://localhost:8000` while explaining network contingency.
