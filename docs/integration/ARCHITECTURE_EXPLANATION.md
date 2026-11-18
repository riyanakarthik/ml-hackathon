# PathSnap System Architecture & Technical Explanation

**Project:** Distributed Map-Matching & Route Reconstruction  
**Course:** UE24CS352A — Machine Learning, PES University (RR Campus)  
**Team ID:** PT-056  
**Members:** RIYANA KARTHIK VISHALAKSHI (PES1UG24AM226), ROHANTH GANESHA C (PES1UG24AM227), ROHIT CHAKI (PES1UG24AM228), ROHITH KARNING (PES1UG24AM229)

---

## 1. System Overview & Deployment Topology

PathSnap solves the problem of reconstructing true vehicle travel trajectories from noisy, discretely sampled GPS observations matched against an underlying topological road network graph.

The system is deployed as a distributed, decoupled client-server architecture across two laptops:

```
                           LAPTOP 2 (Evaluator UI)
                     +----------------------------------+
                     |  Streamlit Dashboard             |
                     |  - Live Folium Map               |
                     |  - Dynamic Trip Selection        |
                     |  - KPI Cards & Evaluation Metrics|
                     +----------------------------------+
                                      |
                                      | HTTP REST
                         (Wi-Fi / LAN | Port 8000)
                                      v
                           LAPTOP 1 (Inference Server)
                     +----------------------------------+
                     |  FastAPI Service                 |
                     |  - GET /health, GET /trips       |
                     |  - POST /predict                 |
                     +----------------------------------+
                                      |
       +------------------------------+------------------------------+
       |                              |                              |
       v                              v                              v
[Data Loader]                 [Road Graph Network]         [Prediction Pipeline]
Loads .track, .nodes, .arcs   Constructs directed graph    - Local Projector (m)
Never touches .route          CSGraph Dijkstra routing     - k-NN Candidates
                                                           - HMM Viterbi Decoder
                                                           - Shortest-Path Route Fill
                                                                     |
                                                                     v
                                                          [Server-Side Evaluation]
                                                          Loads .route in isolation
                                                          Computes Precision, Recall,
                                                          F1, Route Continuity
```

---

## 2. Dataset Representation & Ingestion

Every trip package consists of four distinct whitespace-separated, headerless files:

| File Extension | Content Description | Column Specification | Indexing |
|---|---|---|---|
| `.track` | Raw GPS observations | `longitude`, `latitude`, `timestamp_seconds` | 1 Hz nominal (~97% exactly 1.0 s apart) |
| `.nodes` | Road intersection vertices | `longitude`, `latitude` | 0-based row index = `node_id` |
| `.arcs` | Directed road segments | `from_node_id`, `to_node_id` | 0-based row index = `arc_id` |
| `.route` | Ground-truth reference path | Chronological sequence of traversed `arc_id`s | Arc sequence (not 1:1 with GPS points) |

### Coordinate Conventions
* **Raw Files:** Column 0 = Longitude, Column 1 = Latitude.
* **Folium / Leaflet:** `[latitude, longitude]`.
* **Spatial Distance:** Raw angular degrees $(\Delta\text{lon}, \Delta\text{lat})$ are unsuitable for Euclidean calculations. The pipeline establishes an equirectangular tangent plane centered at the trip centroid:
  $$x = (\lambda - \lambda_0) \cdot \frac{\pi}{180} \cdot R \cdot \cos(\phi_0), \quad y = (\phi - \phi_0) \cdot \frac{\pi}{180} \cdot R$$
  where $R = 6,371,000\text{ m}$. All spatial calculations are executed in metric Cartesian space $(x, y)$.

---

## 3. Map-Matching Algorithm Formulation

### 3.1 Candidate Generation
For each observed GPS point $p_t \in \mathbb{R}^2$, candidate road segments are identified within a search radius $r$ (default $50\text{ m}$, expanding up to $200\text{ m}$ fallback):
* Each road segment $a = (u, v)$ is a directed line segment between node coordinates.
* The orthogonal projection $p'_t$ onto segment $a$ and perpendicular Euclidean distance $d(p_t, a) = \|p_t - p'_t\|$ are computed via vector projection:
  $$t = \text{clamp}\left(\frac{(p_t - u) \cdot (v - u)}{\|v - u\|^2}, 0, 1\right), \quad p'_t = u + t(v - u)$$
* The top $k$ nearest candidate arcs are retained per GPS observation.

### 3.2 Hidden Markov Model (HMM) Formulation
The vehicle's true road segment at time $t$ is modeled as a hidden state $S_t \in \{a_{t,1}, \dots, a_{t,k}\}$.

#### Emission Probability (Observation Likelihood)
Following Newson & Krumm (2009), the probability of observing GPS point $p_t$ given road candidate $a_{t,i}$ is modeled as a zero-mean Gaussian distribution on the perpendicular distance $d$:
$$p(p_t \mid S_t = a_{t,i}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{d(p_t, a_{t,i})^2}{2\sigma^2}\right)$$
In log-space:
$$\log p(p_t \mid a_{t,i}) = -0.5 \left(\frac{d(p_t, a_{t,i})}{\sigma}\right)^2 + C$$
The standard deviation $\sigma$ is estimated per trip using the Median Absolute Deviation (MAD) of the distance to the nearest road candidate: $\sigma = \text{clip}(1.4826 \cdot \text{median}(d_{\min}), 3.0, 30.0)\text{ m}$.

#### Transition Probability
The transition probability between candidate $a_{t-1, i}$ and candidate $a_{t, j}$ penalizes differences between the shortest topological road network distance $d_{\text{route}}$ and the straight-line Euclidean distance $d_{\text{great-circle}}$:
$$p(S_t = a_{t,j} \mid S_{t-1} = a_{t-1,i}) = \frac{1}{\beta} \exp\left(-\frac{|d_{\text{route}}(a_{t-1,i}, a_{t,j}) - d_{\text{gc}}(p_{t-1}, p_t)|}{\beta}\right)$$
In log-space:
$$\log p(a_{t,j} \mid a_{t-1,i}) = -\frac{|d_{\text{route}} - d_{\text{gc}}|}{\beta}$$
* $d_{\text{route}}$ is computed dynamically using Dijkstra's shortest path algorithm over the road graph.
* If two candidates are on disconnected components or unreachable within the distance bound, the transition score is $-\infty$.

### 3.3 Log-Space Viterbi Decoding & Chain Breaks
The most probable sequence of hidden states is found by maximizing the joint log-likelihood:
$$V_t(j) = \max_i \left[ V_{t-1}(i) + \log p(a_{t,j} \mid a_{t-1,i}) \right] + \log p(p_t \mid a_{t,j})$$
* **GPS Gaps & Missing Candidates:** If a GPS point has no candidates, it is skipped without breaking the temporal chain; the transition bridges from the last valid point $s$ directly to $t$.
* **Impossible Transitions:** If all incoming transitions into step $t$ evaluate to $-\infty$, a "chain break" occurs: the previous chain backtracks and finalizes, and a fresh Viterbi chain initiates at $t$ from emission scores.

### 3.4 Route Assembly & Continuity Completion
Once each GPS observation is matched to a discrete arc, the sequence of distinct arcs $[a_1, a_2, \dots, a_M]$ is assembled. If consecutive arcs are not topologically adjacent ($\text{to\_node}(a_m) \neq \text{from\_node}(a_{m+1})$), Dijkstra shortest path routing automatically inserts the intermediate connecting arcs, guaranteeing route continuity.

---

## 4. Bayesian Hyperparameter Optimization (Optuna)

To maximize route reconstruction fidelity, hyperparameter tuning was conducted using Bayesian Optimization (Tree-structured Parzen Estimator via Optuna):
* **Search Space:**
  * Candidate radius $r \in [30, 80]\text{ m}$
  * Candidate count $k \in [4, 16]$
  * Transition scale $\beta \in [5.0, 35.0]\text{ m}$
  * Sigma scale multiplier $\in [0.5, 2.0]$
  * Backtrack tolerance $\in [0, 20]\text{ m}$
* **Validation Strategy:** Group-aware validation over the 80 training trips, strictly isolating validation splits from candidate selection.
* **Tuned Configurations:**
  * `baseline_v1`: Default Newson & Krumm parameters ($k=8, r=50\text{ m}, \beta=15.0$).
  * `tuned_v2`: Bayesian-optimized profile deployed as default in `main.py`.

---

## 5. Frequently Asked Evaluator Questions (Viva Prep)

#### Q1: Why is a two-laptop distributed architecture required?
**Answer:** It simulates real-world production vehicle telematics. A fleet vehicle or mobile client (Laptop 2) collects noisy GPS observations and transmits queries over the network to a centralized map-matching inference service (Laptop 1) which hosts the heavy road network graph and trained decoders.

#### Q2: How is data leakage prevented for evaluation trips?
**Answer:** Strict structural isolation. The data loader module (`src.data_loader`) deliberately does not read `.route` files. Predictions are generated using only `.track`, `.nodes`, and `.arcs`. Reference routes are loaded strictly in `src.evaluation` *after* inference has concluded, and only summary performance metrics are returned to the UI.

#### Q3: Why can't we use simple Euclidean nearest-road snapping?
**Answer:** Independent nearest-road snapping (our geometric baseline) ignores trajectory dynamics and network topology. It frequently assigns consecutive GPS points to opposite sides of divided highways, parallel service roads, or one-way streets in the wrong direction. The HMM transition probability enforces topological connectivity and realistic travel distances.

#### Q4: What does Route Continuity measure?
**Answer:** It measures the proportion of consecutive predicted arcs where the end node of arc $i$ matches the start node of arc $i+1$: $\frac{1}{M-1} \sum \mathbb{I}(\text{to\_node}(a_i) == \text{from\_node}(a_{i+1}))$. A score of $1.0$ indicates an unbroken, drivable vehicle trajectory.
