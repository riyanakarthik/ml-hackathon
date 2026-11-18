"""PathSnap — GPS Route Reconstruction: Evaluator Dashboard.

A lightweight Streamlit + Folium web application that connects over the
network to the PathSnap FastAPI route-reconstruction backend (Laptop 1,
port 8000).  Every trip choice, prediction result, metric and coordinate
displayed in this dashboard is fetched **live** from the backend API —
nothing is hardcoded or cached locally.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import math
from typing import Any

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PathSnap — GPS Route Reconstruction",
    layout="wide",
    page_icon="🗺️",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "backend_url": "http://localhost:8000",
    "health_status": None,          # None | dict | str(error)
    "trips": [],
    "trips_error": None,
    "selected_trip": None,
    "prediction_result": None,
    "prediction_error": None,
    "_last_checked_url": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Helpers — API calls
# ---------------------------------------------------------------------------

def _norm_url(url: str) -> str:
    """Strip trailing slashes from the backend URL."""
    return url.rstrip("/")


def check_health(base_url: str) -> dict | str:
    """GET /health → dict on success, error string on failure."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return "Connection refused – is the FastAPI backend running and reachable?"
    except requests.exceptions.Timeout:
        return "Request timed out – check the network path to Laptop 1."
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        return f"HTTP {exc.response.status_code}: {detail or exc}"
    except ValueError:
        return "Invalid JSON in /health response."


def fetch_trips(base_url: str) -> list[str] | str:
    """GET /trips → list of trip-id strings, or error string."""
    try:
        r = requests.get(f"{base_url}/trips", timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.ConnectionError:
        return "Cannot reach backend – connection refused."
    except requests.exceptions.Timeout:
        return "Timed out fetching trips."
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        return f"HTTP {exc.response.status_code}: {detail or exc}"
    except ValueError:
        return "Invalid JSON in /trips response."

    # Flexible response shapes
    if isinstance(data, list):
        return [str(t) for t in data]
    if isinstance(data, dict):
        for key in ("trips", "trip_ids"):
            if key in data and isinstance(data[key], list):
                return [str(t) for t in data[key]]
        # list of objects with trip_id
        if isinstance(data.get("trips"), list):
            return [str(t) for t in data["trips"]]
        # maybe list of dicts at top level
    return "Unexpected /trips response shape."


def run_prediction(base_url: str, trip_id: str) -> dict | str:
    """POST /predict → result dict, or error string."""
    try:
        r = requests.post(
            f"{base_url}/predict",
            json={"trip_id": trip_id},
            timeout=120,
        )
        r.raise_for_status()
        payload = r.json()
    except requests.exceptions.ConnectionError:
        return "Cannot reach backend – connection refused."
    except requests.exceptions.Timeout:
        return "Prediction timed out (>120 s)."
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        return f"HTTP {exc.response.status_code}: {detail or exc}"
    except ValueError:
        return "Invalid JSON in /predict response."

    # Validate required keys
    required = (
        "trip_id",
        "gps_coordinates",
        "predicted_coordinates",
        "predicted_arc_ids",
        "metrics",
        "method",
        "runtime_ms",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        return f"Response missing required keys: {missing}"
    return payload


# ---------------------------------------------------------------------------
# Helpers — coordinate sanitization & map building
# ---------------------------------------------------------------------------

def _valid_coord(pair: Any) -> tuple[float, float] | None:
    """Return (lat, lon) if the pair is a valid finite coordinate."""
    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return None
    try:
        lat, lon = float(pair[0]), float(pair[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return (lat, lon)


def sanitize_coords(raw: list) -> list[tuple[float, float]]:
    """Filter a coordinate list to valid [lat, lon] pairs."""
    out: list[tuple[float, float]] = []
    for p in raw:
        c = _valid_coord(p)
        if c is not None:
            out.append(c)
    return out


def build_route_map(
    gps_coords: list[tuple[float, float]],
    pred_coords: list[tuple[float, float]],
    trip_id: str,
) -> folium.Map:
    """Build an interactive Folium map with GPS observations & predicted route."""
    # Compute bounding box across all points
    all_pts = gps_coords + pred_coords
    if not all_pts:
        # Fallback: world centre
        return folium.Map(location=[0, 0], zoom_start=2)

    lats = [p[0] for p in all_pts]
    lons = [p[1] for p in all_pts]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    centre = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    fmap = folium.Map(location=centre, tiles="CartoDB positron")

    # Additional tile layer
    folium.TileLayer("OpenStreetMap").add_to(fmap)

    # --- Layer 1: Noisy GPS Observations ---
    gps_group = folium.FeatureGroup(name="Noisy GPS Observations")
    for idx, (lat, lon) in enumerate(gps_coords):
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#1d4ed8",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.85,
            tooltip=f"GPS #{idx}  ({lat:.6f}, {lon:.6f})",
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    # --- Layer 2: Predicted Route ---
    pred_group = folium.FeatureGroup(name="Predicted Route")
    if len(pred_coords) >= 2:
        folium.PolyLine(
            locations=pred_coords,
            color="#16a34a",
            weight=5,
            opacity=0.9,
            tooltip="Predicted Route",
        ).add_to(pred_group)
    if pred_coords:
        # Start marker
        folium.Marker(
            location=pred_coords[0],
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
            tooltip="Route Start",
        ).add_to(pred_group)
    if len(pred_coords) >= 2:
        # End marker
        folium.Marker(
            location=pred_coords[-1],
            icon=folium.Icon(color="red", icon="stop", prefix="fa"),
            tooltip="Route End",
        ).add_to(pred_group)
    pred_group.add_to(fmap)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(fmap)

    # Legend
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 10px 14px; border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,.3); font-size: 13px;
        line-height: 1.6;">
      <b>Legend</b><br>
      <span style="color:#2563eb;">&#9679;</span> Noisy GPS Observations<br>
      <span style="color:#16a34a;">&#9644;</span> Predicted Route
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Fit bounds with padding — handle single-point / identical-coord edge case
    if max_lat - min_lat < 1e-6 and max_lon - min_lon < 1e-6:
        fmap.location = [min_lat, min_lon]
        fmap.zoom_start = 16
    else:
        fmap.fit_bounds(
            [[min_lat, min_lon], [max_lat, max_lon]], padding=(30, 30)
        )

    return fmap


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🗺️ PathSnap")
    st.caption("Evaluator Dashboard")

    st.subheader("Backend Connection")
    url_input = st.text_input(
        "Backend URL",
        value=st.session_state["backend_url"],
        placeholder="http://<laptop-1-ip>:8000",
    )
    st.session_state["backend_url"] = _norm_url(url_input)
    base = st.session_state["backend_url"]

    # Auto-check on first load or URL change
    if st.session_state["_last_checked_url"] != base:
        st.session_state["health_status"] = check_health(base)
        trips_result = fetch_trips(base)
        if isinstance(trips_result, list):
            st.session_state["trips"] = trips_result
            st.session_state["trips_error"] = None
        else:
            st.session_state["trips"] = []
            st.session_state["trips_error"] = trips_result
        st.session_state["_last_checked_url"] = base

    # Manual health check
    if st.button("🔍 Check Backend Status"):
        st.session_state["health_status"] = check_health(base)

    health = st.session_state["health_status"]
    if isinstance(health, dict):
        st.success(f"✅ Backend online — {health}")
    elif isinstance(health, str):
        st.error(f"❌ {health}")
        st.info("💡 Ensure the FastAPI server is running: "
                "`uvicorn main:app --host 0.0.0.0 --port 8000`")
    else:
        st.warning("Backend status unknown — click **Check Backend Status**.")

    st.divider()
    st.subheader("Trip Selection")

    if st.button("🔄 Refresh Trips"):
        trips_result = fetch_trips(base)
        if isinstance(trips_result, list):
            st.session_state["trips"] = trips_result
            st.session_state["trips_error"] = None
        else:
            st.session_state["trips"] = []
            st.session_state["trips_error"] = trips_result

    trips_list = st.session_state["trips"]
    trips_err = st.session_state["trips_error"]

    if trips_err:
        st.error(trips_err)
    if not trips_list:
        st.warning("No trips available — check backend connection.")

    selected = st.selectbox(
        "Select Evaluation Trip",
        options=trips_list if trips_list else [],
        disabled=not trips_list,
        index=0 if trips_list else 0,
    )
    st.session_state["selected_trip"] = selected

    st.divider()

    run_disabled = not selected or not trips_list
    if st.button(
        "🚀 Run Route Reconstruction",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    ):
        # Clear stale errors
        st.session_state["prediction_error"] = None
        st.session_state["prediction_result"] = None
        with st.spinner(f"Running prediction for trip **{selected}** …"):
            result = run_prediction(base, selected)
        if isinstance(result, dict):
            st.session_state["prediction_result"] = result
            st.session_state["prediction_error"] = None
        else:
            st.session_state["prediction_error"] = result
            st.session_state["prediction_result"] = None


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("PathSnap — GPS Route Reconstruction")
st.caption("Interactive evaluator dashboard · connects to FastAPI inference backend")

result = st.session_state["prediction_result"]
pred_err = st.session_state["prediction_error"]

if pred_err:
    st.error(f"Prediction failed: {pred_err}")

if result is None:
    st.info(
        "👈 Select an evaluation trip in the sidebar and click "
        "**Run Route Reconstruction** to begin."
    )
    st.stop()

# ---------------------------------------------------------------------------
# KPI strip  (guarded so test-time import with mocked st.stop is safe)
# ---------------------------------------------------------------------------
if result is not None:
    trip_id = str(result.get("trip_id", ""))
    method = str(result.get("method", "—"))
    runtime = result.get("runtime_ms", 0)
    gps_raw = result.get("gps_coordinates", [])
    pred_raw = result.get("predicted_coordinates", [])
    arc_ids = result.get("predicted_arc_ids", [])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trip ID", trip_id)
    k2.metric("Method", method)
    k3.metric("Runtime", f"{runtime:.1f} ms")
    k4.metric("Points / Arcs", f"{len(gps_raw)} obs · {len(arc_ids)} arcs")

    # -------------------------------------------------------------------
    # Evaluation metrics
    # -------------------------------------------------------------------
    metrics = result.get("metrics")
    if isinstance(metrics, dict) and metrics:
        st.subheader("Returned Evaluation Metrics")
        keys = list(metrics.keys())
        cols = st.columns(min(len(keys), 4))
        for i, key in enumerate(keys):
            col = cols[i % len(cols)]
            val = metrics[key]
            if isinstance(val, float):
                col.metric(key, f"{val:.4f}")
            else:
                col.metric(key, str(val) if val is not None else "—")
    elif metrics is None:
        st.info("No ground-truth metrics available for this trip "
                "(no `.route` reference file).")
    else:
        st.info("Metrics field is empty or not a dictionary.")

    # -------------------------------------------------------------------
    # Interactive Folium map
    # -------------------------------------------------------------------
    st.subheader("Route Map")
    gps_clean = sanitize_coords(gps_raw)
    pred_clean = sanitize_coords(
        [p for p in pred_raw if p is not None]  # filter nulls
    )

    route_map = build_route_map(gps_clean, pred_clean, trip_id)
    st_folium(
        route_map,
        width=None,
        height=560,
        returned_objects=[],
        key=f"map_{trip_id}",
    )

    # -------------------------------------------------------------------
    # Raw output expander
    # -------------------------------------------------------------------
    with st.expander("Predicted Arc IDs & Raw API Response"):
        st.write("**Predicted Arc IDs**")
        st.write(arc_ids)
        st.write("**Full API Response**")
        st.json(result)

