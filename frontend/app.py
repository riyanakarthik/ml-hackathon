"""PathSnap — GPS Route Reconstruction: Evaluator Dashboard.

A sleek, lightweight map-matching console connecting over LAN to the PathSnap
FastAPI backend on Laptop 1 (port 8000).  Strictly zero hardcoded data: every
trip choice, coordinate, and metric is fetched live from the backend API.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

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
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "backend_url": "http://localhost:8000",
    "health_status": None,
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
# Helpers — API calls (strictly adhering to backend contract)
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


def _extract_trip_id(item: Any) -> str | None:
    """Extract a trip_id string from an item (dict, string, or int)."""
    if isinstance(item, dict):
        for k in ("trip_id", "id", "name"):
            if k in item and item[k] is not None:
                return str(item[k]).strip()
        return None
    if isinstance(item, (str, int)):
        s = str(item).strip()
        return s if s else None
    return None


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

    # Flexible response shapes (list of ids, list of objects, wrapped dicts)
    raw_items: list | None = None
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        for key in ("trips", "trip_ids", "data", "items"):
            if key in data and isinstance(data[key], list):
                raw_items = data[key]
                break

    if raw_items is not None:
        out: list[str] = []
        for item in raw_items:
            tid = _extract_trip_id(item)
            if tid:
                out.append(tid)
        return out

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


# Aliases for backward compatibility
check_backend_health = check_health
fetch_available_trips = fetch_trips
request_route_prediction = run_prediction


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


sanitize_coordinates = sanitize_coords


def build_route_map(
    gps_coords: list[tuple[float, float]],
    pred_coords: list[tuple[float, float]],
    trip_id: str = "",
    metrics: dict | None = None,
    runtime_ms: float | None = None,
    arc_count: int | None = None,
) -> folium.Map:
    """Build an interactive Folium map with GPS observations & predicted route.

    Default base tile is OpenStreetMap so roads, highways, parks, and labels
    are clearly visible, with OpenStreetMap as an alternate layer.
    Includes dark floating Leaflet layer controls and bottom-left summary card.
    """
    all_pts = gps_coords + pred_coords
    if not all_pts:
        return folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")

    lats = [p[0] for p in all_pts]
    lons = [p[1] for p in all_pts]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    centre = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    fmap = folium.Map(location=centre, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)

    # --- Layer 1: HMM / Viterbi route (bold royal blue polyline) ---
    pred_layer_name = '<span style="display:inline-block;width:12px;height:4px;background:#2563eb;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>HMM / Viterbi route'
    pred_group = folium.FeatureGroup(name=pred_layer_name, show=True)
    if len(pred_coords) >= 2:
        folium.PolyLine(
            locations=pred_coords,
            color="#2563eb",
            weight=6,
            opacity=0.95,
            tooltip=f"Trip {trip_id} — Predicted Route",
        ).add_to(pred_group)

    # Waypoint markers: crisp white circles with dark border
    if pred_coords:
        folium.CircleMarker(
            location=pred_coords[0],
            radius=7,
            color="#0f172a",
            weight=3,
            fill=True,
            fill_color="#ffffff",
            fill_opacity=1.0,
            tooltip="Route Start",
        ).add_to(pred_group)

    if len(pred_coords) >= 2:
        folium.CircleMarker(
            location=pred_coords[-1],
            radius=7,
            color="#0f172a",
            weight=3,
            fill=True,
            fill_color="#ffffff",
            fill_opacity=1.0,
            tooltip="Route End",
        ).add_to(pred_group)
        # Blue center accent on destination
        folium.CircleMarker(
            location=pred_coords[-1],
            radius=3,
            color="#2563eb",
            weight=1,
            fill=True,
            fill_color="#2563eb",
            fill_opacity=1.0,
            tooltip="Route Destination",
        ).add_to(pred_group)

    pred_group.add_to(fmap)

    # --- Layer 2: Raw GPS fixes (translucent light dots) ---
    gps_layer_name = '<span style="display:inline-block;width:7px;height:7px;background:#ffffff;border:1px solid #94a3b8;border-radius:50%;vertical-align:middle;margin-right:6px;"></span>Raw GPS fixes'
    gps_group = folium.FeatureGroup(name=gps_layer_name, show=True)
    for idx, (lat, lon) in enumerate(gps_coords):
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#94a3b8",
            weight=1,
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.75,
            tooltip=f"GPS #{idx} ({lat:.6f}, {lon:.6f})",
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    # Layer control (styled dark via injected CSS)
    folium.LayerControl(collapsed=False).add_to(fmap)

    # --- Floating dark bottom-left summary card & dark layer control CSS ---
    f1_val = None
    cont_val = None
    if isinstance(metrics, dict):
        f1_val = metrics.get("arc_f1") or metrics.get("macro_f1")
        cont_val = metrics.get("route_continuity")

    f1_str = f"{f1_val * 100:.1f}%" if isinstance(f1_val, (int, float)) else "N/A"
    cont_str = f"{cont_val * 100:.1f}%" if isinstance(cont_val, (int, float)) else "N/A"
    rt_str = f"{runtime_ms:,.0f} ms" if isinstance(runtime_ms, (int, float)) else "—"
    arcs_display = f"{arc_count:,} road arcs matched" if arc_count is not None else "Route matching active"

    overlay_html = f"""
    <style>
    .leaflet-control-layers {{
        background: rgba(17, 20, 26, 0.92) !important;
        color: #f8fafc !important;
        border: 1px solid #2a303c !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        box-shadow: 0 10px 28px rgba(0,0,0,0.5) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-size: 12px !important;
    }}
    .leaflet-control-layers-base label, .leaflet-control-layers-overlays label {{
        color: #f8fafc !important;
        margin-bottom: 4px !important;
        display: flex !important;
        align-items: center !important;
    }}
    .leaflet-control-layers-separator {{
        border-top: 1px solid #2a303c !important;
        margin: 6px 0 !important;
    }}
    </style>
    <div style="
        position: fixed; bottom: 26px; left: 24px; z-index: 1000;
        background: rgba(17, 20, 26, 0.94);
        border: 1px solid #2a303c;
        border-radius: 14px;
        padding: 14px 18px;
        color: #ffffff;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.55);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        min-width: 260px;
        backdrop-filter: blur(8px);
    ">
      <div style="font-weight: 700; font-size: 18px; line-height: 1.2; color: #ffffff;">Trip {trip_id or '—'}</div>
      <div style="color: #94a3b8; font-size: 12px; margin-top: 3px; margin-bottom: 12px;">
        {len(gps_coords):,} GPS fixes · {arcs_display}
      </div>
      <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
        <span style="background: #1e232d; color: #38bdf8; border: 1px solid #2a3240; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600;">F1 {f1_str}</span>
        <span style="background: #1e232d; color: #4ade80; border: 1px solid #2a3240; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600;">Continuity {cont_str}</span>
        <span style="background: #1e232d; color: #e2e8f0; border: 1px solid #2a3240; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600;">{rt_str}</span>
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(overlay_html))

    # Fit map bounds
    if max_lat - min_lat < 1e-6 and max_lon - min_lon < 1e-6:
        fmap.location = [min_lat, min_lon]
        fmap.zoom_start = 16
    else:
        fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(30, 30))

    return fmap


# ---------------------------------------------------------------------------
# UI Rendering (sleek dark console + full-bleed Folium map)
# ---------------------------------------------------------------------------

def render_app():
    # Inject lightweight CSS for sleek dark theme & zero excess whitespace
    st.markdown(
        """
        <style>
        /* Hide default Streamlit headers and footers */
        header[data-testid="stHeader"] {display: none !important;}
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        div[data-testid="stToolbar"] {display: none !important;}
        div[data-testid="stDecoration"] {display: none !important;}
        div[data-testid="stStatusWidget"] {display: none !important;}

        /* Full-width block container */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0.8rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
            max-width: 100% !important;
        }

        /* Dark console styling */
        div[data-testid="column"]:first-child {
            background-color: #0b0d10;
            border-radius: 16px;
            padding: 6px 10px;
        }

        /* Selected trip button: solid crisp white */
        div[data-testid="stBaseButton-primary"] > button {
            background-color: #ffffff !important;
            color: #0b0d10 !important;
            font-weight: 700 !important;
            border: 1px solid #ffffff !important;
            border-radius: 9px !important;
            box-shadow: 0 2px 8px rgba(255,255,255,0.18) !important;
            transition: all 0.15s ease-in-out !important;
        }
        div[data-testid="stBaseButton-primary"] > button:hover {
            background-color: #f1f5f9 !important;
            color: #0b0d10 !important;
        }

        /* Unselected trip button: rounded dark card */
        div[data-testid="stBaseButton-secondary"] > button {
            background-color: #161920 !important;
            color: #cbd5e1 !important;
            font-weight: 500 !important;
            border: 1px solid #262b36 !important;
            border-radius: 9px !important;
            transition: all 0.15s ease-in-out !important;
        }
        div[data-testid="stBaseButton-secondary"] > button:hover {
            background-color: #202530 !important;
            border-color: #3b4252 !important;
            color: #ffffff !important;
        }

        /* Dark input styling */
        div[data-testid="stTextInput"] input {
            background-color: #14171d !important;
            color: #f8fafc !important;
            border: 1px solid #262b36 !important;
            border-radius: 8px !important;
            font-size: 13px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    base = _norm_url(st.session_state["backend_url"])

    # Auto check backend health & trips on initial load or URL change
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

    trips = st.session_state["trips"]
    health = st.session_state["health_status"]
    is_online = isinstance(health, dict)

    # Auto-select and auto-run first trip on initial load
    if trips and not st.session_state["selected_trip"]:
        st.session_state["selected_trip"] = trips[0]

    if (
        st.session_state["selected_trip"]
        and st.session_state["prediction_result"] is None
        and not st.session_state["prediction_error"]
        and is_online
    ):
        with st.spinner(f"Matching trip {st.session_state['selected_trip']} ..."):
            res = run_prediction(base, st.session_state["selected_trip"])
            if isinstance(res, dict):
                st.session_state["prediction_result"] = res
                st.session_state["prediction_error"] = None
            else:
                st.session_state["prediction_error"] = res
                st.session_state["prediction_result"] = None

    # Two-column layout: Left console (1.05) + Right full-bleed Folium map (2.35)
    col_console, col_map = st.columns([1.05, 2.35], gap="small")

    # =======================================================================
    # LEFT COLUMN: Dark Console Panel
    # =======================================================================
    with col_console:
        # 1. Header Bar with Logo and API Live/Offline Badge
        status_pill_html = (
            """<div style="display:inline-flex;align-items:center;gap:6px;background:#0f291e;color:#4ade80;border:1px solid #1e4634;padding:4px 10px;border-radius:9999px;font-size:12px;font-weight:600;"><span style="font-size:9px;">●</span> API live</div>"""
            if is_online
            else """<div style="display:inline-flex;align-items:center;gap:6px;background:#2c1215;color:#f87171;border:1px solid #4c1d24;padding:4px 10px;border-radius:9999px;font-size:12px;font-weight:600;"><span style="font-size:9px;">●</span> API offline</div>"""
        )

        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
              <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: #ffffff; color: #0b0d10; font-weight: 900; font-size: 15px; width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; letter-spacing: -0.5px;">PS</div>
                <div>
                  <div style="color: #ffffff; font-weight: 700; font-size: 18px; line-height: 1.1;">PathSnap</div>
                  <div style="color: #94a3b8; font-size: 12px; font-weight: 500;">Map-matching console</div>
                </div>
              </div>
              {status_pill_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Compact LAN Backend Connection Bar
        bcol1, bcol2 = st.columns([3.2, 1.2], gap="small")
        with bcol1:
            new_url = st.text_input(
                "Backend URL",
                value=st.session_state["backend_url"],
                placeholder="http://<laptop-1-ip>:8000",
                label_visibility="collapsed",
            )
            st.session_state["backend_url"] = _norm_url(new_url)
        with bcol2:
            if st.button("Connect", use_container_width=True):
                st.session_state["_last_checked_url"] = None
                st.rerun()

        # 3. 2-Column Interactive Trip Button Grid
        st.markdown(
            """<div style="color: #64748b; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 12px; margin-bottom: 6px;">SELECT EVALUATION TRIP</div>""",
            unsafe_allow_html=True,
        )

        if not trips:
            if st.session_state["trips_error"]:
                st.error(f"Cannot load trips: {st.session_state['trips_error']}")
            else:
                st.warning("No trips available from backend.")
        else:
            tcol1, tcol2 = st.columns(2, gap="small")
            for idx, tid in enumerate(trips):
                col = tcol1 if idx % 2 == 0 else tcol2
                is_selected = tid == st.session_state["selected_trip"]
                btn_label = f"{tid}   #{idx + 1}"
                if col.button(
                    btn_label,
                    key=f"trip_btn_{tid}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["selected_trip"] = tid
                    with st.spinner(f"Matching trip {tid} ..."):
                        res = run_prediction(base, tid)
                        if isinstance(res, dict):
                            st.session_state["prediction_result"] = res
                            st.session_state["prediction_error"] = None
                        else:
                            st.session_state["prediction_error"] = res
                            st.session_state["prediction_result"] = None
                    st.rerun()

        # 4. Live Status Box
        result = st.session_state["prediction_result"]
        pred_err = st.session_state["prediction_error"]
        sel_trip = st.session_state["selected_trip"] or "—"

        if pred_err:
            st.markdown(
                f"""
                <div style="background: #231215; border: 1px solid #4a1f26; border-radius: 10px; padding: 10px 14px; margin-top: 12px; margin-bottom: 14px; color: #fca5a5; font-size: 12px;">
                  ⚠️ {pred_err}
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif result:
            n_fixes = len(result.get("gps_coordinates", []))
            st.markdown(
                f"""
                <div style="background: #161920; border: 1px solid #262b36; border-radius: 10px; padding: 10px 14px; margin-top: 12px; margin-bottom: 14px; color: #e2e8f0; font-size: 13px;">
                  <span style="color: #38bdf8; font-weight: 600;">Trip {result.get('trip_id', sel_trip)}</span> done – <b>{n_fixes:,}</b> fixes matched by the live API.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: #161920; border: 1px solid #262b36; border-radius: 10px; padding: 10px 14px; margin-top: 12px; margin-bottom: 14px; color: #94a3b8; font-size: 12px;">
                  Select a trip above to run route reconstruction.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 5. ROUTE QUALITY 2x2 Metric Card Grid
        st.markdown(
            """<div style="color: #64748b; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;">ROUTE QUALITY</div>""",
            unsafe_allow_html=True,
        )

        metrics = result.get("metrics") if isinstance(result, dict) else None
        method_str = str(result.get("method", "—")) if isinstance(result, dict) else "—"
        arc_ids_list = result.get("predicted_arc_ids", []) if isinstance(result, dict) else []

        # Strictly extract metrics returned from backend
        prec_val = None
        cont_val = None
        f1_val = None
        rec_val = None
        if isinstance(metrics, dict):
            prec_val = metrics.get("arc_precision") or metrics.get("top1_accuracy")
            cont_val = metrics.get("route_continuity")
            f1_val = metrics.get("arc_f1") or metrics.get("macro_f1")
            rec_val = metrics.get("arc_recall")

        def _fmt_pct(v: Any) -> str:
            if isinstance(v, (int, float)):
                return f"{v * 100:.1f}%" if v <= 1.0 else f"{v:.1f}%"
            return "—"

        prec_str = _fmt_pct(prec_val)
        cont_str = _fmt_pct(cont_val)
        f1_str = _fmt_pct(f1_val)
        rec_str = _fmt_pct(rec_val)

        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
              <!-- Card 1: Arc Precision -->
              <div style="background: #15181e; border: 1px solid #232832; border-radius: 12px; padding: 12px 14px;">
                <div style="color: #94a3b8; font-size: 12px; font-weight: 500; margin-bottom: 4px;">Arc Precision</div>
                <div style="color: #ffffff; font-size: 22px; font-weight: 700; line-height: 1.1;">{prec_str}</div>
                <div style="color: #64748b; font-size: 11px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">method: {method_str}</div>
              </div>

              <!-- Card 2: Route Continuity -->
              <div style="background: #15181e; border: 1px solid #232832; border-radius: 12px; padding: 12px 14px;">
                <div style="color: #94a3b8; font-size: 12px; font-weight: 500; margin-bottom: 4px;">Route Continuity</div>
                <div style="color: #ffffff; font-size: 22px; font-weight: 700; line-height: 1.1;">{cont_str}</div>
                <div style="color: #64748b; font-size: 11px; margin-top: 4px;">connected path ratio</div>
              </div>

              <!-- Card 3: Route Arc F1 -->
              <div style="background: #15181e; border: 1px solid #232832; border-radius: 12px; padding: 12px 14px;">
                <div style="color: #94a3b8; font-size: 12px; font-weight: 500; margin-bottom: 4px;">Route Arc F1</div>
                <div style="color: #ffffff; font-size: 22px; font-weight: 700; line-height: 1.1;">{f1_str}</div>
                <div style="color: #64748b; font-size: 11px; margin-top: 4px;">vs reference route</div>
              </div>

              <!-- Card 4: Arc Recall -->
              <div style="background: #15181e; border: 1px solid #232832; border-radius: 12px; padding: 12px 14px;">
                <div style="color: #94a3b8; font-size: 12px; font-weight: 500; margin-bottom: 4px;">Arc Recall</div>
                <div style="color: #ffffff; font-size: 22px; font-weight: 700; line-height: 1.1;">{rec_str}</div>
                <div style="color: #64748b; font-size: 11px; margin-top: 4px;">{len(arc_ids_list):,} arcs matched</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 6. Console Footer
        st.markdown(
            """
            <div style="margin-top: 18px; color: #475569; font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-align: center; text-transform: uppercase;">
              ALL METRICS · COMPUTED SERVER-SIDE
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Raw Output Inspection Expander
        if result:
            with st.expander("Predicted Arc IDs & Raw API Payload", expanded=False):
                st.write("**Predicted Arc IDs**")
                st.write(result.get("predicted_arc_ids", []))
                st.write("**Full API Response**")
                st.json(result)

    # =======================================================================
    # RIGHT COLUMN: Full-Height Interactive Folium Map
    # =======================================================================
    with col_map:
        if result:
            gps_clean = sanitize_coords(result.get("gps_coordinates", []))
            pred_raw = result.get("predicted_coordinates", [])
            pred_clean = sanitize_coords([p for p in pred_raw if p is not None])
            current_tid = str(result.get("trip_id", sel_trip))

            route_map = build_route_map(
                gps_clean,
                pred_clean,
                trip_id=current_tid,
                metrics=result.get("metrics"),
                runtime_ms=result.get("runtime_ms"),
                arc_count=len(result.get("predicted_arc_ids", [])),
            )
            st_folium(
                route_map,
                width=None,
                height=740,
                returned_objects=[],
                key=f"map_{current_tid}",
            )
        else:
            # Fallback placeholder map
            fallback_map = folium.Map(location=[53.9, 27.5], zoom_start=11, tiles="OpenStreetMap")
            folium.TileLayer("OpenStreetMap").add_to(fallback_map)
            st_folium(
                fallback_map,
                width=None,
                height=740,
                returned_objects=[],
                key="map_placeholder",
            )


# Safeguard so contract tests importing app module do not run UI render on import
if not isinstance(st, MagicMock):
    render_app()
