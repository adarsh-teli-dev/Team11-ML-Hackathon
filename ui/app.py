"""
app.py - PathSnap Evaluator Interface (Role D: UI on Laptop 2)

Run on Laptop 2:
  streamlit run app.py
"""

import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="PathSnap — Team 11 Evaluator UI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/route.png", width=64)
st.sidebar.title("PathSnap Evaluator")
st.sidebar.caption("Team 11 • GPS-to-Road Map Matching")

api_url = st.sidebar.text_input(
    "API Server URL (Laptop 1)",
    value="http://localhost:8000",
    help="Enter the IP and port of Laptop 1, e.g. http://192.168.1.15:8000"
)

# Test API connection
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Trips List"):
    st.cache_data.clear()

@st.cache_data(ttl=10)
def fetch_trips(server_url: str):
    try:
        res = requests.get(f"{server_url}/api/trips", timeout=5)
        if res.status_code == 200:
            return res.json().get("trips", [])
        return []
    except Exception as e:
        return []

trips = fetch_trips(api_url)

if not trips:
    st.sidebar.error("⚠️ Cannot connect to API server. Ensure Laptop 1 API is running!")
    st.sidebar.info("Command to run on Laptop 1:\n```bash\nuvicorn main:app --host 0.0.0.0 --port 8000\n```")
    # Fallback default test trip IDs for demonstration
    trips = ["00000000", "00000007", "00000009", "00000040", "00000052", 
             "00000056", "00000065", "00000076", "00000077", "00000091"]
else:
    st.sidebar.success(f" Connected! {len(trips)} evaluation trips loaded.")

selected_trip = st.sidebar.selectbox("Select Reserved Trip ID", options=trips)

st.title("🗺️ PathSnap: Live Route Reconstruction & Evaluation")
st.markdown("Select an evaluation trip on the left and click **Run Evaluation** to query the inference engine.")

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_eval = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

if run_eval and selected_trip:
    with st.spinner(f"Requesting inference from Laptop 1 for trip {selected_trip}..."):
        try:
            resp = requests.post(f"{api_url}/api/predict/{selected_trip}", timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["result"] = data
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Failed to reach API at {api_url}: {e}")

# Render Results
if "result" in st.session_state and st.session_state["result"].get("trip_id") == selected_trip:
    data = st.session_state["result"]
    metrics = data.get("metrics", {})

    st.subheader(f"Evaluation Results for Trip: `{selected_trip}`")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        acc = metrics.get("top1_accuracy")
        val_acc = f"{acc * 100:.2f}%" if acc is not None else "N/A"
        st.metric("Top-1 Accuracy", val_acc, help="Observation-level accuracy against reference route")
    with m2:
        cont = metrics.get("route_continuity", 0.0)
        st.metric("Route Continuity", f"{cont * 100:.2f}%", help="Degree of road network connectivity")
    with m3:
        lat = metrics.get("inference_time_ms", 0.0)
        st.metric("Inference Latency", f"{lat:.1f} ms", help="Server-side inference execution time")
    with m4:
        length = metrics.get("predicted_route_length", 0)
        st.metric("Predicted Arcs", f"{length}", help="Number of road segments in reconstructed route")

    # Interactive Map Section
    st.markdown("### 🗺️ Geographical Route Reconstruction")
    gps_display = data.get("gps_display", [])
    route_coords = data.get("route_coords", [])

    if gps_display:
        avg_lat = sum(p["lat"] for p in gps_display) / len(gps_display)
        avg_lon = sum(p["lon"] for p in gps_display) / len(gps_display)

        # Create Folium Map
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="CartoDB positron")

        # Layer 1: GPS Observations
        gps_group = folium.FeatureGroup(name="GPS Observations (Raw Track)", show=True)
        for pt in gps_display:
            folium.CircleMarker(
                location=[pt["lat"], pt["lon"]],
                radius=3,
                color="#e74c3c",
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.7,
                popup=f"Obs #{pt.get('step')}"
            ).add_to(gps_group)
        gps_group.add_to(m)

        # Layer 2: Predicted Route
        if route_coords:
            route_group = folium.FeatureGroup(name="Reconstructed Route (Predicted Arcs)", show=True)
            folium.PolyLine(
                locations=route_coords,
                color="#2ecc71",
                weight=5,
                opacity=0.85,
                tooltip="Predicted Road Route"
            ).add_to(route_group)
            route_group.add_to(m)

        folium.LayerControl().add_to(m)

        st_folium(m, width="100%", height=520)
    else:
        st.warning("No GPS coordinate data returned for map plotting.")

    # Route Arc Sequence Preview
    with st.expander("🔍 View Predicted Arc Sequence (first 100 segments)"):
        pred_route = data.get("predicted_route", [])
        st.write(pred_route[:100])
        st.caption(f"Total {len(pred_route)} arcs predicted for {metrics.get('observations_count')} GPS observations.")
else:
    st.info("👈 Click **Run Evaluation** in the sidebar or above to test the selected trip.")
