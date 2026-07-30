import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import datetime
import folium
import pandas as pd
from streamlit_folium import st_folium

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide", page_icon="❄️")

# --- 2. PROFESSIONAL CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .header-box {
        background: linear-gradient(90deg, #1e3a8a, #111827);
        padding: 30px; border-radius: 15px; text-align: center;
        border: 1px solid #1e40af; margin-bottom: 25px;
    }
    .main-title { color: #60a5fa; font-size: 38px; font-weight: 800; margin: 0; }
    [data-testid="stMetric"] { background-color: #161e2e; border: 1px solid #1f2937; border-radius: 12px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ROBUST MODEL LOADER (FIXED for PyTorch 2.6 weights_only issue) ---
@st.cache_resource
def load_yolo_model():
    import torch

    # PyTorch 2.6 changed torch.load default to weights_only=True, which blocks
    # ultralytics' internal classes (Sequential, DetectionModel, etc.) from loading.
    # Since this is our own trained checkpoint, it's safe to force weights_only=False.
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    possible_paths = [
        "best.pt",
        os.path.join(os.getcwd(), "best.pt"),
        "/mount/src/saskatoon-cryospheric-road-ml/best.pt"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                return YOLO(path)
            except Exception as e:
                st.error(f"Model error at {path}: {e}")
                return None

    st.error("❌ 'best.pt' file nahi mila in any of these paths: " + ", ".join(possible_paths))
    return None

model = load_yolo_model()

# --- 4. SIDEBAR CONTROL CENTER ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>❄️</h1>", unsafe_allow_html=True)
    st.title("Control Center")
    st.metric("City", "Saskatoon, SK", "-14°C")
    st.write("---")
    st.subheader("⚙️ Settings")
    conf_val = st.slider("AI Sensitivity", 0.1, 1.0, 0.45)
    st.write("---")
    st.success("Developer: Agha Wafa Abbas")
    st.info("Dataset: DAWN Environment")

# --- 5. MAIN INTERFACE ---
st.markdown('<div class="header-box"><p class="main-title">SASKATOON ROAD SAFETY AI</p></div>', unsafe_allow_html=True)

if model is None:
    st.error("❌ ERROR: 'best.pt' not detected! Please ensure the file is in your main GitHub folder.")
    st.info("Tip: Make sure the file is named exactly 'best.pt' (case-sensitive) and not hidden inside folders.")
else:
    uploaded_file = st.file_uploader("📸 Upload Road Snapshot", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        with st.spinner("🧠 AI Engine Analyzing..."):
            results = model.predict(source=img, conf=conf_val)
            res_plotted = results[0].plot()

            # Keep as a clean uint8 numpy array; let st.image handle BGR->RGB directly (bypasses PIL entirely)
            import numpy as np
            display_img = np.ascontiguousarray(res_plotted).astype(np.uint8)

            # --- Evaluation Data Extraction ---
            boxes = results[0].boxes
            raw_labels = [model.names[int(box.cls)].lower() for box in boxes]
            confidences = [float(box.conf) for box in boxes]

            # Clean Labels Logic
            final_display_labels = set()
            detailed_data = []

            for label, conf in zip(raw_labels, confidences):
                display_name = label.capitalize()
                if label == 'sand':
                    display_name = "Low Visibility (Fog/Salt)"
                    final_display_labels.add("Low Visibility (Fog/Salt)")
                elif label == 'snow':
                    if 'sand' not in raw_labels:
                        final_display_labels.add("Snowy Conditions")
                        display_name = "Snowy Conditions"
                else:
                    final_display_labels.add(label.capitalize())

                detailed_data.append({"Object/Hazard": display_name, "Confidence Score": f"{conf:.2f}"})

        # --- 6. RESULTS LAYOUT ---
        col_vis, col_dash = st.columns([1.6, 1])

        with col_vis:
            st.subheader("🔍 Perception View")
            st.image(display_img, channels="BGR", use_container_width=True)

            # Detailed AI Evaluation Table
            if detailed_data:
                st.subheader("📊 Detailed AI Evaluation Table")
                df = pd.DataFrame(detailed_data)
                st.dataframe(df, use_container_width=True)

        with col_dash:
            st.subheader("🛡️ Safety Dashboard")

            # Advanced Live Evaluation Metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Total Hazards", len(boxes))
            with m_col2:
                avg_conf = sum(confidences)/len(confidences) if confidences else 0.0
                st.metric("Avg Conf Score", f"{avg_conf:.2f}")

            st.write("---")

            if final_display_labels:
                st.write("**Detected Risks:**")
                for label in final_display_labels:
                    st.info(f"📍 {label}")

                st.error("⚠️ STATUS: HAZARDOUS")
                with st.expander("📢 Driving Advice", expanded=True):
                    if "Low Visibility (Fog/Salt)" in final_display_labels:
                        st.write("- **Fog Alert:** Visibility is poor. Use fog lights and low beams.")
                    if "Snowy Conditions" in final_display_labels:
                        st.write("- **Snow Alert:** Slippery cryospheric roads. Reduce speed immediately.")
                    st.write("- **General:** Increase braking distance from other vehicles.")
            else:
                st.success("✅ STATUS: SAFE")
                st.write("No critical hazards or severe weather threats detected on the road surface.")

            st.write("---")
            report_txt = (
                f"SASKATOON ROAD SAFETY AI REPORT\n"
                f"================================\n"
                f"Timestamp: {datetime.datetime.now()}\n"
                f"Total Detections Evaluated: {len(boxes)}\n"
                f"Average Model Confidence: {avg_conf:.2f}\n"
                f"Identified Road Threats: {', '.join(final_display_labels) if final_display_labels else 'None'}\n"
            )
            st.download_button("📥 Save Analysis Report", report_txt, "Road_Safety_Report.txt")

        # --- 7. SASKATOON CONTEXT MAP ---
        st.write("---")
        st.subheader("🗺️ Saskatoon Spatial Node Map")

        # Determine status colors for map markers
        map_color = 'red' if final_display_labels else 'green'
        map_popup = f"Hazards Detected: {', '.join(final_display_labels)}" if final_display_labels else "Road Status: Clear"

        m = folium.Map(location=[52.1332, -106.6700], zoom_start=12, tiles='CartoDB dark_matter')

        # Interactive Node Markers in Saskatoon region
        folium.Marker(
            [52.1332, -106.6700],
            popup=f"<b>Central Node:</b> {map_popup}",
            tooltip="Downtown / Idlewyld Dr Node",
            icon=folium.Icon(color=map_color, icon="cloud")
        ).add_to(m)

        folium.Marker(
            [52.1605, -106.6212],
            popup="<b>Circle Drive North Node:</b> Monitoring Active",
            tooltip="Circle Dr East Bridge Node",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

        st_folium(m, width="100%", height=350)

# Footer
st.markdown(f"<p style='text-align: center; color: #4b5563; font-size: 12px;'>System Node: Active | Developed by Agha Wafa Abbas | {datetime.datetime.now().year}</p>", unsafe_allow_html=True)
