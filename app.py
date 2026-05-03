import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import datetime
import folium
from streamlit_folium import st_folium

# --- 1. Pro UI Configuration ---
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide", page_icon="❄️")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .header-box {
        background: linear-gradient(90deg, #1e3a8a, #111827);
        padding: 30px; border-radius: 15px; text-align: center;
        border: 1px solid #1e40af; margin-bottom: 25px;
    }
    .main-title { color: #60a5fa; font-size: 38px; font-weight: 800; margin: 0; }
    [data-testid="stMetric"] { background-color: #161e2e; border: 1px solid #1f2937; border-radius: 12px; }
    /* Dashboard Fix */
    .status-box { padding: 10px; border-radius: 10px; margin: 5px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Robust Model Loading ---
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        for root, dirs, files in os.walk("."):
            if "best.pt" in files:
                model_path = os.path.join(root, "best.pt")
                break
    try:
        return YOLO(model_path)
    except:
        return None

model = load_model()

# --- 3. Sidebar Control Center ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>❄️</h1>", unsafe_allow_html=True)
    st.title("Control Center")
    st.metric("City", "Saskatoon, SK", "-14°C")
    st.write("---")
    st.subheader("⚙️ Settings")
    conf_val = st.slider("Detection Sensitivity", 0.1, 1.0, 0.45) # Increased default to reduce noise
    st.write("---")
    st.success("Developer: Agha Wafa Abbas")
    st.info("Dataset: DAWN Architecture")

# --- 4. Main Interface ---
st.markdown('<div class="header-box"><p class="main-title">SASKATOON ROAD SAFETY AI</p></div>', unsafe_allow_html=True)

if model is None:
    st.error("❌ Model 'best.pt' not found. Please upload it to your repository.")
else:
    uploaded_file = st.file_uploader("📸 Upload Road Snapshot", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("🧠 AI analyzing conditions..."):
            results = model.predict(source=img, conf=conf_val)
            res_plotted = results[0].plot() 
            
            # --- Senior ML Logic: Label Cleaning ---
            raw_labels = [model.names[int(box.cls)].lower() for box in results[0].boxes]
            
            # Map labels and Remove Duplicates
            final_conditions = set()
            for label in raw_labels:
                if label == 'sand':
                    final_conditions.add("Fog / Road Salt")
                elif label == 'snow':
                    # Only add snow if it's not a clear 'fog' day to avoid confusion
                    if 'sand' not in raw_labels: 
                        final_conditions.add("Snowy Surface")
                else:
                    final_conditions.add(label.capitalize())

        # --- 5. Dashboard Layout ---
        col_vis, col_dash = st.columns([1.6, 1])

        with col_vis:
            st.subheader("🔍 Perception View")
            st.image(res_plotted, use_container_width=True)

        with col_dash:
            st.subheader("🛡️ Safety Dashboard")
            
            if final_conditions:
                st.write("**Current Hazards:**")
                for cond in final_conditions:
                    st.info(f"📍 {cond}")

                # Alert Logic
                st.error("⚠️ STATUS: HAZARDOUS")
                with st.expander("📢 Smart Driving Advice", expanded=True):
                    if "Fog / Road Salt" in final_conditions:
                        st.write("- **Fog:** Low visibility. Turn on fog lights and reduce speed.")
                    if "Snowy Surface" in final_conditions:
                        st.write("- **Snow:** Icy roads. Avoid sudden braking.")
                    st.write("- **Police Alert:** Maintain 2-car distance.")
            else:
                st.success("✅ STATUS: SAFE")
                st.write("Normal driving conditions detected.")

            st.write("---")
            report_txt = f"Saskatoon Report: {datetime.datetime.now()}\nConditions: {', '.join(final_conditions)}"
            st.download_button("📥 Save Analysis Report", report_txt, "Saskatoon_Safety_Log.txt")

        # --- 6. Map ---
        st.write("---")
        m = folium.Map(location=[52.1332, -106.6700], zoom_start=12, tiles='CartoDB dark_matter')
        folium.Marker([52.1332, -106.6700], icon=folium.Icon(color='red' if final_conditions else 'green')).add_to(m)
        st_folium(m, width="100%", height=300)

st.markdown(f"<p style='text-align: center; color: #4b5563; font-size: 12px;'>System Active | Saskatoon Winter AI | {datetime.datetime.now().year}</p>", unsafe_allow_html=True)
