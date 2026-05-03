import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import datetime
import folium
from streamlit_folium import st_folium

# --- 1. Pro UI Configuration ---
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide", page_icon="❄️")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .header-container {
        background: linear-gradient(90deg, #1e3a8a, #111827);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #1e40af;
        margin-bottom: 20px;
    }
    .main-title { color: #60a5fa; font-size: 35px; font-weight: 800; margin: 0; letter-spacing: 1px; }
    [data-testid="stMetric"] {
        background-color: #161e2e;
        border: 1px solid #1f2937;
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Model Loader ---
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error("⚠️ Error: 'best.pt' missing!")
        return None

model = load_yolo_model()

# --- 3. Sidebar with Dynamic Status ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>❄️</h1>", unsafe_allow_html=True)
    st.title("Control Center")
    st.metric("City", "Saskatoon, SK")
    st.metric("Temperature", "-14°C", "Snowy Condition")
    
    st.write("---")
    st.subheader("⚙️ Analysis Mode")
    auto_mode = st.toggle("Auto-Sensitivity Logic", value=True, help="AI adjusts threshold based on weather")
    
    # User can still manually override if auto_mode is off
    conf_val = st.slider("Manual Sensitivity", 0.1, 1.0, 0.40, disabled=auto_mode)
    
    st.write("---")
    st.success(f"Developer: Agha Wafa Abbas")
    st.caption("v2.2.0 | Auto-Tuning Engine Active")

# --- 4. Main Interface ---
st.markdown("""
    <div class="header-container">
        <p class="main-title">SASKATOON ROAD SAFETY AI</p>
        <p style="color: #94a3b8;">Cryospheric Road Perception & Hazard Detection</p>
    </div>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Image (Fog/Snow/Sand from DAWN set)", type=['jpg', 'jpeg', 'png'])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert("RGB")
    
    # 5. Smart AI Inference with Auto-Threshold
    with st.spinner("AI Analysis in progress..."):
        # Step 1: Pre-scan with low confidence to check environment
        pre_results = model.predict(source=img, conf=0.20, verbose=False)
        pre_labels = [model.names[int(box.cls)].lower() for box in pre_results[0].boxes]
        
        # Step 2: Auto-adjust logic
        if auto_mode:
            if 'sand' in pre_labels or 'fog' in pre_labels:
                final_conf = 0.25  # Lower threshold for low visibility
                st.sidebar.info("💡 Auto-Mode: Lowering threshold for Fog/Sand")
            elif 'snow' in pre_labels:
                final_conf = 0.40  # Standard for snow
            else:
                final_conf = 0.50  # Strict for clear conditions
        else:
            final_conf = conf_val

        # Step 3: Final Inference
        results = model.predict(source=img, conf=final_conf, verbose=False)
        res_plotted = results[0].plot()
        
        raw_labels = [model.names[int(box.cls)] for box in results[0].boxes]
        
        # Map labels to Saskatoon Context
        mapped_labels = []
        for L in set(raw_labels):
            if L.lower() == 'sand':
                mapped_labels.append("Low Visibility (Fog/Salt)")
            else:
                mapped_labels.append(L.capitalize())

    # 6. Dashboard Layout
    col_img, col_dash = st.columns([1.7, 1])

    with col_img:
        st.subheader("🔍 AI Vision Analysis")
        st.image(res_plotted, use_container_width=True)

    with col_dash:
        st.subheader("🛡️ Safety Dashboard")
        st.write(f"**Confidence Level:** `{final_conf:.2f}`")
        
        if mapped_labels:
            st.write("**Detected Conditions:**")
            for label in mapped_labels:
                st.info(f"📍 {label}")
            
            is_hazardous = any(x.lower() in ['snow', 'ice', 'sand', 'fog', 'low visibility (fog/salt)'] for x in [m.lower() for m in mapped_labels])
            
            if is_hazardous:
                st.error("🚨 STATUS: HAZARDOUS")
                with st.expander("📢 Driving Advice", expanded=True):
                    st.write("- **Speed:** Max 30-40 km/h")
                    st.write("- **Visibility:** Use Fog Lights")
                    st.write("- **Road:** High risk of black ice")
            else:
                st.success("✅ STATUS: SAFE")
        else:
            st.success("✅ NO HAZARDS DETECTED")

        # Download Report
        st.write("---")
        report = f"Saskatoon AI Report\nTime: {datetime.datetime.now()}\nThreshold Used: {final_conf}\nHazards: {', '.join(mapped_labels)}"
        st.download_button("📥 Save Analysis", report, "Saskatoon_Report.txt")

    # 7. Map Section
    st.write("---")
    st.subheader("📍 Geolocation Context")
    m = folium.Map(location=[52.1332, -106.6700], zoom_start=12, tiles='CartoDB dark_matter')
    folium.Marker([52.1332, -106.6700], icon=folium.Icon(color='red' if mapped_labels else 'green')).add_to(m)
    st_folium(m, width="100%", height=300)

elif not model:
    st.error("Model file not found. Please upload 'best.pt' to the repository.")
else:
    st.info("Waiting for image upload to begin analysis.")

st.markdown(f"<p style='text-align: center; color: #4b5563; font-size: 12px;'>Developed by Agha Wafa Abbas | {datetime.datetime.now().year}</p>", unsafe_allow_html=True)
