import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import datetime
import folium
from streamlit_folium import st_folium

# --- Page Config ---
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide", page_icon="❄️")

# --- Improved UI Styling ---
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: #1e2129;
        border: 1px solid #00d4ff;
        padding: 15px;
        border-radius: 12px;
    }
    .main-title {
        color: #00d4ff;
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    footer {visibility: hidden;}
    .footer-text {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: #808495;
        text-align: center;
        padding: 5px;
        font-size: 12px;
        border-top: 1px solid #262730;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Model Loading with Error Handling ---
@st.cache_resource
def load_model():
    try:
        # Load YOLOv8 model trained on DAWN dataset
        return YOLO("best.pt") 
    except Exception as e:
        return None

model = load_model()

# --- Sidebar: Location & Environment ---
with st.sidebar:
    st.image("https://icons8.com", width=80)
    st.title("Saskatoon Hub")
    st.metric("Location", "Saskatoon, SK", "Canada")
    st.metric("Environment", "-14°C", "❄️ Snowy/Icy")
    
    st.write("---")
    conf_threshold = st.slider("AI Sensitivity (Confidence)", 0.1, 1.0, 0.25)
    
    st.write("---")
    st.subheader("👨‍💻 Developer")
    st.info("Agha Wafa Abbas")
    st.caption("AI Safety System v2.0")

# --- Header ---
st.markdown('<p class="main-title">CRYOSPHERIC ROAD SAFETY</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #808495;'>Advanced Computer Vision for Winter Navigation</p>", unsafe_allow_html=True)
st.write("---")

# --- Logic: Map DAWN Labels to Local Context ---
def get_safety_advice(detected_list):
    # DAWN Dataset typically has: Sand, Snow, Fog, Rain
    hazards = []
    advice = "Normal driving conditions."
    is_danger = False
    
    for d in detected_list:
        d = d.lower()
        if d in ['snow', 'ice']:
            hazards.append("❄️ Heavy Snow/Ice")
            advice = "High risk of skidding. Use winter tires. Speed: Max 40km/h."
            is_danger = True
        elif d == 'sand': # In Saskatoon context, this is often low visibility/road salt
            hazards.append("🌫️ Low Visibility (Fog/Grit)")
            advice = "Reduced visibility. Turn on fog lights. Keep distance."
            is_danger = True
        elif d == 'fog':
            hazards.append("☁️ Dense Fog")
            advice = "Use low-beam headlights. Watch for pedestrians."
            is_danger = True
            
    return hazards, advice, is_danger

# --- Main App Logic ---
col_up, col_stat = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader("📤 Drop road image here...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    if model:
        with st.spinner('Analyzing Frame...'):
            results = model.predict(source=image, conf=conf_threshold)
            res_plotted = results[0].plot()
            
            # Extract names
            detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
            hazards, advice, is_danger = get_safety_advice(set(detected_classes))

        # --- Display Results ---
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            st.subheader("🔍 AI Vision Analysis")
            st.image(res_plotted, use_container_width=True)
            
        with c2:
            st.subheader("🛡️ Safety Dashboard")
            if hazards:
                st.write(f"**Conditions:** {', '.join(hazards)}")
                
                if is_danger:
                    st.error("🚨 DANGER: HAZARDOUS ROAD")
                    st.warning(f"**Action:** {advice}")
                else:
                    st.success("✅ ROAD CLEAR")
            else:
                st.success("✅ NO HAZARDS DETECTED")
                st.write("Clear visibility and road surface.")

            # Report Generation
            report = f"SASKATOON ROAD REPORT\n{'='*25}\nTime: {datetime.datetime.now()}\nConditions: {', '.join(hazards) if hazards else 'Clear'}\nStatus: {'DANGER' if is_danger else 'SAFE'}"
            st.download_button("📊 Download Report", report, "Saskatoon_Road_Report.txt")

        # --- Map ---
        st.write("---")
        st.subheader("📍 Deployment Location")
        m = folium.Map(location=[52.1332, -106.6700], zoom_start=13, tiles="cartodbpositron")
        folium.Marker(
            [52.1332, -106.6700], 
            popup="Current Analysis", 
            icon=folium.Icon(color='red' if is_danger else 'green')
        ).add_to(m)
        st_folium(m, width="100%", height=300)

else:
    st.info("Waiting for image upload to start analysis...")

# --- Footer ---
st.markdown(f"""
    <div class="footer-text">
        System Active • Saskatoon Winter AI • Developed by Agha Wafa Abbas • {datetime.datetime.now().year}
    </div>
    """, unsafe_allow_html=True)
