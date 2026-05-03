import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import datetime
import folium
from streamlit_folium import st_folium

# --- 1. Advanced UI Configuration ---
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide", page_icon="❄️")

st.markdown("""
    <style>
    /* Professional Dark Theme */
    .stApp { background-color: #0e1117; }
    
    /* Clean Header Section */
    .header-container {
        background: linear-gradient(90deg, #1e3a8a, #1e40af);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid #3b82f6;
    }
    .main-title { color: white; font-size: 32px; font-weight: 800; margin: 0; }
    
    /* Dashboard Cards */
    [data-testid="stMetric"] {
        background-color: #161e2e;
        border: 1px solid #1f2937;
        padding: 15px;
        border-radius: 12px;
    }
    
    /* Floating Footer */
    .footer {
        position: fixed;
        bottom: 0; left: 0; width: 100%;
        background: #111827; color: #9ca3af;
        text-align: center; padding: 10px; font-size: 12px;
        border-top: 1px solid #1f2937; z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Model Loading (Secure) ---
@st.cache_resource
def get_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Error: Could not find 'best.pt'. Ensure weights are in the root folder.")
        return None

model = get_model()

# --- 3. Sidebar (Control Center) ---
with st.sidebar:
    st.image("https://icons8.com", width=80)
    st.title("📍 Control Center")
    st.metric("City", "Saskatoon, SK", "-14°C ❄️")
    
    st.write("---")
    st.subheader("⚙️ Settings")
    conf_val = st.slider("AI Sensitivity", 0.1, 1.0, 0.35)
    
    st.write("---")
    st.success(f"Developer: Agha Wafa Abbas")
    st.info("System: YOLOv8 Perception Engine")

# --- 4. Main Interface Header ---
st.markdown("""
    <div class="header-container">
        <p class="main-title">❄️ SASKATOON CRYOSPHERIC ROAD PERCEPTION</p>
        <p style="color: #bfdbfe; margin-top: 5px;">AI-Driven Safety Monitoring for Winter Conditions</p>
    </div>
    """, unsafe_allow_html=True)

# --- 5. Image Upload & Processing ---
uploaded_file = st.file_uploader("📸 Upload Road Image or CCTV Snapshot", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read Image
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Layout for analysis
    col_vis, col_data = st.columns([1.6, 1])
    
    with st.spinner("🧠 AI Engine Running..."):
        # Run Detection
        results = model.predict(source=raw_img, conf=conf_val)
        
        # Get result image (using the first result in list)
        res_img_array = results[0].plot() 
        
        # Logic for Hazards (DAWN Dataset Mapping)
        detected_names = [model.names[int(box.cls)] for box in results[0].boxes]
        
        # Cleaning and Mapping labels
        unique_labels = list(set(detected_names))
        hazard_found = any(x.lower() in ['snow', 'ice', 'sand', 'fog'] for x in unique_labels)

    # --- Display Column 1: Vision ---
    with col_vis:
        st.subheader("🔍 Real-time Detection")
        st.image(res_img_array, use_container_width=True, caption="Inference Result")

    # --- Display Column 2: Safety Analytics ---
    with col_data:
        st.subheader("🛡️ Safety Dashboard")
        
        if unique_labels:
            # Show Labels as Tags
            st.write("**Detected Conditions:**")
            cols = st.columns(len(unique_labels))
            for i, label in enumerate(unique_labels):
                st.info(f"📍 {label.capitalize()}")

            # Alert Logic
            if hazard_found:
                st.error("⚠️ STATUS: HAZARDOUS CONDITIONS")
                st.warning("""
                **Driving Advice:**
                - Speed Limit: 30 km/h
                - Low Visibility: Use Fog Lights
                - Caution: High risk of skidding
                """)
            else:
                st.success("✅ STATUS: ROAD SAFE")
                st.write("Maintain normal speeds. Road surface appears clear.")
        else:
            st.success("✅ NO IMMEDIATE HAZARDS")
            st.write("Detection engine found no critical weather risks.")

        # Download Feature
        st.write("---")
        report_txt = f"Saskatoon Safety Report\nDate: {datetime.datetime.now()}\nConditions: {', '.join(unique_labels)}"
        st.download_button("📥 Download Analysis", report_txt, "Saskatoon_Report.txt")

    # --- 6. Map Section ---
    st.write("---")
    st.subheader("🗺️ Incident Geolocation")
    m = folium.Map(location=[52.1332, -106.6700], zoom_start=12, tiles='CartoDB dark_matter')
    folium.Marker(
        [52.1332, -106.6700], 
        popup="Live Analysis Point", 
        icon=folium.Icon(color='red' if hazard_found else 'green', icon='car', prefix='fa')
    ).add_to(m)
    st_folium(m, width="100%", height=300)

else:
    # Placeholder when no image is uploaded
    st.info("👋 Welcome! Please upload a road image from Saskatoon to begin the AI analysis.")
    st.image("https://unsplash.com", caption="Saskatoon Winter Preview", use_container_width=True)

# --- 7. Footer ---
st.markdown(f"""
    <div class="footer">
        System Node: Active | Version 2.1.0 | Developed by <b>Agha Wafa Abbas</b> | {datetime.datetime.now().year}
    </div>
    """, unsafe_allow_html=True)
