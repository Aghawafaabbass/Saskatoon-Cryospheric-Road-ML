import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import datetime
import folium
from streamlit_folium import st_folium

# --- 1. Page Config ---
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide", page_icon="❄️")

# --- 2. Professional CSS ---
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
    </style>
    """, unsafe_allow_html=True)

# --- 3. Robust Model Loader (Simplified) ---
@st.cache_resource
def load_yolo_model():
    # Direct path is best for Streamlit Cloud
    model_file = "best.pt"
    if os.path.exists(model_file):
        try:
            return YOLO(model_file)
        except Exception as e:
            st.error(f"Model error: {e}")
            return None
    else:
        # Check one level deep just in case
        alt_path = os.path.join(os.getcwd(), "best.pt")
        if os.path.exists(alt_path):
            return YOLO(alt_path)
    return None

model = load_yolo_model()

# --- 4. Sidebar Control Center ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>❄️</h1>", unsafe_allow_html=True)
    st.title("Control Center")
    st.metric("City", "Saskatoon, SK", "-14°C")
    st.write("---")
    st.subheader("⚙️ Settings")
    # Setting default to 0.45 to reduce 'Snow' noise in Foggy images
    conf_val = st.slider("AI Sensitivity", 0.1, 1.0, 0.45)
    st.write("---")
    st.success("Developer: Agha Wafa Abbas")
    st.info("Dataset: DAWN Environment")

# --- 5. Main Interface ---
st.markdown('<div class="header-box"><p class="main-title">SASKATOON ROAD SAFETY AI</p></div>', unsafe_allow_html=True)

if model is None:
    st.error("❌ ERROR: 'best.pt' not detected! Please ensure the file is in your main GitHub folder.")
    st.info("Tip: Make sure the file is named exactly 'best.pt' (case-sensitive).")
else:
    uploaded_file = st.file_uploader("📸 Upload Road Snapshot", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("🧠 AI Engine Analyzing..."):
            results = model.predict(source=img, conf=conf_val)
            res_plotted = results[0].plot() # Using first result index safely
            
            # CRITICAL FIX: Convert YOLO's BGR output array to RGB image format for proper Streamlit display colors
            display_img = Image.fromarray(res_plotted[..., ::-1])
            
            # --- Label Filtering Logic ---
            raw_labels = [model.names[int(box.cls)].lower() for box in results[0].boxes]
            
            # Clean Labels for Dashboard
            final_display_labels = set()
            for label in raw_labels:
                if label == 'sand':
                    final_display_labels.add("Low Visibility (Fog/Salt)")
                elif label == 'snow':
                    # Priority: If it's mostly foggy/sand, don't confuse with 'Snow' label
                    if 'sand' not in raw_labels:
                        final_display_labels.add("Snowy Conditions")
                else:
                    final_display_labels.add(label.capitalize())

        # --- 6. Results Layout ---
        col_vis, col_dash = st.columns([1.6, 1])

        with col_vis:
            st.subheader("🔍 Perception View")
            st.image(display_img, use_container_width=True)

        with col_dash:
            st.subheader("🛡️ Safety Dashboard")
            
            if final_display_labels:
                st.write("**Detected Risks:**")
                for label in final_display_labels:
                    st.info(f"📍 {label}")
                
                st.error("⚠️ STATUS: HAZARDOUS")
                with st.expander("📢 Driving Advice", expanded=True):
                    if "Low Visibility (Fog/Salt)" in final_display_labels:
                        st.write("- **Fog Alert:** Visibility is poor. Use fog lights.")
                    if "Snowy Conditions" in final_display_labels:
                        st.write("- **Snow Alert:** Slippery roads. Reduce speed.")
                    st.write("- **General:** Maintain distance from other vehicles.")
            else:
                st.success("✅ STATUS: SAFE")
                st.write("No critical hazards detected.")

            st.write("---")
            report_txt = f"Saskatoon Safety Log\nTime: {datetime.datetime.now()}\nHazards: {', '.join(final_display_labels)}"
            st.download_button("📥 Save Analysis Report", report_txt, "Road_Safety_Report.txt")

        # --- 7. Map ---
        st.write("---")
        m = folium.Map(location=[52.1332, -106.6700], zoom_start=12, tiles='CartoDB dark_matter')
        folium.Marker([52.1332, -106.6700], icon=folium.Icon(color='red' if final_display_labels else 'green')).add_to(m)
        st_folium(m, width="100%", height=300)

# Footer
st.markdown(f"<p style='text-align: center; color: #4b5563; font-size: 12px;'>System Node: Active | Developed by Agha Wafa Abbas | {datetime.datetime.now().year}</p>", unsafe_allow_html=True)
