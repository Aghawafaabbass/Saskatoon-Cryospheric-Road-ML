import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import datetime
import folium
from streamlit_folium import st_folium

# --- Page Config ---
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide", page_icon="❄️")

# --- Professional CSS ---
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

# --- 1. Robust Model Loading (Senior ML approach) ---
@st.cache_resource
def load_model():
    # Model dhoondne ka behtareen tareeka
    model_path = "best.pt"
    if not os.path.exists(model_path):
        # Agar root mein nahi hai, to pure repository mein check karo
        for root, dirs, files in os.walk("."):
            if "best.pt" in files:
                model_path = os.path.join(root, "best.pt")
                break
    
    try:
        return YOLO(model_path)
    except:
        return None

model = load_model()

# --- 2. Sidebar Control Center ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>❄️</h1>", unsafe_allow_html=True)
    st.title("Control Center")
    st.metric("City", "Saskatoon, SK", "-14°C")
    st.write("---")
    
    st.subheader("⚙️ Analysis Settings")
    # Senior tip: Defaults to 0.40 but user can tweak for night/day
    conf_val = st.slider("AI Confidence Threshold", 0.1, 1.0, 0.40)
    
    st.write("---")
    st.success("Developer: Agha Wafa Abbas")
    st.info("Dataset: DAWN (Weather Perception)")

# --- 3. UI Header ---
st.markdown('<div class="header-box"><p class="main-title">SASKATOON ROAD SAFETY AI</p></div>', unsafe_allow_html=True)

if model is None:
    st.error("❌ CRITICAL ERROR: 'best.pt' not found in repository. Please upload your weights file.")
else:
    uploaded_file = st.file_uploader("📤 Upload Road Image (DAWN Dataset compatible)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        # 4. Professional Inference Logic
        with st.spinner("🧠 AI Analysis in progress..."):
            results = model.predict(source=img, conf=conf_val)
            # YOLOv8 returns a list, we take the first element
            res_plotted = results[0].plot() 
            
            # Extract Detected Classes
            detected_classes = [model.names[int(box.cls)].lower() for box in results[0].boxes]
            unique_labels = list(set(detected_classes))

        # 5. Dashboard Layout
        col_vis, col_dash = st.columns([1.6, 1])

        with col_vis:
            st.subheader("🔍 Real-time Detection")
            st.image(res_plotted, use_container_width=True, caption="Inference Result")

        with col_dash:
            st.subheader("🛡️ Safety Dashboard")
            
            if unique_labels:
                st.write("**Detected Conditions:**")
                for label in unique_labels:
                    # Mapping DAWN 'Sand' to Saskatoon 'Fog/Salt'
                    display_name = "Fog / Road Salt (Visibility)" if label == 'sand' else label.capitalize()
                    st.info(f"📍 {display_name}")

                # Hazard Logic for DAWN Classes
                is_danger = any(x in ['snow', 'ice', 'sand', 'fog', 'rain'] for x in detected_classes)
                
                if is_danger:
                    st.error("🚨 STATUS: HAZARDOUS")
                    with st.expander("📢 Driving Advice", expanded=True):
                        if 'snow' in detected_classes:
                            st.write("- **Snow:** High skidding risk. Use winter tires.")
                        if 'sand' in detected_classes or 'fog' in detected_classes:
                            st.write("- **Visibility:** Low visibility detected. Use fog lights.")
                        st.write("- **Speed:** Recommended max 30 km/h.")
                else:
                    st.success("✅ STATUS: SAFE")
            else:
                st.success("✅ NO HAZARDS DETECTED")

            # Report Feature
            st.write("---")
            report_data = f"Report: {datetime.datetime.now()}\nConditions: {', '.join(unique_labels)}"
            st.download_button("📥 Download Report", report_data, "Saskatoon_Report.txt")

        # 6. Map Section
        st.write("---")
        m = folium.Map(location=[52.1332, -106.6700], zoom_start=12, tiles='CartoDB dark_matter')
        folium.Marker([52.1332, -106.6700], icon=folium.Icon(color='red' if unique_labels else 'green')).add_to(m)
        st_folium(m, width="100%", height=300)

    else:
        st.info("👋 Awaiting image upload for Saskatoon road analysis.")

# Footer
st.markdown(f"<p style='text-align: center; color: #4b5563; font-size: 12px;'>System Active | {datetime.datetime.now().year}</p>", unsafe_allow_html=True)
