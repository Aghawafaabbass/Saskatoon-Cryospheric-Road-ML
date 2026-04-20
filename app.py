import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Saskatoon Winter AI", layout="wide")
st.title("❄️ Saskatoon Cryospheric Road Safety System")

# Model load karne ka function (caching ke saath)
@st.cache_resource
def load_model():
    # Ensure "best.pt" is in the same folder as app.py
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load karne mein masla hua: {e}")

# File uploader
uploaded_file = st.file_uploader("Upload a road image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Image open karna
    image = Image.open(uploaded_file)
    
    # Image ko array mein convert karna
    img_array = np.array(image)
    
    # Model prediction
    # model.predict hamesha list return karta hai
    results = model.predict(source=img_array, conf=0.25)
    
    # Sabse pehla result (index 0) utha kar plot karna
    # results[0].plot() hi sahi syntax hai
    res_plotted = results[0].plot()
    
    # Display the results
    st.image(res_plotted, caption="Processed Image: Road Condition Detection", use_container_width=True)
    st.success("Analysis Complete! Safe travels in Saskatoon! 🚦")

    # Optional: Prediction results ka data dikhane ke liye
    with st.expander("See detection details"):
        st.write(results[0].probs if hasattr(results[0], 'probs') else "No probability data")
