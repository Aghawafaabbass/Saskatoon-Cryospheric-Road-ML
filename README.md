# ❄️ Saskatoon Cryospheric Road ML System (SCRSS)

### *Real-Time Autonomous Classification of Cryospheric Road Surface Pathologies in Sub-Arctic Urban Environments*

---

## 🔗 Quick Links

| | | | |
|:-:|:-:|:-:|:-:|
| [🌐 Live App](https://saskatoon-cryospheric-road-ml-cj68z2ta6kwlseyyyyaytr.streamlit.app/) | [📄 Zenodo Code](https://zenodo.org/records/20001208) | [📊 Zenodo Data](https://zenodo.org/records/20000920) | [🤗 Colab](https://colab.research.google.com/drive/1oUTWV6toX-F5KXNCYWfxUsDw8fsgpJBm) |

---

## 📖 Abstract

Urban sub-arctic environments (Saskatoon, Canada) present extreme challenges for autonomous vehicle perception. **Black ice, compacted snow, slush, and wet frost** cause deficient camera-based detection and contribute significantly to winter accidents.

**SCRSS** achieves **mAP@50 of 0.847** with **34 ms** inference latency on CPU, a **12.3% improvement** in black ice detection over baseline YOLOv8n.

---

## 🧊 Hazard Classes

| Class | Abbr | Friction (μ) | Severity |
|-------|------|--------------|----------|
| Black Ice | BI | 0.10 – 0.18 | 🔴 CRITICAL |
| Compacted Snow | CS | 0.18 – 0.35 | 🟠 HIGH |
| Slush | SL | 0.25 – 0.50 | 🟠 HIGH |
| Wet Frost | WF | 0.15 – 0.25 | 🟡 MEDIUM |
| Clear Asphalt | CA | 0.65 – 0.85 | 🟢 SAFE |

---

## 🏗️ System Modules

| Module | Name | Function |
|--------|------|----------|
| 1 | IAPM | Image preprocessing (640x640, normalization) |
| 2 | STFFDE | YOLOv8n + MHSA + TGU detection engine |
| 3 | SAGRM | Hazard mapping + Folium + Safety reports |

---

## 🖼️ Screenshots

| Home | Detection |
|:----:|:---------:|
| ![Home](screenshots/home.png) | ![Detection](screenshots/detection.png) |

| Black Ice | Snow | Slush | Frost |
|:---------:|:----:|:-----:|:-----:|
| ![BI](screenshots/black_ice_sample.jpg) | ![CS](screenshots/compacted_snow_sample.jpg) | ![SL](screenshots/slush_sample.jpg) | ![WF](screenshots/wet_frost_sample.jpg) |

| Geospatial Map |
|:--------------:|
| ![Map](screenshots/folium_map.png) |

---

## 📊 Performance Comparison

| Model | Precision | Recall | F1 | mAP@50 | mAP@50:95 |
|-------|-----------|--------|-----|--------|-----------|
| YOLOv5n | 0.712 | 0.683 | 0.697 | 0.694 | 0.412 |
| YOLOv8n | 0.741 | 0.718 | 0.729 | 0.735 | 0.448 |
| YOLOv8s | 0.763 | 0.739 | 0.751 | 0.758 | 0.471 |
| Faster R-CNN | 0.724 | 0.706 | 0.715 | 0.721 | 0.439 |
| SSD-MobileNetV3 | 0.698 | 0.671 | 0.684 | 0.678 | 0.401 |
| YOLOv8n + ViT | 0.779 | 0.754 | 0.766 | 0.772 | 0.483 |
| **SCRSS-STFF** | **0.856** | **0.839** | **0.847** | **0.847** | **0.531** |

---

## 📋 Per-Class Results (AP@50)

| Class | Precision | Recall | AP@50 | Samples |
|-------|-----------|--------|-------|---------|
| Black Ice (BI) | 0.821 | 0.798 | 0.809 | 54 |
| Compacted Snow (CS) | 0.883 | 0.861 | 0.872 | 67 |
| Slush (SL) | 0.847 | 0.834 | 0.840 | 48 |
| Wet Frost (WF) | 0.831 | 0.812 | 0.821 | 29 |
| Clear Asphalt (CA) | 0.896 | 0.878 | 0.887 | 18 |
| **Average** | **0.856** | **0.839** | **0.847** | **216** |

---

## ⚡ Inference Latency (640x640)

| Hardware | Preprocessing | Inference | Total |
|----------|:-------------:|:---------:|:-----:|
| Intel i7-1165G7 (CPU) | 8.2 ms | 25.6 ms | 33.8 ms |
| NVIDIA RTX 3080 (GPU) | 2.1 ms | 4.3 ms | 6.4 ms |
| Streamlit Cloud (CPU) | 9.4 ms | 31.2 ms | 40.6 ms |
| Google Colab T4 (GPU) | 2.8 ms | 5.9 ms | 8.7 ms |

---

## 🔬 Ablation Study

| Configuration | MHSA | TGU | Synthetic Aug | mAP@50 | ΔmAP |
|---------------|:----:|:---:|:-------------:|:------:|:----:|
| YOLOv8n Baseline | ✗ | ✗ | ✗ | 0.735 | — |
| + Synthetic Aug | ✗ | ✗ | ✓ | 0.762 | +2.7 |
| + MHSA Only | ✓ | ✗ | ✗ | 0.779 | +4.4 |
| + TGU Only | ✗ | ✓ | ✗ | 0.751 | +1.6 |
| + MHSA + TGU | ✓ | ✓ | ✗ | 0.811 | +7.6 |
| **SCRSS-STFF (Full)** | ✓ | ✓ | ✓ | **0.847** | **+11.2** |

---

## 🚀 Installation

```bash
git clone https://github.com/Aghawafaabbass/Saskatoon-Cryospheric-Road-ML.git
cd Saskatoon-Cryospheric-Road-ML
pip install -r requirements.txt
streamlit run app.py
Dependencies
Package	Version
opencv-python-headless	4.8.1
ultralytics	8.0.200
streamlit	1.28.0
pillow	10.0.0
numpy	1.24.3
pandas	2.0.3
folium	0.14.0
streamlit-folium	0.11.0
🧪 Training Settings
Parameter	Value
Base Model	YOLOv8n (COCO pretrained)
Epochs	150
Batch Size	16
Image Size	640 × 640
Optimizer	AdamW
Initial LR	0.01
Final LR	0.0001
Weight Decay	0.0005
GPU	NVIDIA A100 40GB
Training Time	≈11.2 hours
🌍 Geospatial Settings
Parameter	Value
Latitude	52.1332° N
Longitude	106.6700° W
Zoom	12
Hazard Marker	🔴 Red
Safe Marker	🟢 Green
📝 Citation
bibtex
@article{abbas2024scrss,
  title={Spatio-Temporal Feature Fusion via Transformer-Based Architectures for Autonomous Classification of Cryospheric Road Surface Pathologies in Sub-Arctic Urban Environments},
  author={Abbas, Agha Wafa},
  year={2024}
}
Archive	DOI
Code	10.5281/zenodo.20001208
Dataset	10.5281/zenodo.20000920
👨‍🏫 Author
Agha Wafa Abbas

Institution	Location
University of Portsmouth	Portsmouth, UK
Arden University	Coventry, UK
Pearson	London, UK
IVY College	Lahore, Pakistan
Email: agha.wafa@port.ac.uk | awabbas@arden.ac.uk | wafa.abbas.lhr@rootsivy.edu.pk

📜 License
MIT License

🙏 Acknowledgments
Source	For
Google Colab Pro	A100 GPU
Ultralytics	YOLOv8
Kenk & Hassaballah	DAWN dataset
Streamlit	Cloud hosting
<div align="center">
Made with ❄️ for sub-arctic road safety

</div> ```
