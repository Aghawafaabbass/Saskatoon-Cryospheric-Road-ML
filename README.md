<div align="center">

# 🌨️ Saskatoon Cryospheric Road Safety System (SCRSS)

### Spatio-Temporal Feature Fusion via Transformer-Based Architectures for Autonomous Classification of Cryospheric Road Surface Pathologies in Sub-Arctic Urban Environments

[![DOI Paper](https://img.shields.io/badge/DOI%20Paper-10.5281%2Fzenodo.20000920-blue?style=for-the-badge&logo=zenodo)](https://doi.org/10.5281/zenodo.20000920)
[![DOI Software](https://img.shields.io/badge/DOI%20Software-10.5281%2Fzenodo.20001208-blue?style=for-the-badge&logo=zenodo)](https://doi.org/10.5281/zenodo.20001208)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://saskatoon-cryospheric-road-ml-cj68z2ta6kwlseyyyyaytr.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com)

**Author:** Agha Wafa Abbas

📧 [agha.wafa@port.ac.uk](mailto:agha.wafa@port.ac.uk) | [awabbas@arden.ac.uk](mailto:awabbas@arden.ac.uk) | [wafa.abbas.lhr@rootsivy.edu.pk](mailto:wafa.abbas.lhr@rootsivy.edu.pk)

*University of Portsmouth, UK · Arden University, UK · Pearson, UK · IVY College of Management Sciences, Lahore, Pakistan*

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [System Architecture](#-system-architecture)
- [Screenshots](#-screenshots)
- [Live Demo](#-live-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Hazard Categories](#-hazard-categories)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

Winter road surfaces present one of the most persistent challenges in autonomous vehicle perception. Conditions such as **fog**, **snow**, and **low-visibility environments** transition rapidly and are routinely missed by standard camera-based detection pipelines — precisely when safety is most critical.

The **Saskatoon Cryospheric Road Safety System (SCRSS)** is a fully deployed AI application that addresses this through:

- 🎯 **YOLOv8-based real-time object detection** with configurable AI sensitivity (0.1–1.0 confidence threshold)
- 🛡️ **Intelligent label filtering** — raw detections are mapped to human-readable safety categories (e.g., `sand` → `Low Visibility (Fog/Salt)`, `snow` → `Snowy Conditions`)
- 🗺️ **Folium-based interactive geospatial incident map** for location-aware hazard visualisation
- 📄 **Downloadable safety analysis reports** with timestamped hazard logs for municipal and operational use
- 🖥️ **Streamlit-powered web interface** — no installation required, runs entirely in the browser

The system uses a custom-trained YOLOv8 model (`best.pt`) trained on the **DAWN (Detection in Adverse Weather Nature)** benchmark dataset and deployed on **Streamlit Cloud**.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **mAP@50** (all hazard categories) | **0.847** |
| **Inference Latency** (standard CPU) | **34 ms/frame** |
| **Default Confidence Threshold** | **0.45** (tuned to reduce false positives in foggy conditions) |
| **Training Dataset** | DAWN (Detection in Adverse Weather Nature) benchmark |
| **Model File** | `best.pt` (YOLOv8, custom-trained) |
| **Deployment** | Streamlit Cloud — zero-install, browser-based |

---

## 🏗️ System Architecture

The SCRSS pipeline is structured around three core stages — ingestion, inference, and output — each directly implemented in `app.py`:

**Stage 1 — Image Ingestion**
The user uploads a road image (JPG/PNG) via the Streamlit file uploader. The image is converted to RGB and passed directly to the YOLO inference engine.

**Stage 2 — AI Inference**
A custom YOLOv8 model (`best.pt`) runs detection at the user-configured confidence threshold. Raw predicted class labels (e.g., `sand`, `snow`) are passed through an intelligent label filtering layer that maps them to safety-meaningful categories:
- `sand` → `Low Visibility (Fog/Salt)`
- `snow` (when fog not dominant) → `Snowy Conditions`

**Stage 3 — Output & Reporting**
Results are rendered across three outputs simultaneously:

```
Uploaded Image
      │
      ▼
  YOLOv8 Model (best.pt)
      │
      ▼
  Label Filtering Layer
      │
      ├──▶ 🔍 Perception View (annotated image with bounding boxes)
      ├──▶ 🛡️ Safety Dashboard (hazard list + driving advice)
      ├──▶ 🗺️  Folium Geospatial Map (incident marker)
      └──▶ 📄 Downloadable Safety Report (.txt, timestamped)
```

---

## 📸 Screenshots

### Application Interface

#### 🏠 Main Dashboard & Control Center
![Main Dashboard](screenshots/Sc%201.PNG)

---

#### 🔍 Perception View — AI Detection Output
![Detection Interface](screenshots/Sc%202.PNG)

---

#### 🛡️ Safety Dashboard & Driving Advice
![Safety Dashboard](screenshots/Sc%203.PNG)

---

#### 🗺️ Geospatial Incident Map
![Geospatial Map](screenshots/Sc%204.PNG)

---

## 🚀 Live Demo

The system is deployed and accessible at:

**👉 [https://saskatoon-cryospheric-road-ml-cj68z2ta6kwlseyyyyaytr.streamlit.app/](https://saskatoon-cryospheric-road-ml-cj68z2ta6kwlseyyyyaytr.streamlit.app/)**

The application allows users to:
- Adjust the **AI Sensitivity** slider (0.1–1.0) to control detection confidence
- Upload a road image (JPG/JPEG/PNG) for real-time hazard analysis
- View the **Perception View** — annotated image with bounding boxes
- Read the **Safety Dashboard** — detected hazard labels and contextual driving advice
- Explore the **interactive Folium map** showing the incident location
- Download a timestamped **Safety Analysis Report** (.txt)

---

## ⚙️ Installation

### Prerequisites

- Python 3.11
- pip

### Clone and Install

```bash
git clone https://github.com/Aghawafaabbass/Saskatoon-Cryospheric-Road-ML.git
cd Saskatoon-Cryospheric-Road-ML
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 🧪 Usage

1. **Open the app** — locally at `http://localhost:8501` or via the [live demo](https://saskatoon-cryospheric-road-ml-cj68z2ta6kwlseyyyyaytr.streamlit.app/)
2. **Adjust AI Sensitivity** — use the sidebar slider (default: `0.45`) to tune detection confidence for your image conditions
3. **Upload a road image** — JPG, JPEG, or PNG of a winter road surface
4. **View Perception View** — the AI overlays bounding boxes with detected hazard labels on your image
5. **Read the Safety Dashboard** — hazard types are listed with tailored driving advice (fog lights, speed reduction, etc.)
6. **Check the Folium map** — the incident location is pinned (red = hazard detected, green = safe)
7. **Download the report** — click `📥 Save Analysis Report` to get a timestamped `.txt` safety log

> **Note:** The pre-trained model weights (`best.pt`) must be present in the repository root. The app auto-loads them on startup via `@st.cache_resource`.

---

## 📁 Dataset

The SCRSS model (`best.pt`) is trained on the **DAWN (Detection in Adverse Weather Nature)** benchmark dataset — a publicly available collection of real-world road images captured under adverse weather conditions including fog, rain, snow, and sand storms. DAWN provides the diverse low-visibility scenarios required to train a robust winter road hazard detector.

---

## ❄️ Hazard Categories

The model detects road conditions from the DAWN dataset classes. The app applies an intelligent label filtering layer to translate raw model outputs into safety-meaningful categories displayed to the user:

| Raw Model Label | Displayed As | Driving Advice Triggered |
|---|---|---|
| `sand` | **Low Visibility (Fog/Salt)** | Fog Alert — use fog lights |
| `snow` (fog not dominant) | **Snowy Conditions** | Snow Alert — reduce speed |
| Other classes | Capitalised as-is | General distance advice |
| No detections | ✅ **SAFE** | No hazards detected |

---

## 📖 Citation

If you use this work, please cite both the preprint and the software:

### Research Paper (Preprint)

```bibtex
@misc{abbas2026scrss,
  author       = {Abbas, Agha Wafa},
  title        = {Spatio-Temporal Feature Fusion via Transformer-Based Architectures 
                  for Autonomous Classification of Cryospheric Road Surface Pathologies 
                  in Sub-Arctic Urban Environments},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20000920},
  url          = {https://doi.org/10.5281/zenodo.20000920}
}
```

### Software

```bibtex
@software{abbas2026scrss_software,
  author       = {Agha Wafa Abbas},
  title        = {Aghawafaabbass/Saskatoon-Cryospheric-Road-ML v1.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20001208},
  url          = {https://doi.org/10.5281/zenodo.20001208}
}
```

---

## 🏛️ Affiliations

| Institution | Role |
|---|---|
| School of Computing, **University of Portsmouth**, Portsmouth PO1 2UP, UK | Lecturer |
| School of Computing, **Arden University**, Coventry, UK | Lecturer |
| School of Computing, **Pearson**, London, UK | Lecturer |
| School of Computing and Emerging Technologies, **IVY College of Management Sciences**, Lahore, Pakistan | Lecturer |

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Agha Wafa Abbas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

*Built for safer winter roads. Open-source. Reproducible. Deployable.*

⭐ **Star this repository** if you find it useful!

</div>
