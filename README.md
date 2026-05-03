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

Urban sub-arctic environments — such as **Saskatoon, Saskatchewan, Canada** — present extraordinary challenges to autonomous vehicle perception systems. Cryospheric road surface pathologies including **black ice**, **compacted snow**, **slush formation**, and **wet frost** are responsible for a significant proportion of winter traffic accidents, while simultaneously degrading standard camera-based detection pipelines.

The **Saskatoon Cryospheric Road Safety System (SCRSS)** addresses this challenge through a novel architecture that fuses:

- 🎯 **YOLOv8-based object detection** for real-time road surface hazard localisation
- 🧠 **Spatio-Temporal Transformer Feature Fusion** — Spatial Attention Unit (SAU) + Temporal Context Aggregator (TCA)
- 🗺️ **Folium-based geospatial incident mapping** for municipal traffic management
- 📄 **Downloadable safety report generation** for stakeholder decision support

The system is trained on the **DAWN** benchmark dataset augmented with synthetic sub-arctic imagery and deployed as a fully interactive **Streamlit web application**.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **mAP@50** (5 hazard categories) | **0.847** |
| **Inference Latency** (standard CPU) | **34 ms/frame** |
| **Black Ice Detection Improvement** over baseline YOLOv8n | **+12.3%** |
| **Training Dataset** | DAWN benchmark + synthetic sub-arctic augmentation |
| **Architecture** | YOLOv8n + Spatial Attention Unit + Temporal Context Aggregator |

---

## 🏗️ System Architecture

The SCRSS architecture extends the YOLOv8n backbone with two purpose-built modules:

1. **Spatial Attention Unit (SAU)** — enhances localisation of low-contrast hazards (e.g., black ice, wet frost) by re-weighting feature maps according to spatial saliency.
2. **Temporal Context Aggregator (TCA)** — aggregates features across sequential frames via a lightweight transformer, enabling temporal reasoning for transitional cryospheric states.

```
Input Frames
    │
    ▼
YOLOv8n Backbone
    │
    ├──▶ Spatial Attention Unit (SAU)
    │           │
    ▼           ▼
Feature Pyramid Network (FPN)
    │
    ▼
Temporal Context Aggregator (TCA)
    │
    ▼
Detection Head → Hazard Classification
    │
    ├──▶ Folium Geospatial Map
    └──▶ Safety Report (PDF)
```

---

## 📸 Screenshots

### Application Interface

#### 🏠 Main Dashboard
![Main Dashboard](screenshots/Sc%201.PNG)

---

#### 🔍 Real-Time Road Surface Detection
![Detection Interface](screenshots/Sc%202.PNG)

---

#### 🗺️ Geospatial Incident Map
![Geospatial Map](screenshots/Sc%203.PNG)

---

#### 📄 Safety Report Generation
![Safety Report](screenshots/Sc%204.PNG)

---

## 🚀 Live Demo

The system is deployed and accessible at:

**👉 [https://saskatoon-cryospheric-road-ml-cj68z2ta6kwlseyyyyaytr.streamlit.app/](https://saskatoon-cryospheric-road-ml-cj68z2ta6kwlseyyyyaytr.streamlit.app/)**

The application allows users to:
- Upload road surface images or video frames for real-time analysis
- View detected cryospheric hazards with bounding boxes and confidence scores
- Explore an interactive geospatial incident map
- Download a formatted safety report

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

1. **Open the app** — either locally or via the [live demo link](https://saskatoon-cryospheric-road-ml-cj68z2ta6kwlseyyyyaytr.streamlit.app/)
2. **Upload an image or frame** — JPEG or PNG captured in winter road conditions
3. **View detection results** — hazards are classified and localised in real time
4. **Explore the geospatial map** — incident locations are plotted interactively via Folium
5. **Download your safety report** — a structured PDF is generated for each session

The pre-trained model weights are included as `best.pt`.

---

## 📁 Dataset

The SCRSS is trained on the **DAWN (Detection in Adverse Weather Nature)** benchmark dataset, augmented with **synthetically generated sub-arctic winter imagery** to improve coverage of Saskatchewan-specific cryospheric conditions including black ice glare, blowing snow obscuration, and frost accumulation.

---

## ❄️ Hazard Categories

| # | Hazard | Description |
|---|--------|-------------|
| 1 | **Black Ice** | Transparent ice film on asphalt; highest accident risk |
| 2 | **Compacted Snow** | Dense, pressed snow surface with reduced traction |
| 3 | **Slush Formation** | Semi-liquid snow-water mixture; spray hazard |
| 4 | **Wet Frost** | Surface frost activated by moisture |
| 5 | **Clear/Dry** | Baseline safe condition |

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

*Built for safer roads in sub-arctic cities. Open-source. Reproducible. Deployable.*

⭐ **Star this repository** if you find it useful!

</div>
