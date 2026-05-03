<div align="center">

# 🌨️ Saskatoon Cryospheric Road Safety System (SCRSS)

### Spatio-Temporal Feature Fusion via Transformer-Based Architectures for Autonomous Classification of Cryospheric Road Surface Pathologies in Sub-Arctic Urban Environments

[![DOI Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.20000920.svg)](https://doi.org/10.5281/zenodo.20000920)
[![DOI Software](https://zenodo.org/badge/DOI/10.5281/zenodo.20001208.svg)](https://doi.org/10.5281/zenodo.20001208)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://saskatoon-cryospheric-road-ml-cj68z2ta6kwlseyyyyaytr.streamlit.app/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://ultralytics.com)

**Author:** Agha Wafa Abbas  
📧 [agha.wafa@port.ac.uk](mailto:agha.wafa@port.ac.uk) | [awabbas@arden.ac.uk](mailto:awabbas@arden.ac.uk) | [wafa.abbas.lhr@rootsivy.edu.pk](mailto:wafa.abbas.lhr@rootsivy.edu.pk)

*Lecturer, School of Computing, University of Portsmouth, Southsea, Portsmouth PO1 2UP, United Kingdom*  
*Lecturer, School of Computing, Arden University, Coventry, United Kingdom*  
*Lecturer, School of Computing, Pearson, London, United Kingdom*  
*Lecturer, School of Computing and Emerging Technologies, IVY College of Management Sciences, Lahore, Pakistan*

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
- 🧠 **Spatio-Temporal Transformer Feature Fusion** for contextual reasoning across frames
- 🗺️ **Folium-based geospatial incident mapping** for municipal traffic management
- 📄 **Downloadable safety report generation** for stakeholder decision support

The system is deployed as a fully interactive **Streamlit web application** and is open-source, providing a deployable baseline for cryospheric road intelligence in high-latitude metropolitan regions.

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
2. **Temporal Context Aggregator (TCA)** — aggregates features across sequential frames via a lightweight transformer mechanism, providing temporal reasoning for transitional cryospheric states (e.g., slush-to-ice transition).

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

<!-- Replace the image paths below with the actual screenshot filenames in your repo -->
<!-- Upload screenshots to a `screenshots/` folder in this repository -->

#### 🏠 Main Dashboard
![Main Dashboard](screenshots/dashboard.png)

#### 🔍 Real-Time Road Surface Detection
![Detection Interface](screenshots/detection.png)

#### 🗺️ Geospatial Incident Map
![Geospatial Map](screenshots/geospatial_map.png)

#### 📊 Model Performance Metrics
![Performance Metrics](screenshots/metrics.png)

#### 📄 Safety Report Generation
![Safety Report](screenshots/safety_report.png)

> **Note:** To add your screenshots, create a `screenshots/` folder in the repository root and upload your images. Then update the paths above to match your filenames.

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
2. **Upload an image or frame** — JPEG, PNG, or video frame captured in winter road conditions
3. **View detection results** — the model classifies and localises road surface hazards in real time
4. **Explore the geospatial map** — incident locations are plotted interactively via Folium
5. **Download your safety report** — a structured PDF report is generated for each session

The pre-trained model weights are included in the repository as `best.pt` (YOLOv8n backbone with SAU and TCA modules, trained on DAWN + synthetic data).

---

## 📁 Dataset

The SCRSS is trained on the **DAWN (Detection in Adverse Weather Nature)** benchmark dataset, augmented with **synthetically generated sub-arctic winter imagery** to improve coverage of Saskatchewan-specific cryospheric conditions.

> The DAWN dataset is publicly available. Synthetic augmentation was applied using standard image transformation pipelines to simulate black ice glare, blowing snow obscuration, and frost accumulation patterns characteristic of Saskatoon winters.

---

## ❄️ Hazard Categories

The model classifies five cryospheric road surface pathology categories:

| # | Hazard | Description |
|---|--------|-------------|
| 1 | **Black Ice** | Transparent ice film on asphalt; highest accident risk |
| 2 | **Compacted Snow** | Dense, pressed snow surface with reduced traction |
| 3 | **Slush Formation** | Semi-liquid snow-water mixture; spray hazard |
| 4 | **Wet Frost** | Surface frost activated by moisture; low visibility cue |
| 5 | **Clear/Dry** | Baseline safe condition class |

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
| School of Computing, **University of Portsmouth**, UK | Lecturer |
| School of Computing, **Arden University**, Coventry, UK | Lecturer |
| School of Computing, **Pearson**, London, UK | Lecturer |
| School of Computing and Emerging Technologies, **IVY College of Management Sciences**, Lahore, Pakistan | Lecturer |

---

## 📜 License

This work is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

You are free to share and adapt this work for any purpose, provided appropriate credit is given.

---

<div align="center">

*Built for safer roads in sub-arctic cities. Open-source. Reproducible. Deployable.*

⭐ **Star this repository** if you find it useful!

</div>
