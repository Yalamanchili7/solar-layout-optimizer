*# ☀️ AI-Based Solar Layout Optimizer

An **AI-powered geospatial layout optimization tool** for **utility-scale solar project development**.  
This application analyzes buildable parcels, applies environmental & regulatory constraints, and generates optimal solar tracker layouts with estimated AC capacity — all visualized interactively.

---

## 🌟 Features
- 📍 Upload and parse **KML files** with parcel boundaries and constraint layers
- 🚧 Apply setbacks, rivers, roads, and environmental exclusions to determine buildable areas
- ⚡ Estimate **nominal & AI-optimized AC capacity**
- 🤖 Generate realistic tracker row layouts with AI assistance
- 🗺️ Visualize buildable areas and layouts on an interactive map
- 📊 Compare scenarios and download results

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Yalamanchili7/solar-layout-optimizer.git
cd solar-layout-optimizer


pip install -r requirements.txt


streamlit run app.py



├── app.py                    # Streamlit app entry point
├── ai_layout_optimizer.py    # AI optimization & layout logic
├── layout_optimizer.py       # Tracker row generation algorithms
├── map_renderer.py           # Map visualization using Folium
├── kml_parser.py             # KML parsing utilities
├── capacity.py               # Capacity calculation helpers
├── model_training.py         # ML model training & prediction
├── requirements.txt          # Python dependencies
├── data/
│   └── Module_and_CapEx_250409.xlsx  # Module specs & CAPEX data
├── models/                   # Pre-trained ML models
│   └── sy_gbdt_model.pkl
│   └── sy_scaler.pkl
├── README.md                 # This file
