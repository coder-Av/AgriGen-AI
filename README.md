# 🌾 AgriGen AI
**An Intelligent Multimodal Generative AI-Based Agricultural Advisory System** — DS408 Generative AI

Group: **AgriTech Innovators** · Leader: Avusula Vamshi

| S.No | Student | ID |
|---|---|---|
| 1 | Avusula Vamshi | 23STUCHH010163 |
| 2 | Thipparthi Sri Sai Kumar | 23STUCHH010197 |
| 3 | Nalla Koteshwar Reddy | 23STUCHH010239 |
| 4 | Irinkileela Dattatreya | 23STUCHH010159 |

## Features
Crop image analysis (Gemini Vision + OpenCV quality checks) · Agricultural chatbot · RAG knowledge base over PDFs (FAISS + multilingual Sentence-Transformers) · English/Telugu/Hindi · Personalized crop advisory · PDF reports.

Image output is always a **preliminary, uncertainty-aware assessment**, never a confirmed diagnosis. The system prompt forbids inventing pesticide doses unless they appear in retrieved sources.

## Setup (VS Code, Python 3.12)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then put your key in .env  (get one at aistudio.google.com)
streamlit run app.py
```
The first run downloads the embedding model (~470 MB) and indexes `data/agriculture_documents/`.

## Before the demo
1. **Replace the SAMPLE_*.md documents** with real ICAR / ANGRAU / PJTSAU / TNAU / FAO PDFs (drop them in `data/agriculture_documents/` or upload in the app). Prioritise Telangana & AP crops.
2. **Telugu/Hindi PDF reports** need fonts: download *Noto Sans Telugu* and *Noto Sans Devanagari* from Google Fonts and save as `assets/fonts/NotoSansTelugu-Regular.ttf` and `assets/fonts/NotoSansDevanagari-Regular.ttf`. Without them, PDFs are English (on-screen answers are unaffected).
3. **Image test set (E1):** download PlantVillage from Kaggle, then copy 5–10 images per class into `evaluation/test_images/<Label>/` (e.g. `Tomato___Early_blight`). Use images the model has not been tuned on.

## Experiments
```bash
python -m evaluation.run_evaluation --exp all    # or e1 | e2 | e3 | e4 | e5
```
Writes `evaluation/results_*.csv` and `evaluation/evaluation_results.csv`. Compare no-RAG vs RAG (E3), languages (E4). Use Plotly/Excel to chart your measured results for the report. Judge scores come from an LLM-as-judge: state this limitation in the report, and spot-check some answers manually.

## Structure
```
app.py            Streamlit UI (6 sections)
config.py         settings, paths, crop lists
modules/          llm_handler, image_analyzer, crop_classifier (optional baseline), rag_engine,
                  multilingual, crop_advisor, report_generator
data/             crop_images/, agriculture_documents/
vector_db/        FAISS index (auto-generated)
evaluation/       test_questions.json, run_evaluation.py, test_images/
```

## Team split (suggested)
Vamshi – architecture, LLM integration · Sri Sai Kumar – image data/vision · Koteshwar Reddy – documents & RAG · Dattatreya – frontend & multilingual · everyone – testing, evaluation, report.

## Limitations (mention in viva)
Preliminary image assessment only; depends on photo quality; general-purpose LLM may err in local agronomy; RAG answers are only as good as the documents; Telugu/Hindi quality should be measured (E4).
