"""Central configuration for AgriGen AI."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:  # dotenv is optional
    pass

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "agriculture_documents"
IMAGES_DIR = BASE_DIR / "data" / "crop_images"
VECTOR_DIR = BASE_DIR / "vector_db"
REPORTS_DIR = BASE_DIR / "reports"
FONT_DIR = BASE_DIR / "assets" / "fonts"
EVAL_DIR = BASE_DIR / "evaluation"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Multilingual embedding model: handles English, Telugu and Hindi queries
# against English documents.
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 800      # characters
CHUNK_OVERLAP = 150
TOP_K = 4

LANGUAGES = {"English": "en", "Telugu": "te", "Hindi": "hi"}

CROPS = ["Paddy (Rice)", "Cotton", "Maize", "Chilli", "Tomato", "Groundnut",
         "Red gram (Pigeon pea)", "Soybean", "Turmeric", "Sugarcane", "Other"]
GROWTH_STAGES = ["Nursery / Germination", "Vegetative", "Flowering",
                 "Fruiting / Grain filling", "Maturity / Harvest"]
SEASONS = ["Kharif (Jun-Oct)", "Rabi (Nov-Mar)", "Summer (Mar-Jun)"]
SOILS = ["Red soil", "Black cotton soil", "Alluvial", "Sandy loam", "Clay loam", "Not sure"]
IRRIGATION = ["Rainfed", "Canal", "Borewell / Tubewell", "Drip", "Sprinkler"]

for _d in (DOCS_DIR, IMAGES_DIR, VECTOR_DIR, REPORTS_DIR, FONT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
