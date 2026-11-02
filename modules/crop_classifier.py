"""OPTIONAL local PlantVillage-style classifier (baseline for experiment E1).

The main app uses Gemini Vision. This module lets the team compare Gemini against a
conventional pretrained image classifier from Hugging Face. Requires: pip install transformers torch.

NOTE: verify the model id on huggingface.co before use; swap in any image-classification model
fine-tuned on PlantVillage.
"""
from PIL import Image

DEFAULT_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
_pipe = None


def classify(img: Image.Image, model_id: str = DEFAULT_MODEL, top_k: int = 3):
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("image-classification", model=model_id)
    return [{"label": r["label"], "score": round(r["score"], 4)} for r in _pipe(img, top_k=top_k)]
