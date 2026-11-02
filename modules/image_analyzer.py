"""Crop image analysis: OpenCV quality checks + Gemini Vision preliminary assessment."""
import io

import cv2
import numpy as np
from PIL import Image

from modules.multilingual import language_instruction

MAX_SIDE = 1024


def load_image(file_or_bytes) -> Image.Image:
    data = file_or_bytes.read() if hasattr(file_or_bytes, "read") else file_or_bytes
    return Image.open(io.BytesIO(data)).convert("RGB")


def preprocess(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Downscale large photos (faster, cheaper API calls) while keeping aspect ratio."""
    w, h = img.size
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS) if scale < 1 else img


def assess_quality(img: Image.Image) -> dict:
    """Simple OpenCV checks so we can warn the farmer about unreliable photos."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    green_ratio = float(green_mask.mean() / 255)
    warnings = []
    if blur < 60:
        warnings.append("Image looks blurry - retake it in focus.")
    if brightness < 50:
        warnings.append("Image is too dark - photograph in daylight.")
    elif brightness > 220:
        warnings.append("Image is overexposed - avoid direct harsh sunlight.")
    if min(img.size) < 224:
        warnings.append("Image resolution is low - move closer to the affected leaf.")
    if green_ratio < 0.05:
        warnings.append("Very little green plant tissue detected - make sure the leaf fills the frame.")
    return {"blur_score": round(blur, 1), "brightness": round(brightness, 1),
            "green_ratio": round(green_ratio, 3), "warnings": warnings}


def analyze_image(llm, img: Image.Image, language="English", crop_hint="", notes="") -> dict:
    """Return a structured, preliminary assessment (JSON) from the vision model."""
    img = preprocess(img)
    prompt = (
        "Examine this crop/leaf photograph as an agricultural plant-health assistant.\n"
        f"Farmer's crop hint: {crop_hint or 'not given'}\nFarmer's notes: {notes or 'none'}\n\n"
        "Return ONLY a JSON object with these keys:\n"
        '  "crop_identified": string,\n'
        '  "plant_part": string,\n'
        '  "visible_symptoms": list of short strings (only what is actually visible),\n'
        '  "possible_conditions": list (max 3, most likely first) of objects with keys '
        '"name" (condition name in English, use "Healthy" if no problem is visible), '
        '"confidence" ("low"|"medium"|"high"), "reasoning" (one sentence),\n'
        '  "severity_estimate": "none"|"mild"|"moderate"|"severe"|"unclear",\n'
        '  "image_limitations": list of strings (blur, lighting, missing views, etc.),\n'
        '  "next_steps": list of safe, practical steps (e.g. inspect other plants, isolate, '
        "consult agriculture officer; no chemical doses),\n"
        '  "needs_expert_confirmation": boolean.\n'
        "Do not overstate certainty. Condition names stay in English; every other text value must follow "
        f"this language instruction: {language_instruction(language)}"
    )
    result = llm.generate_json(prompt, images=[img], temperature=0.2)
    if isinstance(result, list):
        result = result[0] if result else {}
    result.setdefault("possible_conditions", [])
    result.setdefault("needs_expert_confirmation", True)
    return result


def top_condition(result: dict) -> str:
    conds = result.get("possible_conditions") or []
    return conds[0].get("name", "") if conds else ""
