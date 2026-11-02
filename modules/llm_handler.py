"""Thin wrapper around the Gemini API (google-genai SDK)."""
import json
import re
import time

SYSTEM_PROMPT = """You are AgriGen AI, an agricultural advisory assistant for farmers in India,
with a focus on Telangana and Andhra Pradesh.
Rules:
1. Be practical, simple and honest. Say clearly when you are uncertain.
2. When context from agricultural documents is provided, base your answer on it and cite sources as [1], [2].
3. Never invent citations, statistics or document names.
4. Image-based findings are PRELIMINARY. Never state a disease diagnosis as certain.
5. Do NOT give specific pesticide/fungicide brand names, doses or mixing quantities unless they appear in the
   provided context. Otherwise recommend confirming with the local agriculture officer, KVK or university
   extension (e.g. ANGRAU, PJTSAU).
6. Prefer integrated pest management (cultural, biological, then chemical as last resort)."""


class LLMHandler:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY in .env or enter it in the sidebar.")
        from google import genai  # imported lazily
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt, system=SYSTEM_PROMPT, images=None, temperature=0.3,
                 json_mode=False, retries=3) -> str:
        from google.genai import types
        contents = list(images or []) + [prompt]
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
                return (resp.text or "").strip()
            except Exception as e:  # network / rate limit
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini request failed after {retries} attempts: {last_err}")

    def generate_json(self, prompt, **kw):
        return parse_json(self.generate(prompt, json_mode=True, **kw))


def parse_json(text: str):
    """Parse JSON from a model response, tolerating markdown fences and extra prose."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\{\[].*[\}\]]", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise
