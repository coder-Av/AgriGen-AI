"""Language helpers: UI strings and LLM language instructions (English, Telugu, Hindi)."""
import config

LANG_INSTRUCTIONS = {
    "English": "Respond in clear, simple English suitable for a farmer.",
    "Telugu": ("Respond entirely in Telugu (తెలుగు) using Telugu script, in simple words a farmer "
               "understands. Keep crop, disease and chemical names in English in brackets when helpful."),
    "Hindi": ("Respond entirely in Hindi (हिन्दी) using Devanagari script, in simple words a farmer "
              "understands. Keep crop, disease and chemical names in English in brackets when helpful."),
}

UI = {
    "English": {
        "app_title": "AgriGen AI", "tagline": "Intelligent Multimodal Agricultural Advisory System",
        "home": "Home", "image": "Crop Image Analysis", "chat": "Ask AgriGen",
        "kb": "Knowledge Base", "advisory": "Crop Advisory", "reports": "Reports",
        "upload_image": "Upload a crop / leaf image", "analyze": "Analyze",
        "ask_placeholder": "Ask your farming question here...",
        "disclaimer": "This is preliminary guidance only. For pesticide or fertilizer decisions, consult your local agriculture officer / KVK.",
        "rep_profile": "Crop Details", "rep_image": "Image Analysis", "rep_obs": "Observations",
        "rep_causes": "Possible Causes", "rep_reco": "Recommendations", "rep_prec": "Precautions",
        "rep_refs": "References", "generate": "Generate", "download": "Download",
    },
    "Telugu": {
        "app_title": "అగ్రిజెన్ AI", "tagline": "తెలివైన వ్యవసాయ సలహా వ్యవస్థ",
        "home": "హోమ్", "image": "పంట చిత్ర విశ్లేషణ", "chat": "అగ్రిజెన్‌ను అడగండి",
        "kb": "జ్ఞాన భాండారం", "advisory": "పంట సలహా", "reports": "నివేదికలు",
        "upload_image": "పంట చిత్రాన్ని అప్‌లోడ్ చేయండి", "analyze": "విశ్లేషించండి",
        "ask_placeholder": "మీ వ్యవసాయ ప్రశ్నను ఇక్కడ అడగండి...",
        "disclaimer": "ఇది ప్రాథమిక సలహా మాత్రమే. పురుగుమందులు లేదా ఎరువుల విషయంలో స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.",
        "rep_profile": "పంట వివరాలు", "rep_image": "చిత్ర విశ్లేషణ", "rep_obs": "పరిశీలనలు",
        "rep_causes": "సాధ్యమయ్యే కారణాలు", "rep_reco": "సిఫార్సులు", "rep_prec": "జాగ్రత్తలు",
        "rep_refs": "ఆధారాలు", "generate": "రూపొందించండి", "download": "డౌన్‌లోడ్ చేయండి",
    },
    "Hindi": {
        "app_title": "एग्रीजेन AI", "tagline": "बुद्धिमान कृषि सलाह प्रणाली",
        "home": "होम", "image": "फसल छवि विश्लेषण", "chat": "AgriGen से पूछें",
        "kb": "ज्ञान आधार", "advisory": "फसल सलाह", "reports": "रिपोर्ट",
        "upload_image": "फसल की तस्वीर अपलोड करें", "analyze": "विश्लेषण करें",
        "ask_placeholder": "अपना कृषि प्रश्न यहाँ पूछें...",
        "disclaimer": "यह केवल प्रारंभिक सलाह है। कीटनाशक या उर्वरक के लिए स्थानीय कृषि अधिकारी से परामर्श करें।",
        "rep_profile": "फसल विवरण", "rep_image": "छवि विश्लेषण", "rep_obs": "अवलोकन",
        "rep_causes": "संभावित कारण", "rep_reco": "सिफारिशें", "rep_prec": "सावधानियाँ",
        "rep_refs": "संदर्भ", "generate": "तैयार करें", "download": "डाउनलोड करें",
    },
}


def t(language: str, key: str) -> str:
    """UI string lookup with English fallback."""
    return UI.get(language, UI["English"]).get(key) or UI["English"].get(key, key)


def language_instruction(language: str) -> str:
    return LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS["English"])


LANG_LIST = list(config.LANGUAGES)
