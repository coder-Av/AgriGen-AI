"""AgriGen AI - Intelligent Multimodal Generative-AI Agricultural Advisory System (Streamlit)."""
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from modules import crop_advisor, image_analyzer, report_generator
from modules.llm_handler import LLMHandler
from modules.multilingual import LANG_LIST, t
from modules.rag_engine import RAGEngine

st.set_page_config(page_title="AgriGen AI", page_icon="🌾", layout="wide")


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_rag() -> RAGEngine:
    rag = RAGEngine()
    if rag.index is None and any(config.DOCS_DIR.iterdir()):
        rag.build()
    return rag


@st.cache_resource
def get_llm(key: str, model: str) -> LLMHandler:
    return LLMHandler(key, model)


def ss(key, default):
    return st.session_state.setdefault(key, default)


# ---------- sidebar ----------
lang = st.sidebar.selectbox("🌐 Language / భాష / भाषा", LANG_LIST)
api_key = config.GEMINI_API_KEY
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
if not api_key:
    api_key = st.sidebar.text_input("Gemini API key", type="password")
model_name = st.sidebar.text_input("Gemini model", config.GEMINI_MODEL)
use_rag = st.sidebar.toggle("Use knowledge base (RAG)", value=True)

PAGES = ["home", "image", "chat", "kb", "advisory", "reports"]
ICONS = {"home": "🏠", "image": "📷", "chat": "💬", "kb": "📚", "advisory": "🌱", "reports": "📄"}
page = st.sidebar.radio("Menu", PAGES, format_func=lambda k: f"{ICONS[k]} {t(lang, k)}")
st.sidebar.caption(t(lang, "disclaimer"))

llm = get_llm(api_key, model_name) if api_key else None
rag = get_rag()

ss("chat", [])
ss("image_result", None)
ss("image_explanation", "")
ss("profile", {})
ss("advice", "")
ss("sources", [])


def retrieve(query):
    if not use_rag:
        return []
    return rag.search(query)


def need_llm() -> bool:
    if llm is None:
        st.warning("Enter your Gemini API key in the sidebar (or set GEMINI_API_KEY in .env).")
        return False
    return True


st.title(f"🌾 {t(lang, 'app_title')}")
st.caption(t(lang, "tagline"))

# ---------- HOME ----------
if page == "home":
    stats = rag.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(stats))
    c2.metric("Knowledge chunks", int(stats["chunks"].sum()) if len(stats) else 0)
    c3.metric("Language", lang)
    c4.metric("Model", model_name)
    st.markdown("### Modules")
    cols = st.columns(3)
    mods = [("📷 Crop Image Analysis", "Upload a leaf photo for a preliminary, uncertainty-aware assessment."),
            ("💬 Ask AgriGen", "Chat about irrigation, soil, fertilizer and crop management."),
            ("📚 Knowledge Base", "Upload ICAR / ANGRAU / TNAU PDFs and ask questions with citations (RAG)."),
            ("🌐 Multilingual", "English, Telugu and Hindi questions and answers."),
            ("🌱 Crop Advisory", "Recommendations from crop, stage, location, soil and symptoms."),
            ("📄 Reports", "Structured, downloadable PDF reports.")]
    for i, (h, d) in enumerate(mods):
        cols[i % 3].info(f"**{h}**\n\n{d}")
    if len(stats):
        st.plotly_chart(px.bar(stats, x="source", y="chunks", title="Knowledge base: chunks per document"),
                        use_container_width=True)
    else:
        st.warning("Knowledge base is empty. Add documents in the Knowledge Base page.")

# ---------- IMAGE ANALYSIS ----------
elif page == "image":
    st.subheader(f"📷 {t(lang, 'image')}")
    up = st.file_uploader(t(lang, "upload_image"), type=["jpg", "jpeg", "png", "webp"])
    c1, c2 = st.columns(2)
    crop_hint = c1.text_input("Crop (optional)")
    notes = c2.text_input("What have you noticed? (optional)")
    if up:
        img = image_analyzer.load_image(up)
        st.image(img, width=380)
        q = image_analyzer.assess_quality(img)
        for w in q["warnings"]:
            st.warning(w)
        if st.button(t(lang, "analyze"), type="primary") and need_llm():
            with st.spinner("Analyzing image..."):
                res = image_analyzer.analyze_image(llm, img, lang, crop_hint, notes)
                query = f"{res.get('crop_identified', crop_hint)} {image_analyzer.top_condition(res)} symptoms management"
                chunks = retrieve(query)
                st.session_state.image_result = res
                st.session_state.image_explanation = crop_advisor.explain_image(llm, res, chunks, lang)
                st.session_state.sources = crop_advisor.sources_list(chunks)
    res = st.session_state.image_result
    if res:
        st.markdown(f"**Crop:** {res.get('crop_identified', '-')}  |  **Part:** {res.get('plant_part', '-')}  |  "
                    f"**Severity:** {res.get('severity_estimate', '-')}")
        if res.get("possible_conditions"):
            st.dataframe(pd.DataFrame(res["possible_conditions"]), hide_index=True, use_container_width=True)
        if res.get("visible_symptoms"):
            st.markdown("**Visible symptoms**\n" + "\n".join(f"- {s}" for s in res["visible_symptoms"]))
        if res.get("image_limitations"):
            st.caption("Limitations: " + "; ".join(map(str, res["image_limitations"])))
        if res.get("needs_expert_confirmation", True):
            st.info("This is a preliminary assessment. Please confirm with an agriculture expert.")
        st.markdown("### Explanation")
        st.write(st.session_state.image_explanation)
        if st.session_state.sources:
            with st.expander("Sources"):
                st.write("\n".join(f"- {s}" for s in st.session_state.sources))

# ---------- CHAT ----------
elif page == "chat":
    st.subheader(f"💬 {t(lang, 'chat')}")
    for role, msg, srcs in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)
            if srcs:
                with st.expander("Sources"):
                    st.write("\n".join(f"- {s}" for s in srcs))
    q = st.chat_input(t(lang, "ask_placeholder"))
    if q and need_llm():
        with st.chat_message("user"):
            st.write(q)
        chunks = retrieve(q)
        hist = [(r, m) for r, m, _ in st.session_state.chat]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = crop_advisor.answer_question(llm, q, chunks, lang, hist)
            st.write(ans)
            srcs = crop_advisor.sources_list(chunks)
            if srcs:
                with st.expander("Sources"):
                    st.write("\n".join(f"- {s}" for s in srcs))
        st.session_state.chat += [("user", q, []), ("assistant", ans, srcs)]
        st.session_state.sources = srcs
    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()

# ---------- KNOWLEDGE BASE ----------
elif page == "kb":
    st.subheader(f"📚 {t(lang, 'kb')}")
    files = st.file_uploader("Upload PDFs / text files (ICAR, ANGRAU, TNAU, FAO...)",
                             type=["pdf", "txt", "md"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    if files and c1.button("Add to knowledge base", type="primary"):
        with st.spinner("Extracting text and building embeddings..."):
            n = sum(rag.add_document(f.name, f.getvalue()) for f in files)
        st.success(f"Indexed {n} chunks.")
    if c2.button("Rebuild entire index"):
        with st.spinner("Rebuilding..."):
            st.success(f"Indexed {rag.build()} chunks.")
    st.dataframe(rag.stats(), hide_index=True, use_container_width=True)
    q = st.text_input("Ask a question about your documents")
    if q and need_llm():
        chunks = rag.search(q)
        with st.spinner("Answering..."):
            st.write(crop_advisor.answer_question(llm, q, chunks, lang))
        with st.expander("Retrieved passages"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**[{i}] {c['source']} p.{c['page']}** (similarity {c['score']:.2f})")
                st.caption(c["text"])
        st.session_state.sources = crop_advisor.sources_list(chunks)

# ---------- ADVISORY ----------
elif page == "advisory":
    st.subheader(f"🌱 {t(lang, 'advisory')}")
    with st.form("advisory"):
        c1, c2, c3 = st.columns(3)
        crop = c1.selectbox("Crop", config.CROPS)
        stage = c2.selectbox("Growth stage", config.GROWTH_STAGES)
        season = c3.selectbox("Season", config.SEASONS)
        c4, c5, c6 = st.columns(3)
        location = c4.text_input("Village / District", placeholder="e.g. Warangal, Telangana")
        soil = c5.selectbox("Soil type", config.SOILS)
        irrigation = c6.selectbox("Irrigation", config.IRRIGATION)
        symptoms = st.text_area("Symptoms / problem (optional)")
        go = st.form_submit_button("Get advisory", type="primary")
    if go and need_llm():
        profile = {"Crop": crop, "Growth stage": stage, "Season": season, "Location": location,
                   "Soil type": soil, "Irrigation": irrigation, "Symptoms": symptoms}
        chunks = retrieve(f"{crop} {stage} {symptoms} management {location}")
        with st.spinner("Preparing advisory..."):
            st.session_state.advice = crop_advisor.advise(llm, profile, chunks, lang,
                                                          st.session_state.image_result)
        st.session_state.profile = profile
        st.session_state.sources = crop_advisor.sources_list(chunks)
    if st.session_state.advice:
        st.write(st.session_state.advice)
        if st.session_state.sources:
            with st.expander("Sources"):
                st.write("\n".join(f"- {s}" for s in st.session_state.sources))

# ---------- REPORTS ----------
elif page == "reports":
    st.subheader(f"📄 {t(lang, 'reports')}")
    st.caption("The report combines your latest crop details, image analysis, advisory and chat.")
    rep_lang = st.selectbox("Report language", LANG_LIST, index=LANG_LIST.index(lang))
    if not report_generator.report_language_ok(rep_lang):
        st.warning(f"Add the {rep_lang} Noto font to assets/fonts/ (see README). The PDF will be in English until then.")
    material = []
    if st.session_state.profile:
        material.append(f"Crop details: {st.session_state.profile}")
    if st.session_state.image_result:
        material.append(f"Image findings: {st.session_state.image_result}")
    if st.session_state.advice:
        material.append(f"Advisory given:\n{st.session_state.advice}")
    if st.session_state.chat:
        material.append("Farmer Q&A:\n" + "\n".join(f"{r}: {m}" for r, m, _ in st.session_state.chat[-8:]))
    if not material:
        st.info("Nothing to report yet. Run an image analysis, advisory or chat first.")
    elif st.button(t(lang, "generate"), type="primary") and need_llm():
        eff_lang = rep_lang if report_generator.report_language_ok(rep_lang) else "English"
        with st.spinner("Composing report..."):
            sections = crop_advisor.compose_report(llm, "\n\n".join(material), eff_lang)
            report = {"profile": st.session_state.profile, "image_findings": st.session_state.image_result,
                      "references": st.session_state.sources, **sections}
            st.session_state.pdf = report_generator.build_pdf(report, eff_lang)
    if st.session_state.get("pdf"):
        st.success("Report ready.")
        st.download_button(f"⬇ {t(lang, 'download')} PDF", st.session_state.pdf,
                           file_name="agrigen_report.pdf", mime="application/pdf")
