"""Evaluation harness for DS408 experiments E1-E5.

Run from the project root:
    python -m evaluation.run_evaluation --exp all        # or e1 / e2 / e3 / e4 / e5
Each experiment writes evaluation/results_<exp>.csv and appends a summary row to
evaluation/evaluation_results.csv. All numbers are measured from real runs - report them as-is.

E1 expects labelled images:  evaluation/test_images/<PlantVillage_label>/<image>.jpg
    e.g. evaluation/test_images/Tomato___Early_blight/img1.jpg   (use 5-10 images x 8-10 classes)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from modules import crop_advisor, image_analyzer, report_generator  # noqa: E402
from modules.llm_handler import LLMHandler  # noqa: E402
from modules.rag_engine import RAGEngine  # noqa: E402

EVAL = config.EVAL_DIR
PAUSE = 1.5  # seconds between API calls (free-tier rate limits)
QUESTIONS = json.loads((EVAL / "test_questions.json").read_text(encoding="utf-8"))


def log_summary(exp, metrics: dict):
    row = pd.DataFrame([{"experiment": exp, **metrics}])
    path = EVAL / "evaluation_results.csv"
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"\n== {exp} summary ==", metrics)


def judge(llm, question, answer, context, points):
    """LLM-as-judge: correctness & groundedness on a 1-5 scale."""
    prompt = (f"Question: {question}\nKey facts a correct answer should reflect: {points}\n"
              f"Retrieved context (may be empty): {context[:3000] or 'NONE'}\n\nAnswer to grade:\n{answer}\n\n"
              'Return ONLY JSON: {"correctness": 1-5, "groundedness": 1-5, "language_quality": 1-5}. '
              "correctness = agreement with the key facts; groundedness = claims supported by the context "
              "(1 if no context given and claims are unsupported); language_quality = fluent and natural.")
    time.sleep(PAUSE)
    return llm.generate_json(prompt, temperature=0.0)


def kw_hit(text, kws):
    text = text.lower()
    return sum(k.lower() in text for k in kws) / len(kws)


def e1(llm, rag):
    rows = []
    for folder in sorted(p for p in (EVAL / "test_images").iterdir() if p.is_dir()):
        crop, _, disease = folder.name.partition("___")
        tokens = [w for w in re.split(r"[_\s]+", disease.lower()) if len(w) > 2]
        for f in sorted(folder.glob("*")):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            res = image_analyzer.analyze_image(llm, image_analyzer.load_image(f.read_bytes()), "English", crop, "")
            names = [c.get("name", "").lower() for c in res.get("possible_conditions", [])]
            top1 = bool(names) and all(t in names[0] for t in tokens)
            top3 = any(all(t in n for t in tokens) for n in names)
            rows.append({"label": folder.name, "file": f.name, "pred": names[:3], "top1": top1, "top3": top3,
                         "crop_ok": crop.split("_")[0].lower() in res.get("crop_identified", "").lower()})
            time.sleep(PAUSE)
    df = pd.DataFrame(rows)
    df.to_csv(EVAL / "results_e1.csv", index=False)
    if len(df):
        log_summary("E1", {"n_images": len(df), "top1_acc": df.top1.mean(), "top3_acc": df.top3.mean(),
                           "crop_id_acc": df.crop_ok.mean()})
    else:
        print("No test images found in evaluation/test_images/<label>/")


def e2(llm, rag):
    rows = []
    for q in QUESTIONS:
        chunks = rag.search(q["question"])
        hits = [kw_hit(c["text"], q["expected_keywords"]) > 0 for c in chunks]
        rows.append({"id": q["id"], "hit_at_k": any(hits), "precision_at_k": sum(hits) / max(len(hits), 1),
                     "top_source": chunks[0]["source"] if chunks else ""})
    df = pd.DataFrame(rows)
    df.to_csv(EVAL / "results_e2.csv", index=False)
    log_summary("E2", {"hit_rate@k": df.hit_at_k.mean(), "precision@k": df.precision_at_k.mean(), "k": config.TOP_K})


def e3(llm, rag):
    rows = []
    for q in QUESTIONS:
        chunks = rag.search(q["question"])
        ctx = crop_advisor.format_context(chunks)
        for mode, ch in (("no_rag", []), ("rag", chunks)):
            ans = crop_advisor.answer_question(llm, q["question"], ch, "English")
            s = judge(llm, q["question"], ans, ctx if ch else "", q["expected_keywords"])
            rows.append({"id": q["id"], "mode": mode, "keyword_coverage": kw_hit(ans, q["expected_keywords"]), **s})
    df = pd.DataFrame(rows)
    df.to_csv(EVAL / "results_e3.csv", index=False)
    agg = df.groupby("mode")[["keyword_coverage", "correctness", "groundedness"]].mean().round(3)
    print(agg)
    for mode, r in agg.iterrows():
        log_summary(f"E3-{mode}", r.to_dict())


def e4(llm, rag):
    rows = []
    for q in QUESTIONS[:6]:
        chunks = rag.search(q["question"])
        ctx = crop_advisor.format_context(chunks)
        for lang in config.LANGUAGES:
            ans = crop_advisor.answer_question(llm, q["question"], chunks, lang)
            s = judge(llm, q["question"], ans, ctx, q["expected_keywords"])
            rows.append({"id": q["id"], "language": lang, **s})
    df = pd.DataFrame(rows)
    df.to_csv(EVAL / "results_e4.csv", index=False)
    agg = df.groupby("language")[["correctness", "groundedness", "language_quality"]].mean().round(3)
    print(agg)
    for lang, r in agg.iterrows():
        log_summary(f"E4-{lang}", r.to_dict())


def e5(llm, rag):
    scenarios = [
        "Crop: Tomato, stage: fruiting, location: Karimnagar. Brown rings on lower leaves, leaves yellowing and dropping.",
        "Crop: Paddy, stage: flowering, location: Guntur. Spindle-shaped grey spots on leaves, humid weather.",
        "Crop: Chilli, stage: vegetative, location: Khammam. Curled small leaves, stunted growth, whiteflies seen.",
    ]
    rows = []
    for s in scenarios:
        rep = crop_advisor.compose_report(llm, s, "English")
        complete = sum(bool(rep[k]) for k in rep) / len(rep)
        pdf = report_generator.build_pdf({"profile": {"Scenario": s}, **rep, "references": []})
        grade = llm.generate_json(
            f"Scenario: {s}\nReport: {rep}\nReturn ONLY JSON {{\"relevance\":1-5,\"actionability\":1-5,\"safety\":1-5}} "
            "(safety=5 if no unsafe pesticide doses or overconfident diagnosis).", temperature=0.0)
        rows.append({"scenario": s[:40], "section_completeness": complete, "pdf_bytes": len(pdf), **grade})
        time.sleep(PAUSE)
    df = pd.DataFrame(rows)
    df.to_csv(EVAL / "results_e5.csv", index=False)
    log_summary("E5", df[["section_completeness", "relevance", "actionability", "safety"]].mean().round(3).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", "e1", "e2", "e3", "e4", "e5"])
    args = ap.parse_args()
    llm = LLMHandler(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    rag = RAGEngine()
    if rag.index is None:
        rag.build()
    runs = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}
    for name in (runs if args.exp == "all" else [args.exp]):
        print(f"\n>>> Running {name.upper()} ...")
        runs[name](llm, rag)
