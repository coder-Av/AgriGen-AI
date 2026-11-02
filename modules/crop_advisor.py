"""Prompt orchestration: grounded Q&A, personalized advisory, and report composition."""
from modules.multilingual import language_instruction

NO_CONTEXT_NOTE = ("No matching documents were found in the knowledge base. Answer from general agricultural "
                   "knowledge, state clearly that this is NOT backed by the uploaded sources, and recommend "
                   "confirming with a local agriculture officer.")


def format_context(chunks) -> str:
    return "\n\n".join(f"[{i}] ({c['source']}, p.{c['page']}) {c['text']}" for i, c in enumerate(chunks, 1))


def sources_list(chunks):
    seen, out = set(), []
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            out.append(f"{c['source']} (page {c['page']})")
    return out


def answer_question(llm, question, chunks, language="English", history=None) -> str:
    hist = ""
    for role, msg in (history or [])[-6:]:
        hist += f"{role.upper()}: {msg}\n"
    context = format_context(chunks) if chunks else NO_CONTEXT_NOTE
    prompt = (f"Conversation so far:\n{hist or '(none)'}\n"
              f"CONTEXT FROM AGRICULTURAL DOCUMENTS:\n{context}\n\n"
              f"FARMER'S QUESTION: {question}\n\n"
              f"Answer helpfully and concisely, citing sources like [1] where used. "
              f"{language_instruction(language)}")
    return llm.generate(prompt)


def advise(llm, profile: dict, chunks, language="English", image_findings=None) -> str:
    details = "\n".join(f"- {k}: {v}" for k, v in profile.items() if v)
    img = f"\nPRELIMINARY IMAGE FINDINGS (uncertain): {image_findings}\n" if image_findings else ""
    context = format_context(chunks) if chunks else NO_CONTEXT_NOTE
    prompt = (f"Prepare a personalized crop advisory.\nFARM / CROP DETAILS:\n{details}\n{img}\n"
              f"CONTEXT FROM AGRICULTURAL DOCUMENTS:\n{context}\n\n"
              "Structure: 1) Situation summary 2) Likely causes (with uncertainty) 3) What to do now "
              "(cultural / biological / chemical-last-resort) 4) Preventive practices for this growth stage "
              "5) When to call an agriculture officer. Cite sources like [1] where used. "
              + language_instruction(language))
    return llm.generate(prompt)


def explain_image(llm, findings: dict, chunks, language="English") -> str:
    context = format_context(chunks) if chunks else NO_CONTEXT_NOTE
    prompt = (f"A vision model produced these PRELIMINARY findings from a crop photo:\n{findings}\n\n"
              f"CONTEXT FROM AGRICULTURAL DOCUMENTS:\n{context}\n\n"
              "Explain to the farmer in simple terms what this may mean and what safe steps to take. "
              "Make clear it is not a confirmed diagnosis. Cite sources like [1] where used. "
              + language_instruction(language))
    return llm.generate(prompt)


def compose_report(llm, material: str, language="English") -> dict:
    """Ask the LLM to turn session material into structured report sections (JSON)."""
    prompt = ("Create a structured agricultural report from the material below.\n\nMATERIAL:\n" + material +
              "\n\nReturn ONLY a JSON object with keys: "
              '"observations" (list of strings), "possible_causes" (list of strings), '
              '"recommendations" (list of strings), "precautions" (list of strings). '
              "Do not invent facts that are not in the material; mark uncertainty. "
              + language_instruction(language))
    data = llm.generate_json(prompt, temperature=0.2)
    return {k: [str(x) for x in (data.get(k) or [])]
            for k in ("observations", "possible_causes", "recommendations", "precautions")}
