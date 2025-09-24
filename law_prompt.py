from textwrap import dedent

def chunk_extraction_prompt(data: str, case_id: str, pages: str = "") -> str:
    return dedent(f"""
    You are a judicial law clerk. Extract only what is present in the text. If unknown, use null or [].
    Quote and cite paragraph/page numbers whenever you can. Never invent facts.

    Output strict JSON with this schema:
    {{
      "case_id": "{case_id}",
      "pages": "{pages}",
      "chunk_summary": "3–5 sentence summary of this chunk only.",
      "facts": ["short fact … [para x]"],
      "procedural_history": ["event … [para/page]"],
      "issues": ["legal question …"],
      "holdings": ["answer to issue …"],
      "rules": ["statute/case rule …"],
      "reasoning": ["court’s reasoning …"],
      "orders": ["order/disposition …"],
      "parties": {{"plaintiffs": [], "defendants": [], "judges": []}},
      "citations": ["case/statute citation …"],
      "entities": ["person/org/place …"],
      "timeline": [{{"date":"YYYY-MM-DD","event":"…","source":"[para/page]"}}],
      "quotes": [{{"text":"exact quote", "source":"[para/page]"}}],
      "confidence": 0.0
    }}

    Constraints:
    - Be extractive; prefer quotes with [para n] or [p n].
    - Keep lists concise (<= 20 items). No commentary outside JSON.

    Text:
    {data}
    """).strip()

def case_synthesis_prompt(extracted_chunks_json: str, case_id: str) -> str:
    return dedent(f"""
    You are preparing a law-report entry. Merge multiple per-chunk JSON records for the same case.

    Produce a single strict JSON object with:
    {{
      "case_id": "{case_id}",
      "headnote": "150–250 words; who/what/why/how/holding; neutral; no fluff.",
      "parties": {{"plaintiffs": [], "defendants": [], "judges": []}},
      "facts": ["…"],
      "procedural_history": ["…"],
      "issues": ["…"],
      "holdings": ["…"],
      "rules": ["…"],
      "reasoning": ["…"],
      "orders": ["…"],
      "citations": ["…"],
      "entities": ["…"],
      "timeline": [{{"date":"YYYY-MM-DD","event":"…","source":"[para/page]"}}],
      "key_quotes": [{{"text":"…","source":"[para/page]"}}],
      "disposition": "e.g., appeal dismissed; costs …",
      "summary_irac": {{
        "issue": ["…"], "rule": ["…"], "analysis": ["…"], "conclusion": ["…"]
      }}
    }}

    Requirements:
    - Deduplicate and reconcile names; sort timeline chronologically.
    - Keep only facts present in the chunks; include sources like [para n] or page ranges.
    - No extra commentary outside JSON.

    Chunks:
    {extracted_chunks_json}
    """).strip()

def qa_from_case_prompt(structured_case_json: str, n: int = 15) -> str:
    return dedent(f"""
    Create {n} diverse, short Q&A pairs strictly grounded in the structured case JSON.
    Mix: facts, procedure, issues, holdings, rules, orders, timeline. Include a [source] like [para n] when possible.

    Output strict JSON array:
    [{{"question":"…","answer":"…","source":"[para/page]"}}]

    Case:
    {structured_case_json}
    """).strip()

def answer_question_prompt(question: str, structured_case_json: str) -> str:
    return dedent(f"""
    You are a judicial law clerk. Answer the question strictly from the structured case JSON.
    If the answer is not present, reply exactly: "Not found in the provided case data."

    Include supporting quotes with [para/page] when available.

    Output strict JSON:
    {{
      "answer": "...",
      "support": ["exact quote … [para/page]"],
      "fields_consulted": ["facts","issues","holdings","rules","procedural_history","timeline","orders","citations","parties","key_quotes"],
      "confidence": 0.0
    }}

    Case:
    {structured_case_json}

    Question:
    {question}
    """).strip()

# Optional: simple single-pass summarization (when the whole case fits context)
def full_case_summary_prompt(text: str, target_words: int = 200) -> str:
    return dedent(f"""
    Summarize the following judgment into a {target_words}-word headnote plus IRAC sections.
    Output JSON:
    {{
      "headnote": "…",
      "irac": {{"issue":["…"],"rule":["…"],"analysis":["…"],"conclusion":["…"]}}
    }}
    Text:
    {text}
    """).strip()


if __name__ == "__main__":
    print(chunk_extraction_prompt("example case chunk text …", case_id="J4/03/2024", pages="pp. 1–3"))