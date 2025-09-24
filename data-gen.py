import torch
import warnings
import os
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore

import json
from pathlib import Path
from litellm import completion
from law_prompt import (
    chunk_extraction_prompt,
    case_synthesis_prompt,
    qa_from_case_prompt,
    answer_question_prompt
)

# ...existing code...
# Suppress MPS warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# --- LLM helper --------------------------------------------------------------
def llm_json(prompt: str, max_tokens: int = 2500) -> dict:
    """
    Calls the LLM and returns parsed JSON. Also strips accidental code fences.
    """
    stream = completion(
        model="ollama_chat/qwen2.5:14b",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={"num_predict": max_tokens},
    )
    buf = ""
    for x in stream:
        delta = x["choices"][0]["delta"].get("content")
        if delta:
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            buf += delta

    s = buf.strip()
    # Remove code fences if present
    if s.startswith("```"):
        first_nl = s.find("\n")
        last_fence = s.rfind("```")
        if first_nl != -1 and last_fence != -1:
            s = s[first_nl + 1:last_fence].strip()

    return json.loads(s)

# ...existing code...
if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(Fore.GREEN + "Using Metal Performance Shaders (GPU)" + Fore.RESET)
    else:
        device = torch.device("cpu")
        print(Fore.YELLOW + "Using CPU" + Fore.RESET)

    pdf_path = "cases/ACHEAMFOUR_GROUP_LTD_&_3_ORS_VRS_ANOKYE_&_4_ORS_(J4-03-2024)_[2024]_GHASC_58_(4_December_2024).pdf"
    case_id = Path(pdf_path).stem

    converter = DocumentConverter()
    doc = converter.convert(pdf_path).document
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    # Ensure output dir
    out_dir = Path("data"); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-chunk extraction
    extracted = []
    enriched_texts = []
    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"\n=== Chunk {i} raw (first 300 chars) ===\n{chunk.text[:300]}..." + Fore.RESET)
        enriched_text = chunker.contextualize(chunk=chunk)
        enriched_texts.append(enriched_text)
        # Try to record page info if available
        pages = getattr(chunk, "page_range", "") or getattr(chunk, "pages", "") or f"chunk-{i}"
        prompt = chunk_extraction_prompt(enriched_text, case_id=case_id, pages=str(pages))
        print(Fore.LIGHTMAGENTA_EX + f"\n--- Prompting extraction for chunk {i} ---" + Fore.RESET)
        rec = llm_json(prompt)
        extracted.append(rec)

    # Save chunk-level JSONL (useful for audit and training)
    with (out_dir / "extracted.jsonl").open("w", encoding="utf-8") as f:
        for rec in extracted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 2) Whole-case synthesis
    print(Fore.CYAN + "\n=== Synthesizing whole case ===" + Fore.RESET)
    synth_prompt = case_synthesis_prompt(json.dumps(extracted, ensure_ascii=False), case_id=case_id)
    structured_case = llm_json(synth_prompt, max_tokens=3000)
    (out_dir / "case_structured.json").write_text(json.dumps(structured_case, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Optional Q&A from structured case (for auxiliary training data)
    print(Fore.CYAN + "\n=== Generating Q&A from structured case ===" + Fore.RESET)
    qa_prompt = qa_from_case_prompt(json.dumps(structured_case, ensure_ascii=False), n=20)
    qa_pairs = llm_json(qa_prompt)
    (out_dir / "qa.json").write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) Build instruction-style records for fine-tuning
    print(Fore.CYAN + "\n=== Writing training JSONL ===" + Fore.RESET)
    train_path = out_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        # chunk tasks
        for text, rec in zip(enriched_texts, extracted):
            f.write(json.dumps({
                "instruction": "Extract a structured legal summary from the case chunk using the given JSON schema.",
                "input": text,
                "output": json.dumps(rec, ensure_ascii=False)
            }, ensure_ascii=False) + "\n")
        # synthesis task
        f.write(json.dumps({
            "instruction": "Merge multiple chunk-level extractions into a single structured case summary as per the schema.",
            "input": json.dumps(extracted, ensure_ascii=False),
            "output": json.dumps(structured_case, ensure_ascii=False)
        }, ensure_ascii=False) + "\n")
        # Q&A task (optional)
        f.write(json.dumps({
            "instruction": "Answer short questions about the case strictly from the structured JSON.",
            "input": json.dumps(structured_case, ensure_ascii=False),
            "output": json.dumps(qa_pairs, ensure_ascii=False)
        }, ensure_ascii=False) + "\n")

    print(Fore.GREEN + f"\nDone. Files written to {out_dir}/: extracted.jsonl, case_structured.json, qa.json, train.jsonl" + Fore.RESET)