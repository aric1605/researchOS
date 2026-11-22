"""Section-aware text chunking with token/character budgeting and overlap."""

import re
from typing import Any, Dict, List

# Section header patterns in academic papers
SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[0-9IVXLCDM]+\.?\s+)?(Abstract|Introduction|Related Work|Background|"
    r"Methodology|Methods?|Model Architecture|Architecture|Model|Approach|Experiments?|Results?|"
    r"Evaluation|Discussion|Limitations?|Conclusions?|Broader Impact)[\s\:\.\n]",
    re.IGNORECASE,
)


def _split_into_sections(text: str) -> List[Dict[str, str]]:
    """Splits paper text into identifiable sections."""
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [{"section": "Main", "text": text.strip()}]

    sections = []
    # Text before the first matched section header
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append({"section": "Abstract", "text": preamble})

    for i, match in enumerate(matches):
        sec_name = match.group(1).strip().title()
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_body = text[start_idx:end_idx].strip()
        if sec_body:
            sections.append({"section": sec_name, "text": sec_body})

    return sections


def chunk_paper(
    paper: Dict[str, Any],
    pdf_text: str = "",
    target_chunk_size: int = 1500,  # ~375-400 tokens
    overlap_size: int = 150,        # ~40-50 tokens
) -> List[Dict[str, Any]]:
    """Chunks a paper into section-aware overlapping chunks with rich metadata.
    If pdf_text is empty or too short, chunks the abstract/summary.
    """
    paper_id = paper.get("paper_id", "")
    title = paper.get("title", "")
    authors = paper.get("authors", [])
    year = paper.get("year", 2024)
    arxiv_url = paper.get("arxiv_url", "")
    summary = paper.get("summary", "")

    raw_text = pdf_text if len(pdf_text.strip()) > 300 else summary
    sections = _split_into_sections(raw_text)

    # Always ensure abstract is represented as chunk 0 if missing from sections
    if not any(s["section"].lower() == "abstract" for s in sections) and summary:
        sections.insert(0, {"section": "Abstract", "text": summary})

    chunks: List[Dict[str, Any]] = []
    chunk_global_idx = 0

    for section_obj in sections:
        sec_name = section_obj["section"]
        body = section_obj["text"]

        # Split body into overlapping slices
        body_len = len(body)
        if body_len <= target_chunk_size:
            if body.strip():
                chunks.append({
                    "chunk_id": f"{paper_id}_c{chunk_global_idx}",
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "arxiv_url": arxiv_url,
                    "section": sec_name,
                    "chunk_index": chunk_global_idx,
                    "text": body.strip(),
                })
                chunk_global_idx += 1
        else:
            start = 0
            while start < body_len:
                end = min(start + target_chunk_size, body_len)
                # Try to break at paragraph or sentence boundary
                if end < body_len:
                    # Look backwards for newline or period
                    punct_pos = max(
                        body.rfind("\n\n", start, end),
                        body.rfind(". ", start, end),
                    )
                    if punct_pos > start + (target_chunk_size // 2):
                        end = punct_pos + 1

                chunk_str = body[start:end].strip()
                if chunk_str:
                    chunks.append({
                        "chunk_id": f"{paper_id}_c{chunk_global_idx}",
                        "paper_id": paper_id,
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "arxiv_url": arxiv_url,
                        "section": sec_name,
                        "chunk_index": chunk_global_idx,
                        "text": chunk_str,
                    })
                    chunk_global_idx += 1

                if end >= body_len:
                    break
                start = max(start + 1, end - overlap_size)

    return chunks
