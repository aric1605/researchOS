"""Synthesis agent: generates a comprehensive cited research summary in Markdown
with strict citation enforcement and citation coverage evaluation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from app.clients.llm import LLMClient

logger = logging.getLogger("researchos.synthesis")

CITATION_REGEX = re.compile(r"\[(\d+)\]")


class SynthesisReport(BaseModel):
    title: str = Field(description="Scholarly title for the synthesis report")
    executive_summary: str = Field(description="High-level executive summary with [n] citations")
    key_findings: List[str] = Field(description="Bulleted key findings, each ending with [n] citations")
    critical_comparison: str = Field(description="Detailed comparative analysis with [n] citations")
    limitations_and_contradictions: str = Field(
        description="Discussion of methodological constraints, limitations, and contradictions with [n] citations"
    )
    open_questions: List[str] = Field(description="List of unresolved research questions")
    full_markdown: Optional[str] = Field(
        default=None,
        description="Complete formatted markdown synthesis report",
    )


def extract_sentences(text: str) -> List[str]:
    """Extracts non-heading, non-empty candidate sentences from Markdown text body.
    Stops before the ## References section and ignores list item citations.
    """
    lines = text.split("\n")
    sentences = []
    in_references = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect start of References section
        if re.match(r"^#{1,3}\s*references", line, re.IGNORECASE):
            in_references = True
            continue

        if in_references:
            continue

        # Skip headers, horizontal rules, and bibliography entries
        if line.startswith("#") or line.startswith("---") or re.match(r"^[-*]?\s*\[\d+\]", line):
            continue

        # Split on sentence terminals: ., !, ?
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p_clean = p.strip()
            # Informative factual sentence: at least 4 words and doesn't start with Markdown headers
            if len(p_clean.split()) >= 4 and not p_clean.startswith("#"):
                sentences.append(p_clean)

    return sentences


def check_and_enforce_citations(
    markdown_text: str,
    valid_citation_indices: Set[int],
) -> Dict[str, Any]:
    """Analyzes citation coverage and flags uncited factual statements."""
    sentences = extract_sentences(markdown_text)
    if not sentences:
        return {
            "citation_coverage": 1.0,
            "total_sentences": 0,
            "cited_sentences": 0,
            "uncited_sentences": [],
            "invalid_citations": [],
            "processed_markdown": markdown_text,
        }

    cited_count = 0
    uncited_sentences = []
    invalid_citations = []

    for s in sentences:
        found_cites = [int(m) for m in CITATION_REGEX.findall(s)]
        if not found_cites:
            uncited_sentences.append(s)
        else:
            # Check if all citations in this sentence are valid reference indices
            valid_cites = [c for c in found_cites if c in valid_citation_indices]
            invalid_in_s = [c for c in found_cites if c not in valid_citation_indices]
            if invalid_in_s:
                invalid_citations.extend(invalid_in_s)

            if valid_cites:
                cited_count += 1
            else:
                uncited_sentences.append(s)

    coverage = round(cited_count / len(sentences), 4) if sentences else 1.0

    return {
        "citation_coverage": coverage,
        "total_sentences": len(sentences),
        "cited_sentences": cited_count,
        "uncited_sentences": uncited_sentences,
        "invalid_citations": list(set(invalid_citations)),
        "processed_markdown": markdown_text,
    }


class SynthesisAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    async def execute(
        self,
        question: str,
        papers: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        comparison: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesizes research findings into cited Markdown report with reference index mapping."""
        if not papers:
            return {
                "title": f"Research Synthesis: {question}",
                "markdown": "# Research Synthesis\n\nNo papers were found matching the research question.",
                "references": [],
                "citation_coverage": 0.0,
                "uncited_sentences": [],
            }

        # Build stable numbered reference list
        # [1] Authors (Year). Title. URL
        reference_list = []
        paper_to_ref_idx: Dict[str, int] = {}
        for idx, p in enumerate(papers, start=1):
            pid = p.get("paper_id", "")
            paper_to_ref_idx[pid] = idx
            authors = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            ref_entry = {
                "index": idx,
                "paper_id": pid,
                "title": p.get("title", ""),
                "authors": authors or "Unknown Authors",
                "year": p.get("year", 2024),
                "arxiv_url": p.get("arxiv_url", f"https://arxiv.org/abs/{pid}"),
                "citation_text": f"[{idx}] {authors or 'Unknown'} ({p.get('year', 2024)}). {p.get('title')}. {p.get('arxiv_url')}",
            }
            reference_list.append(ref_entry)

        valid_indices = {r["index"] for r in reference_list}

        # Format verified claims with citation numbers
        formatted_claims = []
        for c in claims:
            pid = c.get("paper_id", "")
            ref_idx = paper_to_ref_idx.get(pid, 1)
            formatted_claims.append(
                f"- Claim [{c.get('claim_id')}] (Ref [{ref_idx}]): {c.get('claim')}\n"
                f"  Evidence: \"{c.get('quote')}\""
            )

        claims_context = "\n".join(formatted_claims)
        ref_context = "\n".join([r["citation_text"] for r in reference_list])

        prompt = (
            f"You are a principal research scientist writing an authoritative, evidence-grounded research synthesis.\n\n"
            f"RESEARCH QUESTION:\n\"{question}\"\n\n"
            f"AVAILABLE REFERENCES:\n{ref_context}\n\n"
            f"VERIFIED EVIDENCE & CLAIMS:\n{claims_context}\n\n"
            f"COMPARATIVE THEMES & LIMITATIONS:\n{comparison}\n\n"
            f"CRITICAL CITATION RULES:\n"
            f"1. Every factual sentence MUST end with one or more citation markers like [1] or [2].\n"
            f"2. You may ONLY cite numbers corresponding to the available references (1 to {len(reference_list)}).\n"
            f"3. Base all factual statements strictly on the verified evidence provided. Do not extrapolate or hallucinate.\n"
            f"4. Provide an Executive Summary, Key Findings, Critical Comparison, Limitations & Contradictions, and Open Questions."
        )

        synthesis = await self.llm.generate_structured(
            prompt=prompt,
            schema=SynthesisReport,
            system_instruction="You are an uncompromisingly rigorous scientific writer.",
        )

        # Assemble formatted markdown
        findings_md = "\n".join([f"- {kf}" for kf in synthesis.key_findings])
        open_q_md = "\n".join([f"- {oq}" for oq in synthesis.open_questions])
        references_md = "\n".join([f"- {r['citation_text']}" for r in reference_list])

        final_md = (
            f"# {synthesis.title}\n\n"
            f"## Executive Summary\n{synthesis.executive_summary}\n\n"
            f"## Key Findings\n{findings_md}\n\n"
            f"## Critical Comparison & Analysis\n{synthesis.critical_comparison}\n\n"
            f"## Limitations & Contradictions\n{synthesis.limitations_and_contradictions}\n\n"
            f"## Open Research Questions\n{open_q_md}\n\n"
            f"## References\n{references_md}\n"
        )

        # Post-check citation coverage
        citation_check = check_and_enforce_citations(final_md, valid_indices)

        return {
            "title": synthesis.title,
            "markdown": final_md,
            "reference_list": reference_list,
            "citation_coverage": citation_check["citation_coverage"],
            "total_sentences": citation_check["total_sentences"],
            "cited_sentences": citation_check["cited_sentences"],
            "uncited_sentences": citation_check["uncited_sentences"],
            "invalid_citations": citation_check["invalid_citations"],
        }
