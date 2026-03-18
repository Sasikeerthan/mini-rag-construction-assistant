"""Markdown-aware document chunker for the RAG pipeline."""

import os
import re
from pathlib import Path


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 words per token, so tokens ~ words / 0.75."""
    return int(len(text.split()) / 0.75)


def _split_by_paragraphs(text: str, max_tokens: int = 400) -> list[str]:
    """Split text into paragraph-based sub-chunks that stay under max_tokens."""
    paragraphs = re.split(r"\n\n+", text.strip())
    chunks = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current and current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _parse_markdown_sections(text: str) -> list[dict]:
    """Parse markdown into sections keyed by header hierarchy.

    Returns a list of {"header": "H2 > H3", "content": "..."} dicts.
    """
    lines = text.split("\n")
    sections = []
    current_h2 = None
    current_h3 = None
    current_lines: list[str] = []

    def flush():
        if current_lines:
            header_parts = []
            if current_h2:
                header_parts.append(current_h2)
            if current_h3:
                header_parts.append(current_h3)
            header = " > ".join(header_parts) if header_parts else "Introduction"
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({"header": header, "content": content})

    for line in lines:
        h2_match = re.match(r"^##\s+(.+)$", line)
        h3_match = re.match(r"^###\s+(.+)$", line)

        if h2_match:
            flush()
            current_h2 = h2_match.group(1).strip()
            current_h3 = None
            current_lines = [line]
        elif h3_match:
            flush()
            current_h3 = h3_match.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()
    return sections


def chunk_documents(doc_dir: str) -> list[dict]:
    """Chunk all markdown files in doc_dir into RAG-ready pieces.

    Args:
        doc_dir: Path to directory containing .md files.

    Returns:
        List of dicts with keys: "doc", "header", "content".
    """
    doc_path = Path(doc_dir)
    md_files = sorted(doc_path.glob("*.md"))

    all_chunks = []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        filename = md_file.name
        sections = _parse_markdown_sections(text)

        for section in sections:
            token_count = _estimate_tokens(section["content"])

            if token_count <= 400:
                all_chunks.append({
                    "doc": filename,
                    "header": section["header"],
                    "content": section["content"],
                })
            else:
                # Split long sections by paragraphs
                sub_chunks = _split_by_paragraphs(section["content"], max_tokens=400)
                for i, sub in enumerate(sub_chunks):
                    header = section["header"]
                    if len(sub_chunks) > 1:
                        header = f"{header} (part {i + 1})"
                    all_chunks.append({
                        "doc": filename,
                        "header": header,
                        "content": sub,
                    })

    return all_chunks
