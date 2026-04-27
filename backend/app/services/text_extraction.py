"""Text extraction from PDF, DOCX, and plain text files."""

import io
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from a PDF file using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_content))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
        else:
            logger.warning(f"Page {i + 1}: no extractable text (possibly scanned/image)")

    full_text = "\n\n".join(pages)
    if not full_text.strip():
        raise ValueError("No text could be extracted from PDF. It may be scanned/image-based.")
    return full_text


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(io.BytesIO(file_content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    full_text = "\n\n".join(paragraphs)
    if not full_text.strip():
        raise ValueError("No text could be extracted from DOCX.")
    return full_text


def extract_text(file_content: bytes, file_type: str) -> str:
    """Extract text from a file based on its type.

    Args:
        file_content: Raw file bytes
        file_type: One of 'pdf', 'docx', 'txt'

    Returns:
        Extracted text as a string
    """
    if file_type == "pdf":
        return extract_text_from_pdf(file_content)
    elif file_type == "docx":
        return extract_text_from_docx(file_content)
    elif file_type == "txt":
        return file_content.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
