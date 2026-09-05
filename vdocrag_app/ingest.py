"""
PDF -> per-page PIL images. No model involved (Step 2 of the build plan) —
testable without a GPU, which is why this module and its test run happily
outside Colab.

Requires poppler-utils on the system (apt-get install poppler-utils in the
Colab install cell; already present in most Linux dev environments).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

from vdocrag_app.telemetry import log_call, logger

# Paper's own quickstart resizes to 1344x1344 before handing off to the
# processor (Section 4.2 of the handoff doc); we rasterize at a DPI that
# comfortably covers that without wasting time on pages that get downscaled
# anyway. 200 DPI on a standard US Letter / A4 page lands close to that
# target long-edge size for typical documents.
DEFAULT_DPI = 200

# Soft cap on any single page's long edge before we downscale ourselves,
# to avoid handing pathologically large rasterizations (e.g. a PDF built
# from huge embedded images) to the retriever's own dynamic-resolution
# preprocessing.
MAX_LONG_EDGE = 1600


class PDFIngestError(Exception):
    """Raised when a PDF can't be rasterized (corrupt file, missing poppler,
    encrypted PDF without a password, etc.) — callers should catch this and
    surface a clean message in the Gradio UI rather than a raw traceback."""


def _downscale_if_needed(img: Image.Image, max_long_edge: int = MAX_LONG_EDGE) -> Image.Image:
    long_edge = max(img.size)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / long_edge
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)


@log_call("ingest")
def pdf_to_images(pdf_path: str | Path, dpi: int = DEFAULT_DPI) -> List[Image.Image]:
    """Rasterize every page of a PDF to a PIL Image, in page order.

    Raises PDFIngestError on failure rather than letting pdf2image's
    lower-level exceptions (which mention poppler internals) bubble up
    straight to the Gradio UI.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PDFIngestError(f"PDF not found: {pdf_path}")

    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:  # pdf2image raises poppler-specific exceptions
        raise PDFIngestError(f"Failed to rasterize {pdf_path.name}: {e}") from e

    if not pages:
        raise PDFIngestError(f"{pdf_path.name} has no pages or could not be read")

    pages = [_downscale_if_needed(p.convert("RGB")) for p in pages]
    logger.info(
        f"Rasterized {pdf_path.name}: {len(pages)} pages",
        extra={"extra_fields": {
            "component": "ingest",
            "function": "pdf_to_images",
            "num_pages": len(pages),
            "dpi": dpi,
        }},
    )
    return pages


def page_count(pdf_path: str | Path) -> int:
    """Cheap page count without full rasterization — use this to evaluate
    the page-cap policy (limits.py) *before* committing to indexing."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PDFIngestError(f"PDF not found: {pdf_path}")
    try:
        info = pdfinfo_from_path(str(pdf_path))
        return int(info["Pages"])
    except Exception as e:
        raise PDFIngestError(f"Failed to read page count for {pdf_path.name}: {e}") from e
