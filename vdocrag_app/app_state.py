"""
Orchestration logic for the two main user actions (index a PDF, ask a
question), kept separate from app.py's Gradio wiring so it's testable with
mocked models/retriever/generator/index -- none of which require a GPU to
exercise the control flow, error handling, and logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

from vdocrag_app.index import DocumentIndex, PageMetadata
from vdocrag_app.ingest import PDFIngestError, page_count, pdf_to_images
from vdocrag_app.limits import estimated_wait_s, page_cap
from vdocrag_app.telemetry import log_call, logger

PAGE_CACHE_DIR = Path("/content/drive/MyDrive/vdocrag-project/page_cache")
INDEX_DIR = Path("/content/drive/MyDrive/vdocrag-project/faiss_index")
EMBEDDING_DIM = 3072  # Phi-3-vision hidden size; confirm against a real
# encode_query() call in Step 1 -- if the true dim differs, DocumentIndex's
# constructor will simply reject mismatched-shape embeddings at add() time
# rather than silently corrupting the index, so this is a safe default to
# start from, not a silent assumption.

TOP_K = 3  # matches the paper's own finding (supplementary Figure C)


@dataclass
class IndexingResult:
    source_pdf: str
    num_pages: int
    doc_ids: List[int]
    warnings: List[str] = field(default_factory=list)


@dataclass
class AskResult:
    answer: str
    retrieved_pages: List[tuple[PageMetadata, float]]  # (metadata, similarity_score)


class PageCapExceeded(Exception):
    """Raised when a PDF exceeds the hard-block threshold and the caller
    hasn't passed force=True (the UI's "index anyway" opt-in)."""

    def __init__(self, num_pages: int, warn_pages: int, estimated_s: float):
        self.num_pages = num_pages
        self.warn_pages = warn_pages
        self.estimated_s = estimated_s
        super().__init__(
            f"{num_pages} pages exceeds the {warn_pages}-page soft cap "
            f"(estimated ~{estimated_s:.0f}s to index). Pass force=True to proceed anyway."
        )


class VDocRAGApp:
    """Holds the live model/index state for one Colab session. One instance,
    constructed once at notebook startup, reused across every Gradio callback."""

    def __init__(self, retriever, generator, observed_s_per_page: Optional[float] = None):
        self.retriever = retriever
        self.generator = generator
        self.observed_s_per_page = observed_s_per_page  # None until Step 1 gives us a real number
        self.index = self._load_or_create_index()
        PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_or_create_index(self) -> DocumentIndex:
        if (INDEX_DIR / "metadata.json").exists():
            logger.info(f"Loading existing index from {INDEX_DIR}")
            return DocumentIndex.load(INDEX_DIR)
        logger.info("No existing index found -- starting fresh.")
        return DocumentIndex(embedding_dim=EMBEDDING_DIM)

    def check_page_cap(self, pdf_path: str) -> dict:
        """Call before indexing to decide what the UI should show: proceed
        silently, show a confirm-before-start banner, or block by default.
        Returns a dict the UI layer maps directly to its three states."""
        n = page_count(pdf_path)
        caps = page_cap(self.observed_s_per_page) if self.observed_s_per_page else page_cap()
        estimated = estimated_wait_s(n, caps["observed_s_per_page"])

        if n <= caps["no_warning_pages"]:
            state = "proceed"
        elif n <= caps["warn_pages"]:
            state = "confirm"
        else:
            state = "blocked"

        return {"num_pages": n, "estimated_s": estimated, "state": state, **caps}

    @log_call("app")
    def index_pdf(self, pdf_path: str, force: bool = False) -> IndexingResult:
        """Full pipeline: rasterize -> encode -> cache page images to disk ->
        add to FAISS index -> persist. Raises PDFIngestError (bad file) or
        PageCapExceeded (too many pages, no force=True) rather than silently
        truncating -- the caller (Gradio callback) decides how to surface
        that to the user."""
        cap_info = self.check_page_cap(pdf_path)
        if cap_info["state"] == "blocked" and not force:
            raise PageCapExceeded(cap_info["num_pages"], cap_info["warn_pages"], cap_info["estimated_s"])

        source_name = Path(pdf_path).name
        # re-indexing the same filename replaces rather than duplicates --
        # avoids silently accumulating stale duplicate pages on re-upload
        removed = self.index.remove_by_source(source_name)
        if removed:
            logger.info(f"Replacing {removed} previously-indexed pages for '{source_name}'")

        pages = pdf_to_images(pdf_path)

        image_paths = []
        doc_cache_dir = PAGE_CACHE_DIR / source_name
        doc_cache_dir.mkdir(parents=True, exist_ok=True)
        for i, page_img in enumerate(pages):
            img_path = doc_cache_dir / f"page_{i + 1:04d}.png"
            page_img.save(img_path)
            image_paths.append(str(img_path))

        embeddings = self.retriever.encode_documents_batch(pages)
        doc_ids = self.index.add_batch(embeddings, source_pdf=source_name, image_paths=image_paths)

        self.index.save(INDEX_DIR)

        warnings = []
        if cap_info["state"] == "confirm":
            warnings.append(f"Indexed {cap_info['num_pages']} pages (~{cap_info['estimated_s']:.0f}s).")
        if force and cap_info["state"] == "blocked":
            warnings.append(
                f"Indexed {cap_info['num_pages']} pages despite exceeding the soft cap, per your request."
            )

        return IndexingResult(source_pdf=source_name, num_pages=len(pages), doc_ids=doc_ids, warnings=warnings)

    @log_call("app")
    def ask(self, question: str, top_k: int = TOP_K) -> AskResult:
        """Full query pipeline: encode question -> retrieve top-k pages ->
        load their cached images -> generate an answer grounded in them."""
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")
        if len(self.index) == 0:
            raise ValueError("No documents indexed yet -- upload a PDF first.")

        query_embedding = self.retriever.encode_query(question)
        results = self.index.search(query_embedding, top_k=top_k)

        if not results:
            raise ValueError("Retrieval returned no results despite a non-empty index -- this shouldn't happen.")

        retrieved_images = [Image.open(meta.image_path) for meta, _score in results]
        answer = self.generator.answer(question, retrieved_images)

        return AskResult(answer=answer, retrieved_pages=results)
