"""
FAISS index wrapper for retrieved page embeddings.

Design notes:
- Embeddings from VDocRetriever are L2-normalized (normalize=True, per NTT's
  own test.py / README quickstart) — so inner product on normalized vectors
  IS cosine similarity, which is exactly the paper's SIM(h_q, h_d) formula
  (Section 4.1 of the paper: h_q^T h_d / (||h_q|| ||h_d||)). We use
  `IndexFlatIP` deliberately, not `IndexFlatL2`, to match this without a
  redundant extra normalization step at search time.
- `faiss-cpu`, not `faiss-gpu` — no faiss-gpu wheel matches Colab's current
  CUDA version, and at demo scale (hundreds to low-thousands of page vectors)
  exact CPU search is effectively instant; this was already covered in the
  handoff doc's Section 3 reasoning and doesn't need revisiting.
- `IndexIDMap2` wraps the flat index so we can use our own stable integer ids
  (rather than relying on FAISS's implicit insertion-order ids), which matters
  once pages can be deleted/re-indexed (re-uploading a corrected PDF, etc.).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from vdocrag.telemetry import log_call, logger


@dataclass
class PageMetadata:
    """Everything needed to show a retrieved page back to the user and to
    trace it back to its source PDF."""

    doc_id: int
    source_pdf: str
    page_number: int  # 1-indexed, for display
    image_path: str  # where the rasterized page image is cached on disk


class DocumentIndex:
    """Wraps a FAISS IndexFlatIP + a parallel metadata store, with save/load
    for Drive persistence (Colab session survives a killed runtime as long as
    the index + page image cache are on Drive — see docs/colab_workflow.md).
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))
        self._metadata: dict[int, PageMetadata] = {}
        self._next_id = 0

    def __len__(self) -> int:
        return self._index.ntotal

    @log_call("index")
    def add(self, embedding: np.ndarray, source_pdf: str, page_number: int, image_path: str) -> int:
        """Adds one page embedding + its metadata. Returns the assigned doc_id."""
        embedding = self._validate_embedding(embedding)
        doc_id = self._next_id
        self._next_id += 1

        self._index.add_with_ids(embedding.reshape(1, -1), np.array([doc_id], dtype=np.int64))
        self._metadata[doc_id] = PageMetadata(
            doc_id=doc_id, source_pdf=source_pdf, page_number=page_number, image_path=image_path
        )
        return doc_id

    @log_call("index")
    def add_batch(self, embeddings: np.ndarray, source_pdf: str, image_paths: List[str]) -> List[int]:
        """Adds all pages of one PDF in a single FAISS call — cheaper than
        calling `add()` in a loop for large documents."""
        n = embeddings.shape[0]
        if len(image_paths) != n:
            raise ValueError(f"embeddings has {n} rows but got {len(image_paths)} image_paths")

        embeddings = np.stack([self._validate_embedding(e) for e in embeddings])
        doc_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n

        self._index.add_with_ids(embeddings, doc_ids)
        for i, doc_id in enumerate(doc_ids):
            self._metadata[int(doc_id)] = PageMetadata(
                doc_id=int(doc_id), source_pdf=source_pdf, page_number=i + 1, image_path=image_paths[i]
            )
        return doc_ids.tolist()

    @log_call("index")
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[tuple[PageMetadata, float]]:
        """Returns up to top_k (metadata, similarity_score) pairs, sorted by
        descending similarity. top_k=3 matches the paper's own finding
        (supplementary Figure C) that 3 retrieved documents is optimal for
        VDocGenerator — fewer misses context, more introduces retrieval noise."""
        if len(self) == 0:
            return []

        query_embedding = self._validate_embedding(query_embedding)
        k = min(top_k, len(self))
        scores, ids = self._index.search(query_embedding.reshape(1, -1), k)

        results = []
        for score, doc_id in zip(scores[0], ids[0]):
            if doc_id == -1:  # FAISS pads with -1 if k > ntotal in some edge cases
                continue
            results.append((self._metadata[int(doc_id)], float(score)))
        return results

    def remove_by_source(self, source_pdf: str) -> int:
        """Removes every page belonging to a given source PDF (e.g. before
        re-indexing a corrected upload). Returns the number of pages removed."""
        ids_to_remove = [doc_id for doc_id, meta in self._metadata.items() if meta.source_pdf == source_pdf]
        if not ids_to_remove:
            return 0
        self._index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
        for doc_id in ids_to_remove:
            del self._metadata[doc_id]
        logger.info(f"Removed {len(ids_to_remove)} pages for source '{source_pdf}'")
        return len(ids_to_remove)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        meta_serializable = {str(k): asdict(v) for k, v in self._metadata.items()}
        with open(path / "metadata.json", "w") as f:
            json.dump({"embedding_dim": self.embedding_dim, "next_id": self._next_id, "metadata": meta_serializable}, f)
        logger.info(f"Index saved to {path} ({len(self)} pages)")

    @classmethod
    def load(cls, path: str | Path) -> "DocumentIndex":
        path = Path(path)
        with open(path / "metadata.json") as f:
            data = json.load(f)
        instance = cls(embedding_dim=data["embedding_dim"])
        instance._index = faiss.read_index(str(path / "index.faiss"))
        instance._metadata = {int(k): PageMetadata(**v) for k, v in data["metadata"].items()}
        instance._next_id = data["next_id"]
        logger.info(f"Index loaded from {path} ({len(instance)} pages)")
        return instance

    @staticmethod
    def _validate_embedding(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 1:
            embedding = embedding.reshape(-1)
        return embedding
