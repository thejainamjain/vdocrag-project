"""
Thin wrapper around NTT's VDocRetriever, exposing encode_query/encode_document
as plain numpy-in-numpy-out functions for the FAISS index (index.py) to consume.

Prompt templates below are copied verbatim from NTT's own test.py and README
quickstart (github.com/nttmdlab-nlp/VDocRAG) -- NOT reconstructed from the
paper's prose description. The paper doesn't spell out the literal instruction
string; getting this exactly right matters, since the retriever was fine-tuned
against these specific strings and an approximately-similar-but-different
instruction wording would silently degrade retrieval quality without ever
raising an error.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from vdocrag_app.telemetry import log_call, logger

# torch and ModelManager are imported lazily inside methods that need them
# (see note in encode_query) -- this keeps build_query_prompt/prepare_doc_image
# importable and unit-testable without torch installed, which matters for
# testing this module's pure logic outside a GPU environment.

# Verbatim from NTT's test.py / README (Retrieval section). The trailing
# "</s>" is part of their fine-tuning format, not a typo to "clean up".
QUERY_PROMPT_TEMPLATE = "Instruct: I\u2019m looking for an image that answers the question.\nQuery: {query}</s>"
DOC_PROMPT = "<|image_1|>\nWhat is shown in this image?</s>"

# Their quickstart resizes every document image to exactly this size before
# processing -- the model was fine-tuned against this input size, so this
# isn't a free parameter to tune down for speed without expecting some
# retrieval-quality cost.
DOC_IMAGE_SIZE = (1344, 1344)

QUERY_MAX_LENGTH = 256  # matches their query_inputs processor call
DOC_MAX_LENGTH = 4096  # matches their doc_inputs processor call


def build_query_prompt(query: str) -> str:
    """Pure function, no model/GPU involved -- kept separate from
    encode_query() specifically so it's unit-testable without a GPU."""
    return QUERY_PROMPT_TEMPLATE.format(query=query)


def prepare_doc_image(image: Image.Image) -> Image.Image:
    """Pure function: resize to the exact size NTT's checkpoint expects.
    RGB conversion guards against paletted/CMYK PDFs producing non-RGB
    rasterizations upstream in ingest.py."""
    return image.convert("RGB").resize(DOC_IMAGE_SIZE)


class VDocRetrieverWrapper:
    """Owns no model state itself -- everything comes from the shared
    ModelManager, so retrieval and generation can share one base model
    (Section 4.4 of the handoff doc) without this class needing to know
    which mode ("shared" vs "independent") is active."""

    def __init__(self, model_manager):  # type: (ModelManager) -> None
        self._mm = model_manager

    @log_call("retriever")
    def encode_query(self, query_text: str) -> np.ndarray:
        """Returns a single L2-normalized embedding vector (float32 numpy),
        ready to hand to DocumentIndex.search()."""
        import torch

        model = self._mm.use_retriever()
        processor = self._mm.processor

        prompt = build_query_prompt(query_text)
        inputs = processor(
            [prompt], return_tensors="pt", padding="longest", max_length=QUERY_MAX_LENGTH, truncation=True
        ).to("cuda:0")

        with torch.no_grad():
            output = model(query=inputs, use_cache=False)

        embedding = output.q_reps[0].detach().cpu().float().numpy()
        return embedding

    @log_call("retriever")
    def encode_document(self, image: Image.Image) -> np.ndarray:
        """Returns a single L2-normalized embedding vector for one page image."""
        import torch

        model = self._mm.use_retriever()
        processor = self._mm.processor

        prepared = prepare_doc_image(image)
        inputs = processor(
            DOC_PROMPT, images=prepared, return_tensors="pt", padding="longest", max_length=DOC_MAX_LENGTH, truncation=True
        ).to("cuda:0")

        with torch.no_grad():
            output = model(document=inputs, use_cache=False)

        embedding = output.p_reps[0].detach().cpu().float().numpy()
        return embedding

    @log_call("retriever")
    def encode_documents_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Batched version of encode_document -- matches NTT's own
        `doc_inputs` stacking pattern in test.py, one processor call per
        image (their processor doesn't support batched multi-image calls
        directly) followed by a single stacked forward pass. Cuts per-page
        Python/dispatch overhead relative to calling encode_document() in a
        loop, though the forward pass itself still scales with batch size."""
        import torch

        model = self._mm.use_retriever()
        processor = self._mm.processor

        collated = [
            processor(
                DOC_PROMPT,
                images=prepare_doc_image(img),
                return_tensors="pt",
                padding="longest",
                max_length=DOC_MAX_LENGTH,
                truncation=True,
            ).to("cuda:0")
            for img in images
        ]
        doc_inputs = {
            key: torch.stack([item[key][0] for item in collated], dim=0)
            for key in ["input_ids", "attention_mask", "pixel_values", "image_sizes"]
        }

        with torch.no_grad():
            output = model(document=doc_inputs, use_cache=False)

        embeddings = output.p_reps.detach().cpu().float().numpy()
        logger.info(f"Batch-encoded {len(images)} document images")
        return embeddings
