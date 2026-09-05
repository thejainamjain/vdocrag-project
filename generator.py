"""
Thin wrapper around NTT's VDocGenerator. Prompt construction and generation
args below are copied verbatim from their test.py / README (Generation
section) -- same rationale as retriever.py: the fine-tuned checkpoint expects
this exact chat-template + image-token format, not an approximately similar
one.
"""
from __future__ import annotations

from typing import List

from PIL import Image

from vdocrag_app.retriever import prepare_doc_image
from vdocrag_app.telemetry import log_call, logger

# Matches NTT's test.py exactly: "\n Answer briefly." is their instruction
# suffix for getting short, direct answers rather than verbose completions --
# also documented as the standard instruction template in the paper's own
# supplementary material (Section B, "Instruction templates").
ANSWER_SUFFIX = "\n Answer briefly."

DEFAULT_GENERATION_ARGS = {
    "max_new_tokens": 64,
    "temperature": 0.0,
    "do_sample": False,
}


def build_image_tokens(num_images: int) -> str:
    """Pure function: "<|image_1|>\\n<|image_2|>\\n..." for however many
    retrieved pages are being passed in. Order matters -- must match the
    order `images` are given to the processor."""
    return "\n".join(f"<|image_{i + 1}|>" for i in range(num_images))


def build_chat_prompt(processor, question: str, num_images: int) -> str:
    """Pure-ish function (only needs the processor's tokenizer, not a live
    model) -- kept separate from answer() for unit testing the templating
    logic against a real processor without running the model itself."""
    image_tokens = build_image_tokens(num_images)
    query = f"{question.strip()}{ANSWER_SUFFIX}"
    messages = [{"role": "user", "content": f"{image_tokens}\n{query}"}]
    return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class VDocGeneratorWrapper:
    """Same design as VDocRetrieverWrapper: holds no model state itself,
    everything comes from the shared ModelManager (Section 4.4)."""

    def __init__(self, model_manager):  # type: (ModelManager) -> None
        self._mm = model_manager

    @log_call("generator")
    def answer(self, question: str, images: List[Image.Image], generation_args: dict | None = None) -> str:
        """Generates an answer given a question and its retrieved page
        images (already in relevance order from DocumentIndex.search()).
        Returns the decoded answer string, stripped."""
        model = self._mm.use_generator()
        processor = self._mm.processor

        if not images:
            raise ValueError("answer() requires at least one retrieved image -- got an empty list")

        prepared_images = [prepare_doc_image(img) for img in images]
        prompt = build_chat_prompt(processor, question, len(prepared_images))

        processed = processor(prompt, images=prepared_images, return_tensors="pt").to("cuda:0")

        args = {**DEFAULT_GENERATION_ARGS, "eos_token_id": processor.tokenizer.eos_token_id}
        if generation_args:
            args.update(generation_args)

        generate_ids = model.generate(processed, generation_args=args)
        # strip the input prompt tokens back off -- matches NTT's own
        # test.py slicing exactly (generate() returns prompt+completion together)
        generate_ids = generate_ids[:, processed["input_ids"].shape[1]:]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.strip()

        logger.info(
            f"Generated answer ({len(prepared_images)} images, {len(response)} chars)",
            extra={"extra_fields": {
                "component": "generator", "function": "answer",
                "num_images": len(prepared_images), "answer_len": len(response),
            }},
        )
        return response
