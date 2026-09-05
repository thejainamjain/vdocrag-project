"""
Gradio Blocks app -- thin UI wiring over vdocrag_app.app_state.VDocRAGApp. Every
callback here does argument marshalling (Gradio types <-> plain Python) and
error-to-UI-message translation only; the actual logic is in app_state.py and
already has direct test coverage that doesn't depend on Gradio at all.

Run from a Colab cell after ModelManager.setup() has succeeded (Step 1) --
see notebooks/00_smoke_test.ipynb and docs/colab_workflow.md.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from vdocrag_app.app_state import PageCapExceeded, VDocRAGApp
from vdocrag_app.generator import VDocGeneratorWrapper
from vdocrag_app.ingest import PDFIngestError
from vdocrag_app.model_manager import ModelManager, ModelManagerConfig
from vdocrag_app.retriever import VDocRetrieverWrapper
from vdocrag_app.telemetry import logger, setup_logging


def on_index_pdf(vdocrag_app: VDocRAGApp, pdf_file, progress=None):
    """Module-level, not a closure -- takes vdocrag_app explicitly so this
    can be unit-tested with a mock app, independent of Gradio entirely.
    `progress` defaults to None rather than gr.Progress() so tests don't need
    a live Gradio context; build_app() binds the real gr.Progress() at
    UI-wiring time."""
    if pdf_file is None:
        return "Upload a PDF first.", gr.update()

    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file

    try:
        cap_info = vdocrag_app.check_page_cap(pdf_path)
    except PDFIngestError as e:
        return f"⚠️ Couldn't read this PDF: {e}", gr.update()

    if cap_info["state"] == "blocked":
        return (
            f"⚠️ This PDF has {cap_info['num_pages']} pages, estimated "
            f"~{cap_info['estimated_s'] / 60:.1f} minutes to index -- above the "
            f"soft cap. Use the 'Index anyway' button below if you want to proceed."
        ), gr.update(visible=True)

    if progress:
        progress(0, desc=f"Indexing {cap_info['num_pages']} pages...")
    try:
        result = vdocrag_app.index_pdf(pdf_path)
    except PDFIngestError as e:
        return f"⚠️ Failed to process this PDF: {e}", gr.update()
    except Exception as e:
        logger.info(f"Unexpected error in on_index_pdf: {e!r}")
        return f"⚠️ Unexpected error while indexing: {e}", gr.update()

    if progress:
        progress(1.0, desc="Done")
    msg = f"✅ Indexed '{result.source_pdf}' -- {result.num_pages} pages."
    if result.warnings:
        msg += " " + " ".join(result.warnings)
    return msg, gr.update(visible=False)


def on_index_pdf_forced(vdocrag_app: VDocRAGApp, pdf_file, progress=None):
    if pdf_file is None:
        return "Upload a PDF first.", gr.update()
    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
    if progress:
        progress(0, desc="Indexing (forced past soft cap)...")
    try:
        result = vdocrag_app.index_pdf(pdf_path, force=True)
    except PDFIngestError as e:
        return f"⚠️ Failed to process this PDF: {e}", gr.update(visible=False)
    if progress:
        progress(1.0, desc="Done")
    msg = f"✅ Indexed '{result.source_pdf}' -- {result.num_pages} pages. " + " ".join(result.warnings)
    return msg, gr.update(visible=False)


def on_ask(vdocrag_app: VDocRAGApp, question: str):
    if not question or not question.strip():
        return "Type a question first.", []

    try:
        result = vdocrag_app.ask(question)
    except ValueError as e:
        return f"⚠️ {e}", []
    except Exception as e:
        logger.info(f"Unexpected error in on_ask: {e!r}")
        return f"⚠️ Unexpected error while answering: {e}", []

    gallery_items = []
    for meta, score in result.retrieved_pages:
        caption = f"{meta.source_pdf} — page {meta.page_number} (similarity {score:.3f})"
        gallery_items.append((meta.image_path, caption))

    return result.answer, gallery_items


def build_app(vdocrag_app: VDocRAGApp) -> gr.Blocks:
    """Takes an already-constructed VDocRAGApp (models loaded, index ready)
    and wires up the Gradio Blocks UI around it. Every callback here is a
    thin lambda binding vdocrag_app + a live gr.Progress() into the
    module-level handler functions above, which carry the actual logic and
    already have direct test coverage."""

    with gr.Blocks(title="VDocRAG demo") as demo:
        gr.Markdown(
            "# VDocRAG demo\n"
            "Retrieval-augmented generation over document *images* -- no OCR step. "
            "Reproduction of [VDocRAG (Tanaka et al., CVPR 2025)](https://arxiv.org/abs/2504.09795) "
            "on [NTT's released checkpoints](https://github.com/nttmdlab-nlp/VDocRAG)."
        )

        with gr.Tab("Upload & Index"):
            pdf_input = gr.File(label="Upload a PDF", file_types=[".pdf"])
            index_status = gr.Markdown()
            index_btn = gr.Button("Index this PDF", variant="primary")
            force_btn = gr.Button("Index anyway (large PDF)", visible=False, variant="stop")

            index_btn.click(
                lambda f, p=gr.Progress(): on_index_pdf(vdocrag_app, f, p),
                inputs=[pdf_input], outputs=[index_status, force_btn],
            )
            force_btn.click(
                lambda f, p=gr.Progress(): on_index_pdf_forced(vdocrag_app, f, p),
                inputs=[pdf_input], outputs=[index_status, force_btn],
            )

        with gr.Tab("Ask"):
            question_input = gr.Textbox(label="Question", placeholder="What does this document say about...?")
            ask_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", interactive=False)
            retrieved_gallery = gr.Gallery(label="Retrieved pages (used to generate the answer above)", columns=3)

            ask_btn.click(
                lambda q: on_ask(vdocrag_app, q), inputs=[question_input], outputs=[answer_output, retrieved_gallery]
            )
            question_input.submit(
                lambda q: on_ask(vdocrag_app, q), inputs=[question_input], outputs=[answer_output, retrieved_gallery]
            )

    return demo


def main():
    """Entry point for a Colab cell: `from app import main; demo = main()`
    then `demo.launch(share=True)` in the next cell (kept separate so the
    launch call, which blocks, is its own cell and can be re-run without
    reloading the models)."""
    setup_logging("app_session")

    manager = ModelManager(ModelManagerConfig())
    manager.setup()
    logger.info(f"Models loaded in '{manager.mode}' mode. VRAM: {manager.vram_report()}")

    retriever = VDocRetrieverWrapper(manager)
    generator = VDocGeneratorWrapper(manager)
    vdocrag_app = VDocRAGApp(retriever=retriever, generator=generator)

    return build_app(vdocrag_app)


if __name__ == "__main__":
    demo = main()
    demo.launch(share=True)
