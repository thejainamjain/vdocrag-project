# VDocRAG demo

A working reproduction of [VDocRAG](https://arxiv.org/abs/2504.09795) (Tanaka et
al., CVPR 2025) — retrieval-augmented generation over visually-rich documents
(PDFs, charts, tables), retrieving and generating directly from page images with
no OCR step, built on [NTT's released checkpoints](https://github.com/nttmdlab-nlp/VDocRAG).

Upload a PDF, ask a question, get an answer grounded in a retrieved page image.
Runs entirely in Google Colab (free tier, T4 GPU) via a Gradio app.

## Status

**All application code is written and unit-tested; Step 1 (the GPU smoke test)
has not been run yet — that has to happen in your Colab, not here.** Every
module above the model-loading boundary (ingest, index, page-cap logic, the
Gradio handlers, all prompt-template construction) is built against NTT's
*actual* source code (read directly from their repo, not inferred) and has
real, passing tests with mocked models. The only things genuinely unverified
are the two calls that require a live GPU: does Phi-3-vision actually load
under `eager` attention + 4-bit quantization on a T4, and does PEFT's
multi-adapter hot-swap work cleanly on this `trust_remote_code` model class.
Run `notebooks/00_smoke_test.ipynb` first — see `docs/implementation_plan.md`
Section 6.1 and `docs/ntt_api_reference.md` for what it checks and why.

## Repo layout

```
vdocrag_app/
├── telemetry.py      # structured JSONL logging + timing/VRAM instrumentation
├── ingest.py         # PDF -> per-page PIL images (tested against a real PDF)
├── limits.py         # page-count UX cap, formula-driven from real telemetry
├── index.py          # FAISS wrapper (tested: add/search/save/load/remove)
├── model_manager.py  # loads Phi-3-vision + both adapters; shared-base hot-swap
│                        OR NTT's tested independent-load path, config-selectable
├── retriever.py      # wraps VDocRetriever; prompt templates verified byte-for-byte
│                        against NTT's confirmed source strings
├── generator.py       # wraps VDocGenerator; same verification approach
└── app_state.py       # orchestration (index_pdf / ask pipelines), fully unit-tested
                          with mocked retriever/generator -- no GPU needed to test this layer

app.py                # Gradio Blocks UI, thin wiring over app_state.py,
                         handler logic unit-tested with mocked app + real gr.update()

notebooks/
└── 00_smoke_test.ipynb   # Step 1 — run this in Colab first, before anything else

docs/
├── implementation_plan.md   # full handoff doc — read this first
├── ntt_api_reference.md     # confirmed API facts read from NTT's actual source
├── licenses.md              # why NTT's package is never vendored into this repo
└── colab_workflow.md        # session-start checklist, dev loop
```

### What's actually been verified vs. what's still assumption
Every GPU-free module (`ingest`, `index`, `limits`, `app_state`, `app.py`'s
handlers, and the pure prompt-template functions in `retriever`/`generator`) has
been run against real inputs in a sandboxed test environment — see the test
output referenced in the project's build history for exact commands. What has
**not** been run anywhere yet: `ModelManager.setup()` itself, and therefore
`encode_query`/`encode_document`/`answer`'s actual model forward passes. That
gap is exactly what `00_smoke_test.ipynb` closes.

## Important: this repo never contains NTT's package or their model weights

NTT's `vdocrag` package is installed fresh every Colab session via
`pip install git+https://github.com/nttmdlab-nlp/VDocRAG.git` — never copied into
this repo. This isn't a style choice; their license (evaluation-only, no
redistribution to third parties) requires it. See `docs/licenses.md`.

## Running

See `docs/colab_workflow.md` for the full session-start checklist. Short version:
open `notebooks/00_smoke_test.ipynb` in Colab, run it top to bottom, confirm every
check passes.
