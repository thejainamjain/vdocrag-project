# VDocRAG (from scratch) — Retrieval-Augmented Generation over Visually-Rich Documents

This repo re-implements the core ideas of **VDocRAG** (Tanaka et al., CVPR 2025) —
a RAG pipeline that retrieves and answers questions over document **page images**
directly (no OCR/text-parsing step), preserving charts/tables/layout information
that text-based RAG loses.

Reference: https://arxiv.org/abs/2504.09795
Official repo: https://github.com/nttmdlab-nlp/VDocRAG

## Project structure

- **Phase 1 — Baseline reproduction**: run NTT's released pretrained checkpoints
  (`VDocRetriever-Phi3-vision`, `VDocGenerator-Phi3-vision`) on a subset of
  OpenDocVQA to confirm we understand and can reproduce the pipeline.
- **Phase 2 — From-scratch training**: our own bi-encoder retriever + generator,
  LoRA fine-tuned on top of a smaller open backbone (**Qwen2-VL-2B-Instruct**)
  so training is feasible on a single consumer GPU (8–16GB).
- **Phase 3 (later)** — Extension work (reranking / backbone comparison / etc.)
  will live in a separate branch/folder once base implementation is solid.

## Repo layout

```
configs/            YAML configs for training runs
data/                dataset download + prep scripts
src/vdocrag/
  retriever/         bi-encoder model, dataset, train, encode, search
  generator/         LVLM answer generator, dataset, train, generate
  utils/             eval metrics, image preprocessing helpers
scripts/             shell scripts to run each stage
notebooks/           exploratory notebooks
tests/               unit tests
docs/                write-ups, implementation notes
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Compute plan

- **MacBook (no GPU)**: data prep, small tests, orchestration, writing configs.
- **Friend's RTX GPU laptop**: actual training runs (Phase 2) and heavier
  inference batches (Phase 1 baseline on the full subset).

See `docs/implementation_plan.md` for the full step-by-step plan.
