# Implementation Plan

## Compute split
- **MacBook (no GPU)**: environment setup, data download/prep, writing/debugging
  code with tiny inputs (batch size 1, CPU or MPS), git management, report writing.
- **Friend's RTX GPU laptop**: all actual training runs (Phase 2) and any
  full-subset inference batches (Phase 1) that would be too slow on CPU.

You can develop entirely on the Mac and only `git pull` + run scripts on the
RTX machine when you need real training speed -- the code itself doesn't
change between machines.

## Phase 0 — Environment
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `pip install -e .` (installs `vdocrag` package in editable mode)
4. On the RTX machine specifically: confirm `torch.cuda.is_available()` returns
   `True` and `nvidia-smi` shows the GPU, before running any training script.

## Phase 1 — Baseline reproduction (uses NTT's pretrained checkpoints)
1. `python data/download_opendocvqa.py --configs chartqa --max_examples 300`
   (start small; ChartQA images are lightweight, good for a first run)
2. `bash scripts/run_baseline_inference.sh`
   - encodes queries + corpus with `VDocRetriever-Phi3-vision`
   - retrieves top-10 per query
   - generates answers with `VDocGenerator-Phi3-vision` using top-3 retrieved images
3. Evaluate:
   ```python
   from vdocrag.utils.eval import evaluate_qa, recall_at_k, ndcg_at_k, load_qrels, load_rankings
   metrics = evaluate_qa("outputs/answers/baseline_answers.json", "data/processed/gold_answers.json")
   print(metrics)
   ```
4. **Deliverable for the 6% implementation mark**: a short table comparing your
   reproduced numbers (on your subset) against the paper's reported numbers,
   plus 2-3 qualitative examples (question, retrieved image, generated answer).

## Phase 2 — From-scratch training (Qwen2-VL-2B, on the RTX GPU)
1. Reuse the same `data/processed/` files (already in the right JSONL format).
2. `bash scripts/train_retriever_lora.sh` — trains the bi-encoder with
   in-batch contrastive loss (see `src/vdocrag/retriever/train.py` for the
   full loop, deliberately written as plain PyTorch so every step is explainable).
3. Encode + search again, this time with your own LoRA adapter:
   ```bash
   python -m vdocrag.retriever.encode --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
     --lora_name_or_path outputs/retriever-qwen2vl2b-lora \
     --input_jsonl data/processed/test_queries.jsonl --mode query \
     --output_path outputs/embeddings/query.pkl
   # (repeat for corpus, then run search.py)
   ```
4. `bash scripts/train_generator_lora.sh` — SFT on (question, retrieved images) -> answer.
5. Generate + evaluate the same way as Phase 1, now with your own model.
6. Compare three rows in your final report: **NTT baseline (Phi-3-vision)** vs
   **your from-scratch model (Qwen2-VL-2B)** vs **paper's reported numbers**.

## Phase 3 — Extension (separate, later)
Candidates already discussed: reranking stage, backbone comparison (this repo
already sets you up for that since the backbone is a config value), multi-hop
iterative retrieval, or a distractor-robustness benchmark. Plan is to branch
off (`git checkout -b extension/reranker`) once Phases 1-2 are solid.

## What to actually explain in the viva from this repo
- `retriever/modeling.py` — how the bi-encoder pools a single vector per
  image/query (EOS-token pooling) and the in-batch InfoNCE loss.
- `retriever/train.py` — LoRA + 4-bit quantization, why (fits on 8-16GB GPU).
- `generator/modeling.py` / `generate.py` — how retrieved images + question
  become a single generation call, no OCR anywhere in the pipeline.
- `utils/eval.py` — Recall@k / nDCG for retrieval, EM/F1/relaxed-accuracy for QA,
  matching how the original paper evaluates.
