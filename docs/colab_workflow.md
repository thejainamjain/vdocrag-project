# Colab workflow

Dev loop for this project: edit code on your machine → `git push` → in Colab,
`git pull` (or fresh clone) → run notebook cells → `demo.launch(share=True)` →
test in your browser → find bugs → repeat.

## Session start checklist (every fresh Colab runtime)
1. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
2. Set `HF_HOME` to a Drive-backed path so weights download once, not every session.
3. `!apt-get install -y poppler-utils` (needed by `pdf2image`, not a pip package).
4. `pip install -r requirements-colab.txt` — this pulls NTT's package by URL,
   never from a local copy. See `docs/licenses.md` for why.
5. Clone/pull this repo into `/content/repo` and `sys.path.insert(0, '/content/repo')`
   so `vdocrag.*` is importable.

## Order of operations
1. `notebooks/00_smoke_test.ipynb` — must pass before anything else. Gate, not a
   suggestion: every later step assumes eager-attention + 4-bit + LoRA hot-swap
   actually works on this hardware.
2. Once it passes, its logged output (`logs/step1_smoke_test.jsonl`) becomes the
   first real data point for `vdocrag.limits.page_cap()` — no longer an estimate.
3. `app.py` (Gradio) is built and run from a separate cell/notebook once Steps
   2–5 of the implementation plan exist.

## Known Colab constraints this workflow works around
- ~90 min idle timeout / ~12hr hard session cap — Drive persistence (HF cache,
  FAISS index, page-image cache) means a killed session doesn't mean starting
  from zero on the next run.
- No `faiss-gpu` wheel for current CUDA on Colab — `faiss-cpu` is used
  deliberately, and is not a bottleneck at demo scale (hundreds–low-thousands of
  vectors, exact search).
