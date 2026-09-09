# Colab workflow

Dev loop for this project: edit code on your machine → `git push` → in Colab,
`git pull` (or fresh clone) → run notebook cells → `demo.launch(share=True)` →
test in your browser → find bugs → repeat.

**Practical note learned the hard way**: don't hand-patch cells in a live
Colab session and expect them to sync back to the repo. When something needs
fixing, fix the notebook file locally, push, then open a **fresh** Colab tab
(File → Open notebook → GitHub) rather than reusing a session that's drifted.
Reusing a live session across many patches is how cell-execution-order bugs
and stale-variable bugs crept in during development — a fresh session +
running top-to-bottom is more reliable than it feels like it should be.

## Session start checklist (every fresh Colab runtime)
1. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
2. Set `HF_HOME` to a Drive-backed path so weights download once, not every session.
3. `!apt-get install -y poppler-utils` (needed by `pdf2image`, not a pip package).
4. Install dependencies in **two separate `pip install` calls**, not one
   `pip install -r requirements-colab.txt` pass — see that file's header
   comment for why (short version: a `huggingface-hub` version conflict
   between `transformers` and `gradio` needs a later command to override an
   earlier one, which a single resolve pass can't do). All three notebooks do
   this correctly already as sequential shell cells.
5. Install NTT's package by URL, never from a local copy — see `docs/licenses.md`.
6. Clone/pull this repo into `/content/repo` and `sys.path.insert(0, '/content/repo')`
   so `vdocrag_app.*` is importable (note: `vdocrag_app`, not `vdocrag` — see
   `docs/implementation_plan.md` Section 4.6a for why the local package isn't
   named `vdocrag`, which collides with NTT's own package of that name).

## Order of operations
1. `notebooks/00_smoke_test.ipynb` — confirms the hardware/config works at all
   (eager attention, 4-bit quantization, LoRA hot-swap, the `num_crops`
   memory/accuracy trade-off). **Passed** — see `docs/implementation_plan.md`
   Section 4.6g for the confirmed results.
2. `notebooks/01_wrapper_test.ipynb` — confirms the actual production classes
   (`ModelManager`, `VDocRetrieverWrapper`, `VDocGeneratorWrapper`) reproduce
   Step 1's confirmed-working behavior, not just the hand-built raw code.
   **Passed** — generation output matched Step 1's baseline exactly.
3. `notebooks/run_in_colab.ipynb` — launches the real Gradio app via `app.py`.
   This is the first run that exercises the *entire* pipeline together
   (real PDF → `ingest.py` → `VDocRetrieverWrapper` → `DocumentIndex` →
   `VDocGeneratorWrapper`) rather than NTT's two canned example images.

## Known Colab constraints this workflow works around
- ~90 min idle timeout / ~12hr hard session cap — Drive persistence (HF cache,
  FAISS index, page-image cache) means a killed session doesn't mean starting
  from zero on the next run.
- No `faiss-gpu` wheel for current CUDA on Colab — `faiss-cpu` is used
  deliberately, and is not a bottleneck at demo scale (hundreds–low-thousands of
  vectors, exact search).
- Unpinned dependencies drift fast and silently. Every hard version pin in
  `requirements-colab.txt` exists because something broke without it —
  `transformers` (twice: the 4.x/5.x major-version break, and the
  `huggingface-hub` transitive conflict) is the repeat offender. When adding
  a new dependency, assume it needs a pin until proven otherwise, not the
  other way around. This includes `torch` itself: Colab ships a working
  CUDA-enabled build by default, but `accelerate`/`bitsandbytes`/`peft`/
  `gradio`'s own version requirements can cause pip to silently swap it for
  a CPU-only build from plain PyPI if `torch` isn't pinned in the same
  install command — confirmed happening (`AssertionError: Torch not
  compiled with CUDA enabled`, right after a `pip install` cell that never
  mentioned torch at all). Every notebook now pins `torch` explicitly.
  