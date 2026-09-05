# VDocRAG — Implementation Handoff Document (v2)

**Purpose of this document**: Complete handoff for continuing work on this project in a
fresh AI session with zero prior context. Covers the paper being reproduced, every
decision made so far and *why*, what was corrected during a research pass and on what
evidence, and what remains to verify empirically once code actually runs.

**Status as of this document**: No code has been written yet for this stage of the
plan. All five open questions from v1 have been resolved on paper (with sourced
reasoning, not guesses) or reduced to a single concrete verification step. This
document reflects those resolutions. Nothing below has been run yet — Step 1 (the
smoke test) is still the first thing to actually execute.

---

## 1. What this project is

We are building a working reproduction of **VDocRAG** (Tanaka et al., CVPR 2025),
a Retrieval-Augmented Generation (RAG) system for **visually-rich documents**
(PDFs, slides, charts, tables). Paper: https://arxiv.org/abs/2504.09795
Official code + released checkpoints: https://github.com/nttmdlab-nlp/VDocRAG

### Core idea of the paper
Normal RAG parses a document to text (OCR) before retrieval/generation. This loses
visual information (chart colors, table layout, spatial structure). VDocRAG instead
retrieves and generates **directly from page images**, with no OCR/parsing step at
any point in the pipeline.

### The paper's two components
- **VDocRetriever**: a bi-encoder built on a vision-language model (LVLM) backbone
  (Phi-3-vision-128k-instruct, 4.2B params). Encodes a text query and a document
  *page image* into the same vector space (single dense vector per item, EOS-token
  pooled, L2-normalized). Retrieval = cosine similarity via max inner product search.
- **VDocGenerator**: same LVLM backbone, takes the question + top-k retrieved page
  images, generates the answer directly (still no OCR). Paper finds k=3 optimal
  (Figure C, supplementary).

### What NTT (the paper's authors) released publicly
- Pretrained/fine-tuned checkpoints on Hugging Face, both LoRA adapters over
  `microsoft/Phi-3-vision-128k-instruct`:
  `NTT-hil-insight/VDocRetriever-Phi3-vision`, `NTT-hil-insight/VDocGenerator-Phi3-vision`
- Full training/eval code: https://github.com/nttmdlab-nlp/VDocRAG (built on the
  Tevatron retrieval toolkit — same lineage as `Tevatron/dse-phi3-docmatix-v1`,
  which matters for Section 4.1 below)
- The `OpenDocVQA` benchmark dataset (43k QA pairs, 206k document images) on
  Hugging Face: `NTT-hil-insight/OpenDocVQA` and `NTT-hil-insight/OpenDocVQA-Corpus`

---

## 2. Project goal and scope (unchanged from v1)

- **This is a course/project-context reproduction + light extension**, not a
  from-scratch re-training of the models. Build on NTT's released checkpoints.
- **Extension work** (reranking, backbone comparison, robustness testing) is a
  separate, later phase — not part of this build.
- **Conference-grade permanent hosting** (Hugging Face Spaces) is also deferred.
- **Right now**: build a working demo — upload a PDF, ask a question, get an answer
  grounded in a retrieved page image, all backed by NTT's released checkpoints —
  runnable and testable solo, entirely inside **Google Colab**, using the Mac only
  as a code editor and browser for testing.

### Hardware / environment constraints
- MacBook, no GPU. Editing/git/browser only.
- Compute: **Google Colab free tier** (T4 GPU, ~15GB VRAM, Turing architecture,
  ~90 min idle timeout / ~12hr hard cap).
- Kaggle rejected for hosting specifically (Gradio `share=True` has known tunnel
  problems on Kaggle's sandboxing); usable later as pure-compute fallback only.

---

## 3. Planned architecture (single Colab notebook + Gradio app)

```
┌─────────────────────────────────────────────────────────────┐
│              Single Gradio Blocks app (app.py)                │
│         runs inside one Google Colab notebook session         │
│                                                                 │
│  Tab 1 "Upload & Index"                                        │
│    PDF upload → per-page images (pdf2image + poppler)         │
│    → encode each page with VDocRetriever → add to FAISS index │
│    → soft page-count warning / hard cap (Section 5.4)         │
│                                                                 │
│  Tab 2 "Ask"                                                    │
│    question text → encode with VDocRetriever → FAISS search   │
│    → top-k (=3, per paper) retrieved page images → VDocGenerator│
│    → UI shows answer + the retrieved page image(s) together   │
│                                                                 │
│  Tab 3 "Benchmark" (optional, stretch goal)                    │
│    runs a small OpenDocVQA subset through the same pipeline,   │
│    reports Recall@k / nDCG / Exact-Match / F1                  │
└─────────────────────────────────────────────────────────────┘
```

Dev loop: edit code on Mac → `git push` → in Colab, `git pull` → run notebook cells
→ `demo.launch(share=True)` → test in Mac browser → find bugs → repeat.

### Repo structure (updated: no vendored NTT code — see Section 4.5)
```
vdocrag-project/
├── app.py                     # Gradio Blocks app (3 tabs above)
├── requirements-colab.txt     # includes git+https://github.com/nttmdlab-nlp/VDocRAG.git
├── vdocrag/
│   ├── retriever.py           # wraps VDocRetriever.load(...) + encode calls
│   ├── generator.py           # wraps VDocGenerator.load(...) + generate calls
│   ├── index.py                # FAISS wrapper: add / save / load / search
│   ├── ingest.py               # PDF -> list of page images (pdf2image)
│   ├── limits.py                # page-cap formula, driven by observed timing (Section 5.4)
│   └── telemetry.py            # structured logging + timing/VRAM instrumentation (Section 6)
├── eval/
│   ├── run_opendocvqa_eval.py
│   └── metrics.py
├── notebooks/
│   └── run_in_colab.ipynb     # mount Drive, set HF_HOME, pip install (incl. NTT's repo
│                                 by URL, never vendored), run app
├── docs/
│   └── colab_workflow.md
└── README.md
```

**NTT's `vdocrag` package is never copied into this repo.** It's installed fresh
each session via `pip install git+https://github.com/nttmdlab-nlp/VDocRAG.git` in
`requirements-colab.txt` / the install cell. This is a licensing requirement, not
a style choice — see Section 4.5.

### Colab operational details — corrected from v1

- **Attention implementation: `eager`, not `sdpa`.** v1 assumed `sdpa` was a safe,
  "marginally slower" substitute for `flash_attention_2` on non-Ampere hardware.
  This was **wrong and would fail to load**, not just be slower. Confirmed at the
  source level: `microsoft/Phi-3-vision-128k-instruct`'s custom `modeling_phi3_v.py`
  declares `Phi3VPreTrainedModel._supports_sdpa = False` (alongside
  `_supports_flash_attn_2 = True`). `transformers`' `_check_and_enable_sdpa` hard-gates
  on this flag and raises `ValueError` before the model loads if you request
  `attn_implementation="sdpa"` — confirmed by an identical, reported failure on the
  sibling text-only Phi-3 model on a T4 (HF/transformers issue #31863). A fully
  implemented `Phi3SdpaAttention` class exists in the file and is wired into
  `PHI3_ATTENTION_CLASSES`, but Microsoft's own team has stated in a HF discussion
  thread that SDPA is deliberately untested/ungated for this model — "we have not
  fully tested SDPA yet." Community-confirmed working fallback on non-flash-attention
  hardware: `attn_implementation="eager"`. FlashAttention itself has never shipped
  Turing (T4) support at all, confirming the original decision to avoid it — just not
  the substitute chosen.
  - **Known risk to watch in the smoke test**: `eager` attention is O(seq_len²) memory
    (no IO-aware tiling like flash/sdpa), and one community report shows a T4 OOM using
    `eager` on the unquantized text-only Phi-3 model. Under 4-bit quantization this is
    probably fine, but it's the specific failure mode Step 1 needs to catch, not assume
    away — see mitigations in Section 6.1.

- **Quantization**: still 4-bit via `bitsandbytes` (NF4, double quant, bf16 compute
  dtype). Estimated footprint: `4.2e9 × ~4.5 bits / 8 ≈ 2.36GB` per model copy
  (NF4's effective bits/param including quantization-constant overhead), not the
  exact "~4GB total for two copies" guessed in v1 — see Section 4.4 for the full
  VRAM accounting that this feeds into.

- **FAISS**: `faiss-cpu`, unchanged from v1 — no faiss-gpu wheel for current CUDA on
  Colab, and at demo scale (hundreds–low-thousands of vectors) exact CPU search is
  effectively instant. Confirmed this is not a VRAM concern either way (Section 5.4).

- **Session persistence**: unchanged — mount Drive at session start, `HF_HOME` on
  Drive so weights download once, FAISS index + page images persisted to Drive with
  an existence check before rebuilding.

- **PDF only for now**; PPTX deferred (LibreOffice headless install cost).

---

## 4. Resolved corrections (formerly "open questions," now settled with sourced reasoning)

### 4.1 — Do NOT reimplement NTT's retriever/generator wrapper logic (unchanged, reaffirmed)
Call their actual `VDocRetriever.load(...)` / `VDocGenerator.load(...)` and exact
prompt templates as shown in their README quickstart. Our code (Gradio app, FAISS
wrapper, PDF ingestion) is a thin layer *around* their code.

**Resolution on the `attn_implementation` override question (was a loose end):**
Couldn't directly read NTT's `load()` source (not surfaced by search/fetch tools),
but found strong precedent: `Tevatron/dse-phi3-docmatix-v1` — a sibling model from
the exact toolkit VDocRAG is built on, and one of the paper's own retrieval
baselines — loads via a thin pass-through wrapper around plain
`AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2", ...)`,
i.e. the kwarg is forwarded, not hardcoded inside a custom loader. **This is now
directly confirmed** by cloning and reading their actual source
(`src/vdocrag/vdocretriever/modeling/vdocretriever.py`) — see
`docs/ntt_api_reference.md` for the full confirmed API surface, including the
more significant finding that `load()` calls `merge_and_unload()` internally
when given a LoRA adapter, which affects the Section 4.4 shared-model decision
(resolved via a `share_base_model` config flag in `vdocrag_app/model_manager.py`
— both the VRAM-optimal shared/hot-swap path and NTT's simpler tested
independent-`.load()` path are implemented, selectable without other code
changes).

### 4.2 — Don't reimplement "dynamic cropping" (unchanged, reaffirmed)
Phi-3-vision's own `AutoProcessor` handles dynamic-resolution image splitting
internally. Call the processor as NTT's quickstart does.

### 4.3 — Attention implementation + quantization + LoRA — resolved via the smoke test's target config
See "Colab operational details" above for the `sdpa`→`eager` correction. The
original concern here (does `trust_remote_code` + `bitsandbytes` 4-bit +
`attn_implementation` override + PEFT LoRA all work together, simultaneously, on
this custom-code model) is still **empirically unverified** — that's what Step 1
is for — but the target configuration to test is now correct (`eager`, not `sdpa`),
so the smoke test is testing the right thing. See Section 6.1 for the concrete
recipe and pass/fail criteria.

### 4.4 — Shared base model with adapter hot-swap: **decided, not just "worth considering"**
v1 flagged this as an open question needing a VRAM measurement before deciding.
Resolved as follows, with the reasoning that tips it:

- **Weight VRAM math** (analytical, to be confirmed by Step 1's actual numbers):
  one 4-bit Phi-3-vision copy ≈ 2.36GB; two full separate copies ≈ 4.7GB; LoRA
  adapters (rank 8, alpha 64, targeting `*_proj`) are tens of MB each, negligible
  either way. On a 15GB T4 this delta (≈2.3GB) is real but not the dominant cost —
  eager-attention activation memory during page encoding is likely the bigger driver,
  and that's identical whether you share the base model or not.
- **The adapter-swap mechanism itself is cheap, confirmed from a working reference
  implementation** (a production HF Space hot-swapping PEFT LoRA adapters on a
  shared Whisper backbone): `load_adapter()` — first load, ~2s; `set_adapter()` —
  subsequent swaps, **~50ms**. Confirmed via a HF forum thread that PEFT correctly
  isolates adapters from each other (no cross-contamination) once both are loaded.
- **Decision: share the base model, hot-swap adapters.** Your query flow swaps
  adapters twice per question (retriever → generator), so ~100ms/query of swap
  overhead is trivial against generation latency (paper: ~790ms on an A100; a T4
  under eager attention will be slower still). This saves ~2.3GB of VRAM for
  near-zero latency cost and meaningfully less code complexity (one model object).
- **Verification item for Step 1**: PEFT's multi-adapter API is standard for
  mainline `transformers` architectures; Phi-3-vision is a `trust_remote_code=True`
  custom class. It *should* work identically (PEFT operates on `nn.Linear` module
  names, not architecture internals) but this is exactly the kind of assumption
  that deserves a pass/fail check rather than being taken on faith — include
  "does `load_adapter`/`set_adapter` work cleanly on the custom Phi3V class" as an
  explicit smoke-test check (see Section 6.1).

### 4.5 — License check: **done**, with a concrete decision it drives

**Phi-3-vision**: confirmed MIT (Microsoft's own model card states this directly).
No restrictions relevant to this project.

**NTT License**: could not fetch `nttmdlab-nlp/VDocRAG`'s exact `LICENSE` file
directly (not surfaced by available tools), but found NTT's standard **"SOFTWARE
LICENSE AGREEMENT FOR EVALUATION"** — identical boilerplate text confirmed across
multiple sibling NTT research-lab repos and NTT-affiliated HF Spaces, strongly
suggesting it's applied org-wide. **High confidence, not certainty — confirm the
actual file before writing Step 3's code**, since the terms below drive a real
architectural decision:

- **§1**: grants use "internally for the purposes of testing, analyzing, and
  evaluating the methods or mechanisms as shown in the research paper" — a
  course-project reproduction/evaluation is squarely inside this scope. Fine on
  purpose.
- **§4(b)(i)**: "User shall not... sell, assign, lease, **distribute**, or otherwise
  transfer the Software to any third party," and shall not "copy or reproduce the
  Software in any manner" except as allowed.
- **§4(b)(iv)**: "User shall not... modify, disassemble, decompile, reverse
  engineer or translate the Software."

**Decision this drives (resolves v1's open question 5 — vendor vs. pip-install):**
**Do not vendor.** Copying NTT's package source into this repo and pushing it would
plausibly constitute "distribute... to any third party" the moment the repo is
visible to anyone else. Install fresh each session via
`pip install git+https://github.com/nttmdlab-nlp/VDocRAG.git`; never commit a copy
of their source or weights into this repo. This is already compatible with the
existing session-persistence design (re-clone/pull at the start of every ephemeral
Colab session) — no architecture change needed, just a constraint on what goes in
version control.

**Secondary implication for Step 3**: if `load()` turns out to hardcode
`attn_implementation` internally (see 4.1's action item) rather than accepting an
override, patching their `.py` files in place would fall outside §4(b)(iv)'s grant,
even for a private, non-published class project. The safer path in that scenario is
overriding behavior from *your* wrapper code (subclassing, monkey-patching a class
attribute from your own module) rather than editing their shipped files — check
this during Step 3 before assuming either path is needed.

### 4.6a — `vdocrag` vs `vdocrag_app`: a naming collision found the hard way
Our own local package was originally named `vdocrag`. NTT's released package is
**also** named `vdocrag` (`setup(name='vdocrag', ...)` in their `setup.py`, read
during the Section 4.1 research pass — the name was visible then, but the
collision risk wasn't connected until it actually broke). Two non-namespace
Python packages can't share a top-level name on `sys.path`: Python resolves
whichever is found first, and the other's submodules become invisible under
that name. This surfaced as `ModuleNotFoundError: No module named
'vdocrag.telemetry'` when actually running Step 1 in Colab — our package was
correctly on `sys.path`, but Python had already resolved `vdocrag` to NTT's
installed package first. Fixed by renaming our local package to `vdocrag_app`
throughout the repo (imports, notebook, docs). Concrete example of why reading
source and actually running code catch different classes of problem — this one
was sitting in plain sight in a file we'd already read.

### 4.7 — Whole-page-only retrieval granularity (unchanged from v1)
Known limitation of the paper's own approach, not something to "fix" in this base
implementation. Stated explicitly so imperfect fine-detail accuracy isn't mistaken
for a reproduction bug.

---

## 5. Remaining verification items (things that need Step 1's real numbers, not more research)

These are no longer "undecided" — each has a concrete resolution path and, where
applicable, a provisional number to start from. They become fully resolved once
Step 1 runs.

1. **Smoke test must pass with `eager` attention** (Section 4.3 / 6.1). If it
   doesn't — even after the mitigations in 6.1 — the whole hardware plan needs
   reconsidering, not just this one config value.
2. **VRAM measurement confirms or corrects the 4.4 shared-model decision.** The
   ~2.36GB/copy estimate and the "share is worth it" conclusion should hold, but
   verify rather than assume — particularly whether eager attention's activation
   memory changes the calculus at higher image resolutions.
3. **Confirm the actual NTT `LICENSE` file text** matches the boilerplate found
   (Section 4.5) before Step 3.
4. **PDF page-volume cap — formula-driven, not a fixed number.** Not a VRAM
   constraint (page count doesn't scale VRAM — you encode one page at a time); it's
   a wall-clock UX / Colab-idle-timeout constraint. Analytical estimate pending
   real data: paper reports 204.4ms/page encoding on an A100 (Table 7); T4 is
   roughly 4–6x slower on both compute-bound and memory-bandwidth-bound workloads
   (spec-based ratio), and eager attention adds overhead beyond that scaling, while
   4-bit quantization mostly helps memory-bandwidth-bound steps (decode) rather than
   compute-bound encoding — net estimate **~0.8–1.5s/page on a T4**, treat as
   order-of-magnitude only. Design (see `vdocrag/limits.py` in Section 3's repo
   layout):
   ```python
   TARGET_MAX_WAIT_S = 120   # no warning needed below this
   WARN_MAX_WAIT_S   = 480   # hard-block by default above this (opt-in override available)

   def page_cap(observed_s_per_page: float) -> dict:
       return {
           "no_warning_pages": int(TARGET_MAX_WAIT_S / observed_s_per_page),
           "warn_pages":       int(WARN_MAX_WAIT_S / observed_s_per_page),
       }
   ```
   Seed `observed_s_per_page` from the estimate above; after Step 1 and any real
   indexing run, recompute it from the telemetry log (Section 6) rather than leaving
   it static:
   ```python
   df = pd.read_json(log_path, lines=True)
   enc = df[(df.component == "retriever") & (df.function == "encode_document") & (df.status == "ok")]
   observed_s_per_page = enc["duration_ms"].median() / 1000
   ```
   UI behavior: ≤`no_warning_pages` → index immediately with a progress bar;
   between the two thresholds → confirm-before-start banner with a time estimate;
   above `warn_pages` → blocked by default with a suggestion to trim the PDF, plus
   an explicit "index anyway" opt-in (not a hard wall — someone testing multi-hop
   behavior on a large corpus is a legitimate use case for this tool).
5. **Vendor vs. pip-install** — resolved, not just "leaning": install by URL,
   confirmed by the license (Section 4.5). No longer open.

---

## 6. Logging / telemetry — new in this revision

Not part of v1. Added because (a) debugging a `trust_remote_code` + quantization +
LoRA stack on unfamiliar hardware needs more than print statements, and (b) items
5.2 and 5.4 above need real timing/VRAM data to resolve, and instrumenting the code
once gets you that data as a side effect of normal operation instead of a separate
benchmarking pass.

**Not using OpenTelemetry.** OTel is built for correlating traces across
distributed services via a collector backend (Jaeger, Honeycomb, etc.) — this is a
single Python process in a single ephemeral Colab notebook. There's no second
service to correlate against, so OTel's actual value proposition (multi-service
trace stitching) doesn't apply here; adopting it would mean taking on its span/
tracer API and exporter setup for no correlation benefit. Using stdlib `logging`
configured for structured output gets the same debugging value and directly
produces analyzable data, with no new dependency.

**Design**: `vdocrag/telemetry.py` — a JSONL log handler (human-readable console
output + machine-readable file persisted to Drive) plus a `@log_call(component)`
decorator applied to every GPU-touching function (model load, `encode_query`,
`encode_document`, `index.search`, `generator.answer`, per-page PDF ingestion).
Each call logs: component, function, duration_ms, VRAM before/after/peak, and
status (ok/error, with the exception repr on failure).

```python
import logging, json, time, functools, torch
from pathlib import Path
from datetime import datetime, timezone

LOG_DIR = Path("/content/drive/MyDrive/vdocrag-project/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

class JsonlHandler(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.path = path
    def emit(self, record):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            entry.update(record.extra_fields)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

def setup_logging(session_name: str):
    logger = logging.getLogger("vdocrag")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(JsonlHandler(LOG_DIR / f"{session_name}.jsonl"))
    return logger

logger = logging.getLogger("vdocrag")

def log_call(component: str):
    """Decorator: logs timing + VRAM delta + peak for any GPU-touching function."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
            t0 = time.time()
            status, err = "ok", None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                status, err = "error", repr(e)
                raise
            finally:
                duration_ms = (time.time() - t0) * 1000
                fields = {
                    "component": component,
                    "function": fn.__name__,
                    "duration_ms": round(duration_ms, 1),
                    "vram_before_gb": round(mem_before / 1e9, 3),
                    "vram_after_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                    "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                    "status": status,
                }
                if err:
                    fields["error"] = err
                logger.info(f"{component}.{fn.__name__} {status} ({duration_ms:.0f}ms)",
                             extra={"extra_fields": fields})
        return wrapper
    return decorator
```

Usage:
```python
class VDocRetriever:
    @log_call("retriever")
    def encode_document(self, image):
        ...
```

The resulting JSONL is directly `pandas.read_json(path, lines=True)`-able, which is
exactly what Section 5 items 2 and 4 need — VRAM peak under real load, and per-page
encoding time distribution — as data you already have from normal runs, not a
separate measurement task.

---

## 7. Step-by-step build plan (updated)

**Step 1 — Isolated smoke test (must pass before anything else is built)**
Corrected target config: `attn_implementation="eager"` (not `sdpa`), 4-bit via
`bitsandbytes`, both LoRA adapters loaded onto a shared base model via PEFT
hot-swap. Reproduce NTT's own README quickstart example (their two hardcoded
queries + images) through this configuration.
```python
import torch, time
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "microsoft/Phi-3-vision-128k-instruct"
RETRIEVER_ADAPTER = "NTT-hil-insight/VDocRetriever-Phi3-vision"
GENERATOR_ADAPTER = "NTT-hil-insight/VDocGenerator-Phi3-vision"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, device_map="cuda",
    torch_dtype=torch.bfloat16, attn_implementation="eager",
    quantization_config=bnb_config,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Base load: {time.time()-t0:.1f}s, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
      f"peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

peft_model = PeftModel.from_pretrained(base_model, RETRIEVER_ADAPTER, adapter_name="retriever")
peft_model.load_adapter(GENERATOR_ADAPTER, adapter_name="generator")
print(f"Adapters attached, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
      f"peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

peft_model.set_adapter("retriever")
# ... run NTT's README query/image pair, confirm sane similarity score ...
peft_model.set_adapter("generator")
# ... run their generation example, confirm sane output ...
```
Pass/fail criteria: loads without error at each stage; similarity scores /
generated answer are in the same ballpark as NTT's published README output; peak
VRAM stays comfortably under T4's ~15GB with margin for Gradio + PDF conversion;
adapter hot-swap works cleanly on the custom Phi3V class. If `eager` attention OOMs
even under 4-bit, mitigation order: reduce max image resolution first (paper's own
Table G shows 672×672 keeps ~72.8 vs 1344×1344's 72.9 nDCG@5 at meaningfully less
compute — a legitimate lever, not just for speed), then batch size (should be 1
anyway for this demo), then reconsider whether gradient checkpointing is relevant
(likely not — no training happening here, inference only).

Instrument this step with `telemetry.py` (Section 6) from the start — its output
directly answers Section 5 items 2 and 4.

**Step 2 — PDF ingestion module** (`vdocrag/ingest.py`)
PDF upload → list of PIL page images via `pdf2image` (`poppler-utils`). No model
involved, testable without GPU.

**Step 3 — Retriever wrapper** (`vdocrag/retriever.py`)
First action: open NTT's actual `load()` signature (after installing their package
by URL, per 4.5) and confirm whether `attn_implementation` passes through cleanly
(per 4.1's precedent-based assumption) or is hardcoded. Then thin wrapper around
`VDocRetriever.load(...)`, exposing `encode_query(text)` / `encode_document(image)`,
decorated with `@log_call("retriever")`.

**Step 4 — FAISS index wrapper** (`vdocrag/index.py`)
`add(doc_id, embedding)`, `search(query_embedding, top_k)`, `save(path)` / `load(path)`
for Drive persistence. `top_k` default = 3, per the paper's own finding (Figure C).

**Step 5 — Generator wrapper** (`vdocrag/generator.py`)
Thin wrapper around `VDocGenerator.load(...)`, exposing `answer(question, images) -> str`,
decorated with `@log_call("generator")`.

**Step 6 — Gradio app** (`app.py`)
Wire Steps 2–5 into the 3-tab UI. Include the page-cap UI behavior from Section 5
item 4 in the "Upload & Index" tab. Build and test tab by tab.

**Step 7 — `run_in_colab.ipynb`**
Mount Drive, set `HF_HOME`, install `poppler-utils` + Python deps (including
`pip install git+https://github.com/nttmdlab-nlp/VDocRAG.git`, never a local
vendored copy), `git clone`/`pull` this repo, run `app.py`.

**Step 8 (stretch)** — Benchmark tab / `eval/` scripts reproducing OpenDocVQA
metrics against a small subset.

---

## 8. Explicitly out of scope for this stage (unchanged)
- Any model training/fine-tuning — NTT's checkpoints only.
- PPTX ingestion.
- Hugging Face Spaces / ZeroGPU deployment, or any permanent public link.
- Extension work (reranking, backbone comparison, multi-hop reasoning, etc.).
- Multi-user / auth / production concerns — solo dev/test only.

---

## 9. Summary for the next session

Read Section 4 (resolved corrections) and Section 5 (remaining verification items)
before writing any code. The key mental model: **v1's open questions are now
closed on paper** — each has a decision and sourced reasoning, not just a flag —
but "closed on paper" is not the same as "verified," and Step 1 is where paper
reasoning meets actual hardware. Specifically:

- Use `attn_implementation="eager"`, not `sdpa` — this isn't a performance
  preference, `sdpa` will fail to load at all on this model.
- Share one base model, hot-swap LoRA adapters — decided, ~50ms/swap confirmed cheap.
- Never vendor NTT's code into this repo — install by URL every session; this is
  a license constraint, not a convenience choice.
- Instrument everything with `telemetry.py` from Step 1 onward — the page-cap
  formula and the VRAM-sharing decision both get *confirmed or corrected* by this
  data, not just logged for debugging's sake.
- Do not proceed past Step 1 if the smoke test fails, even with the corrected
  `eager` config — that's still the cheapest point to discover a hardware blocker,
  and every later step assumes it passed.
