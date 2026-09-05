# Licenses

## Phi-3-vision-128k-instruct — MIT
Confirmed directly from Microsoft's model card. No restrictions relevant to this
project (use, modification, redistribution all permitted).

## NTT VDocRAG (github.com/nttmdlab-nlp/VDocRAG) — evaluation-only, NOT MIT/Apache

**Status: CONFIRMED.** Read directly from the actual `LICENSE` file in NTT's repo
(cloned and read verbatim, not inferred from a sibling repo). It's the "SOFTWARE
LICENSE AGREEMENT FOR EVALUATION," identical to what was found circumstantially
in earlier sibling-repo research — that inference turned out correct, and is now
backed by the literal source instead of precedent.

Key terms:
- **§1 (grant)**: use "internally for the purposes of testing, analyzing, and
  evaluating the methods or mechanisms as shown in the research paper." A
  course-project reproduction/evaluation is inside this scope.
- **§4(b)(i)**: no selling, assigning, leasing, **distributing**, or transferring
  the Software to any third party; no copying/reproducing except as allowed.
- **§4(b)(iv)**: no modifying, disassembling, decompiling, reverse engineering,
  or translating the Software.

### What this means for how this repo is built
1. **NTT's package is never committed to this repo.** It's installed fresh every
   Colab session via `pip install git+https://github.com/nttmdlab-nlp/VDocRAG.git`
   (see `requirements-colab.txt`). Publishing a copy of their source in this
   (potentially public) repo would plausibly count as "distribute... to any third
   party" under §4(b)(i) the moment anyone else can view or clone it.
2. **We don't patch their source files.** If a compatibility fix is ever needed
   (e.g. if `VDocRetriever.load()` turns out to hardcode
   `attn_implementation="flash_attention_2"` rather than accepting it as a
   pass-through kwarg — unconfirmed, see below), the fix belongs in *our* wrapper
   code (subclassing, monkey-patching a class attribute from our own module), not
   as an edit to their shipped `.py` files, per §4(b)(iv).
3. **No weights or their released checkpoints are committed either** — pulled
   from Hugging Face each session, cached to Drive-backed `HF_HOME`.

### `attn_implementation` pass-through — CONFIRMED, no longer an open item
Read directly from `src/vdocrag/vdocretriever/modeling/vdocretriever.py` and the
`vdocgenerator` equivalent: `load()`'s signature is
`load(cls, model_name_or_path, pooling='cls', normalize=False, lora_name_or_path=None, **hf_kwargs)`,
and `**hf_kwargs` is forwarded straight to `cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)`.
`attn_implementation` and `quantization_config` pass through cleanly. Their own
`test.py` confirms this in practice — it calls `.load()` with
`attn_implementation="flash_attention_2"` directly. No wrapper-level workaround
needed for this.

### A more significant finding from reading the source: `load()` merges LoRA
`VDocRetriever.load()` / `VDocGenerator.load()`, when given `lora_name_or_path`,
internally do:
```python
lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
lora_model = lora_model.merge_and_unload()
```
This **collapses the LoRA delta into the base weights and discards the PEFT
wrapper**. Calling `.load()` independently for the retriever and generator (as
NTT's own `test.py` does — sequentially, reusing one variable name, never both
resident at once) produces two fully separate merged models with nothing left to
hot-swap. This directly affects the shared-base-model decision from the
handoff doc's Section 4.4 — see `vdocrag/model_manager.py`'s module docstring for
the resolution (a `share_base_model` config flag supporting both the VRAM-optimal
shared/hot-swap path, built by calling `VDocRetriever(encoder=...)` /
`VDocGenerator(decoder=...)` directly rather than through `.load()`, and a
simpler fallback that matches their tested `.load()` path exactly).

Building the shared path this way is legitimate under §4(b)(iv) (no modification)
— it calls their public `__init__` constructors with a manually-attached,
unmerged `PeftModel`, rather than editing any of their shipped `.py` files.
