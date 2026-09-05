"""
Loads the Phi-3-vision backbone + both LoRA adapters (retriever, generator).

This module exists because reading NTT's actual source
(github.com/nttmdlab-nlp/VDocRAG, src/vdocrag/vdocretriever/modeling/vdocretriever.py
and the vdocgenerator equivalent) surfaced something the original design didn't
account for: `VDocRetriever.load()` / `VDocGenerator.load()`, when given
`lora_name_or_path`, call `PeftModel.from_pretrained(...).merge_and_unload()`
internally — the LoRA delta is merged into the base weights and the PEFT
wrapper is discarded. Calling `.load()` for both models independently, as
NTT's own test.py / README quickstart does, produces two fully separate merged
model instances with no shared base and nothing left to hot-swap.

This means the shared-base + adapter-hot-swap design (handoff doc Section 4.4)
is NOT something NTT's public `.load()` API gives you — it has to be built by
calling their public `VDocRetriever(encoder=...)` / `VDocGenerator(decoder=...)`
constructors directly with a manually-attached, unmerged PeftModel instead.
That's legitimate use of their public classes (not a modification of their
shipped files — see docs/licenses.md's §4(b)(iv) note), but it is a deviation
from their tested code path, and PEFT's multi-adapter API has never been
validated by NTT against this specific trust_remote_code model class.

So this module supports BOTH modes behind one interface:
  - "shared": one base model, both adapters attached via PeftModel, hot-swapped
    with set_adapter() before each call. VRAM-optimal (~2.3GB saved), but
    unproven on this custom model class until Step 1's smoke test Check 2
    confirms it.
  - "independent": calls NTT's own `.load()` for each model exactly as their
    README does. Simpler, uses their tested path, costs ~2.3GB more VRAM.

Default is "shared"; flip SHARE_BASE_MODEL to False if Step 1 testing shows
hot-swap doesn't behave correctly on this model class, with no other code
changes required — retriever.py / generator.py only ever call this module's
public methods, never touch the mode-specific loading logic directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from vdocrag_app.telemetry import log_call, logger

# torch is imported lazily inside methods that actually load models (setup(),
# vram_report()) rather than at module level. This keeps ModelManagerConfig
# and ModelManager importable in environments without a working CUDA torch
# install -- which matters for testing everything that depends on this
# module (retriever.py, generator.py, app.py) without needing a GPU present.

MODEL_ID = "microsoft/Phi-3-vision-128k-instruct"
RETRIEVER_ADAPTER = "NTT-hil-insight/VDocRetriever-Phi3-vision"
GENERATOR_ADAPTER = "NTT-hil-insight/VDocGenerator-Phi3-vision"


def _ensure_transformers_compat_shims() -> None:
    """NTT's vdocretriever.py / vdocgenerator.py import `AutoModelForVision2Seq`
    at module level, unconditionally. That symbol is a long-deprecated alias
    for `AutoModelForImageTextToText` and has been removed from `transformers`
    ahead of the officially documented v5.0 cutoff (confirmed empirically in
    Step 1: `ImportError: cannot import name 'AutoModelForVision2Seq'` even on
    a pinned 4.x release). NTT's own retriever/generator actually load via
    `AutoModelForCausalLM` (TRANSFORMER_CLS in their source) -- the missing
    symbol is dead code in their file, never actually called -- so providing
    a working alias unblocks the import without touching their shipped files
    (legitimate under the license's no-modification clause: this patches what
    our own code sees when importing `transformers`, not their source)."""
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText
        logger.info("Shimmed AutoModelForVision2Seq -> AutoModelForImageTextToText")


ShareMode = Literal["shared", "independent"]


@dataclass
class ModelManagerConfig:
    share_base_model: bool = True
    attn_implementation: str = "eager"  # NOT "sdpa" -- see handoff doc Section 4.3.
    # NTT's own test.py/README use flash_attention_2 (they ran on A100s); we use
    # eager because flash-attn has no Turing/T4 support and this model's custom
    # modeling code explicitly declares _supports_sdpa = False.
    torch_dtype_name: str = "bfloat16"  # resolved to an actual torch.dtype lazily
    # via `resolved_dtype` below -- kept as a string here so this dataclass
    # (and everything that imports it) stays importable without torch present.
    load_in_4bit: bool = True

    @property
    def resolved_dtype(self):
        import torch

        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            self.torch_dtype_name
        ]


class ModelManager:
    """Owns the base model, both adapters, and the processor. retriever.py and
    generator.py hold a reference to this and call `use_retriever()` /
    `use_generator()` before their model calls -- a no-op in "independent" mode,
    an adapter swap in "shared" mode."""

    def __init__(self, config: Optional[ModelManagerConfig] = None):
        self.config = config or ModelManagerConfig()
        self.processor = None
        self._retriever_model = None  # VDocRetriever instance
        self._generator_model = None  # VDocGenerator instance
        self._peft_model = None  # only set in "shared" mode -- the single shared PeftModel
        self._active_adapter: Optional[str] = None
        self._loaded = False

    @property
    def mode(self) -> ShareMode:
        return "shared" if self.config.share_base_model else "independent"

    def _bnb_config(self):
        from transformers import BitsAndBytesConfig

        if not self.config.load_in_4bit:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.config.resolved_dtype,
        )

    @log_call("model_manager")
    def setup(self) -> None:
        """Loads everything. Call once per session. Idempotent -- calling
        again after a successful setup is a no-op, so notebook cells can be
        re-run without reloading multi-GB models by accident."""
        if self._loaded:
            logger.info("ModelManager.setup() called again -- already loaded, skipping.")
            return

        _ensure_transformers_compat_shims()

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        if self.mode == "shared":
            self._setup_shared()
        else:
            self._setup_independent()

        self._loaded = True
        logger.info(f"ModelManager ready in '{self.mode}' mode.")

    def _load_config(self):
        """Builds the model config with attn_implementation set as a direct
        attribute rather than passed as a from_pretrained() kwarg. Confirmed
        necessary empirically in Step 1 (see handoff doc Section 4.3 update):
        passing attn_implementation="eager" as a plain from_pretrained() kwarg
        raised `ValueError: Phi3VForCausalLM does not support Flash Attention 2
        yet` even though flash attention was never requested -- something in
        this environment's attention-implementation validation path mishandles
        the kwarg-based route for this custom trust_remote_code model. Setting
        it directly on the config object before from_pretrained() bypasses
        whatever that path does differently and is confirmed working."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        config._attn_implementation = self.config.attn_implementation
        return config

    @log_call("model_manager")
    def _setup_shared(self) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from vdocrag.vdocgenerator.modeling import VDocGenerator
        from vdocrag.vdocretriever.modeling import VDocRetriever

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            config=self._load_config(),
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype=self.config.resolved_dtype,
            quantization_config=self._bnb_config(),
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0  # matches VDocRetriever/.load()'s own default

        peft_model = PeftModel.from_pretrained(base_model, RETRIEVER_ADAPTER, adapter_name="retriever")
        peft_model.load_adapter(GENERATOR_ADAPTER, adapter_name="generator")
        self._peft_model = peft_model
        self._active_adapter = "retriever"  # PeftModel.from_pretrained activates the first adapter by default

        # VDocRetriever / VDocGenerator are thin nn.Module wrappers around an
        # encoder/decoder they don't own exclusively -- both can legitimately
        # wrap the SAME peft_model object. Switching peft_model.set_adapter(...)
        # before a call changes behavior for whichever wrapper is used next,
        # since both hold a reference to the same underlying module.
        self._retriever_model = VDocRetriever(encoder=peft_model, pooling="eos", normalize=True)
        self._generator_model = VDocGenerator(decoder=peft_model)

    @log_call("model_manager")
    def _setup_independent(self) -> None:
        """Matches NTT's own test.py / README exactly: two separate `.load()`
        calls, each internally merging its LoRA adapter into its own base
        model copy. No adapter switching needed or possible -- each model is
        already specialized."""
        from vdocrag.vdocgenerator.modeling import VDocGenerator
        from vdocrag.vdocretriever.modeling import VDocRetriever

        common_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=self.config.resolved_dtype,
            quantization_config=self._bnb_config(),
        )
        self._retriever_model = VDocRetriever.load(
            MODEL_ID, lora_name_or_path=RETRIEVER_ADAPTER, pooling="eos", normalize=True,
            config=self._load_config(), **common_kwargs
        ).to("cuda:0")
        self._generator_model = VDocGenerator.load(
            MODEL_ID, lora_name_or_path=GENERATOR_ADAPTER,
            config=self._load_config(), **common_kwargs
        ).to("cuda:0")

    def use_retriever(self):
        """Call before any retriever forward pass. Swaps the active LoRA
        adapter in "shared" mode (~50ms, per the PEFT hot-swap reference
        measurement in the handoff doc); no-op in "independent" mode."""
        self._require_loaded()
        if self.mode == "shared" and self._active_adapter != "retriever":
            self._peft_model.set_adapter("retriever")
            self._active_adapter = "retriever"
        return self._retriever_model

    def use_generator(self):
        """Call before any generator forward pass. See use_retriever()."""
        self._require_loaded()
        if self.mode == "shared" and self._active_adapter != "generator":
            self._peft_model.set_adapter("generator")
            self._active_adapter = "generator"
        return self._generator_model

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError("ModelManager.setup() must be called before use_retriever()/use_generator().")

    def vram_report(self) -> dict:
        """Current + peak VRAM, for logging into the telemetry stream after
        setup -- this is the number that confirms or corrects the ~2.3GB/copy
        estimate in the handoff doc's Section 4.4."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {"vram_allocated_gb": 0.0, "vram_peak_gb": 0.0}
            return {
                "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            }
        except Exception:
            return {"vram_allocated_gb": 0.0, "vram_peak_gb": 0.0}
