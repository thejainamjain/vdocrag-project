# NTT VDocRAG — confirmed API reference

Everything below was read directly from `github.com/nttmdlab-nlp/VDocRAG`
(`src/vdocrag/vdocretriever/modeling/vdocretriever.py`,
`src/vdocrag/vdocgenerator/modeling/vdocgenerator.py`, `test.py`, `README.md`,
`src/vdocrag/vdocretriever/arguments.py`) — not reconstructed from the paper's
prose, not inferred from a sibling model. Where our wrapper code
(`vdocrag_app/retriever.py`, `vdocrag_app/generator.py`, `vdocrag_app/model_manager.py`)
depends on one of these facts, it's cited back to here.

## `VDocRetriever`

```python
class VDocRetriever(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    @classmethod
    def load(cls, model_name_or_path, pooling='cls', normalize=False,
             lora_name_or_path=None, **hf_kwargs):
        ...
```

- `**hf_kwargs` is forwarded straight to `AutoModelForCausalLM.from_pretrained()` —
  `attn_implementation`, `quantization_config`, `torch_dtype`, `trust_remote_code`
  all pass through cleanly.
- If `lora_name_or_path` is given, `load()` calls `PeftModel.from_pretrained(...)`
  then **`.merge_and_unload()`** — the LoRA delta is merged into the base weights,
  the `PeftModel` wrapper is discarded. Calling `load()` this way produces a plain
  merged model, not something you can hot-swap adapters on afterward.
- Forward signature: `forward(query=None, document=None, pair=None, use_cache=True)`
  → `EncoderOutput(q_reps, p_reps, loss, scores)`. `query`/`document` are dicts of
  tokenized+processed tensors (`input_ids`, `attention_mask`, and for documents
  also `pixel_values`, `image_sizes`).
- Pooling: `_pooling()` supports `'cls'/'first'`, `'mean'/'avg'/'average'`,
  `'last'/'eos'`. NTT's actual released checkpoints use `pooling='eos'` — this is
  explicit in their own `test.py`/README, not the class's own default (`'cls'`).
  **Passing the wrong pooling mode won't error — it'll silently produce degraded,
  wrong-shaped-but-valid-looking embeddings.** Always pass `pooling='eos'`
  explicitly when loading their retriever checkpoint.
- `normalize=True` (also explicit in their example, not the class default of
  `False`) — L2-normalizes both query and document representations, which is
  what makes `IndexFlatIP` (inner product) equivalent to cosine similarity in
  `vdocrag_app/index.py`.

## `VDocGenerator`

```python
class VDocGenerator(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def generate(self, input, generation_args, use_cache=True):
        return self.decoder.generate(**input, **generation_args, use_cache=use_cache)
```

Same `.load()` / `merge_and_unload()` behavior as the retriever. No pooling/
normalize args — it's a plain decoder wrapper, `.generate()` just forwards to
the underlying HF `.generate()`.

## Confirmed prompt templates (verbatim from `test.py` / README)

**Query prompt** (retrieval):
```
Instruct: I'm looking for an image that answers the question.
Query: {question}</s>
```
(`vdocrag_app/retriever.py`'s `QUERY_PROMPT_TEMPLATE` / `build_query_prompt()`)

**Document prompt** (retrieval):
```
<|image_1|>
What is shown in this image?</s>
```
(`vdocrag_app/retriever.py`'s `DOC_PROMPT`)

**Generation prompt**: built via `processor.tokenizer.apply_chat_template()` on
a single user message: `f"{image_tokens}\n{question}\n Answer briefly."`, where
`image_tokens` is `"<|image_1|>\n<|image_2|>\n..."` for however many images are
being passed. (`vdocrag_app/generator.py`'s `build_chat_prompt()` / `build_image_tokens()`)

**Generation args** (matches their example exactly):
```python
{"max_new_tokens": 64, "temperature": 0.0, "do_sample": False, "eos_token_id": processor.tokenizer.eos_token_id}
```

**Document image size**: every page image is resized to exactly `(1344, 1344)`
before processing — this is the size their checkpoints were fine-tuned against,
not a free parameter (`vdocrag_app/retriever.py`'s `prepare_doc_image()`).

**Processor max_length**: `256` for queries, `4096` for documents — matches
their `query_inputs`/`doc_inputs` processor calls exactly.

## Their own reference numbers (flash_attention_2, A100, bf16, no quantization)

From `test.py`'s comments — useful as a sanity check in the Step 1 smoke test,
not as an exact target under our `eager` + 4-bit config (see
`notebooks/00_smoke_test.ipynb` Check 3 for why ordering matters more than
exact values here):

- Retrieval similarities for the two example queries against the two example
  images: `[0.515625, 0.38476562]` and `[0.37890625, 0.5703125]`.
- Generation example answer: `"28.69m"` (to "How many international visitors
  came to Japan in 2017?").

## LoRA adapter config (from `arguments.py` defaults)
`r=8`, `alpha=64`, `dropout=0.1`, target modules
`q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj` — matches the VRAM
estimate in the handoff doc's Section 4.4 (tens of MB per adapter, not a
meaningful factor in the shared-vs-independent VRAM decision).
