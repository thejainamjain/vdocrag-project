"""
Bi-encoder retriever wrapper.

Design goal: same interface works for
  (a) Phase 1 baseline  -> NTT's pretrained VDocRetriever (Phi-3-vision + LoRA)
  (b) Phase 2 from-scratch -> our own LoRA fine-tune on Qwen2-VL-2B-Instruct

Both a query and a document PAGE IMAGE are encoded by the *same* underlying
LVLM (shared weights), producing a single dense vector each (EOS-token pooling,
L2-normalized), scored with cosine similarity.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetrieverOutput:
    q_reps: Optional[torch.Tensor] = None   # (batch, dim) query embeddings
    p_reps: Optional[torch.Tensor] = None   # (batch, dim) passage/image embeddings
    loss: Optional[torch.Tensor] = None


class VDocRetriever(nn.Module):
    """
    Wraps a causal LVLM (Phi-3-vision or Qwen2-VL) as a bi-encoder.

    Query encoding: text-only forward pass -> pool last non-pad token hidden state.
    Document encoding: image + short prompt forward pass -> same pooling.
    """

    def __init__(self, base_model: nn.Module, pooling: str = "eos", normalize: bool = True,
                 temperature: float = 0.01):
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature

    @staticmethod
    def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool the representation at the last non-padded token (EOS-style pooling)."""
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), seq_lengths]

    def _encode(self, inputs: dict) -> torch.Tensor:
        if inputs is None:
            return None
        outputs = self.base_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        reps = self._last_token_pool(last_hidden, inputs["attention_mask"])
        if self.normalize:
            reps = F.normalize(reps, p=2, dim=-1)
        return reps

    def encode_query(self, query_inputs: dict) -> torch.Tensor:
        return self._encode(query_inputs)

    def encode_document(self, document_inputs: dict) -> torch.Tensor:
        return self._encode(document_inputs)

    def forward(self, query: Optional[dict] = None, document: Optional[dict] = None,
                **kwargs) -> RetrieverOutput:
        q_reps = self.encode_query(query) if query is not None else None
        p_reps = self.encode_document(document) if document is not None else None

        loss = None
        if q_reps is not None and p_reps is not None and self.training:
            # In-batch negatives contrastive loss (InfoNCE), matching VDocRAG's setup.
            scores = torch.matmul(q_reps, p_reps.transpose(0, 1)) / self.temperature
            labels = torch.arange(scores.size(0), device=scores.device)
            loss = F.cross_entropy(scores, labels)

        return RetrieverOutput(q_reps=q_reps, p_reps=p_reps, loss=loss)

    @classmethod
    def load(cls, model_name_or_path: str, lora_name_or_path: Optional[str] = None,
             use_lora: bool = True, pooling: str = "eos", normalize: bool = True,
             torch_dtype=torch.bfloat16, **kwargs):
        """
        Loads a backbone + optional LoRA adapter.

        NOTE: import of transformers/peft kept local so this file can be imported
        (e.g. for unit tests) even in environments without GPU deps installed.
        """
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True, **kwargs
        )

        if use_lora and lora_name_or_path is not None:
            from peft import PeftModel
            base_model = PeftModel.from_pretrained(base_model, lora_name_or_path)

        return cls(base_model=base_model, pooling=pooling, normalize=normalize)
