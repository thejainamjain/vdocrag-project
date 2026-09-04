"""
Generator wrapper: takes retrieved page image(s) + question, generates the
answer directly (no text extraction). Same underlying LVLM class as the
retriever, but used for standard causal-LM generation instead of embedding.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image


class VDocGenerator(nn.Module):
    def __init__(self, base_model: nn.Module, processor):
        super().__init__()
        self.base_model = base_model
        self.processor = processor

    def build_prompt(self, question: str, num_images: int) -> str:
        image_tokens = "\n".join([f"<|image_{i + 1}|>" for i in range(num_images)])
        messages = [{"role": "user", "content": f"{image_tokens}\n{question}"}]
        return self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def answer(self, question: str, images: List[Image.Image], max_new_tokens: int = 64) -> str:
        prompt = self.build_prompt(question, len(images))
        inputs = self.processor(prompt, images=images, return_tensors="pt").to(self.base_model.device)

        generate_ids = self.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return response

    @classmethod
    def load(cls, model_name_or_path: str, lora_name_or_path: Optional[str] = None,
              use_lora: bool = True, torch_dtype=torch.bfloat16, **kwargs):
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True, **kwargs
        )

        if use_lora and lora_name_or_path is not None:
            from peft import PeftModel
            base_model = PeftModel.from_pretrained(base_model, lora_name_or_path)

        return cls(base_model=base_model, processor=processor)
