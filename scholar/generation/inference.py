"""Phi-3-mini-4k-instruct inference."""
from __future__ import annotations

import re
from typing import Optional

import torch

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


class Phi3Generator:
    _instance: Optional["Phi3Generator"] = None

    def __init__(self, load_adapter: bool = True) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from scholar.config import settings

        # trust_remote_code=False: use transformers' built-in Phi-3 implementation,
        # which is updated for the 5.x DynamicCache API. The cached modeling_phi3.py
        # from HuggingFace Hub still references removed attributes (seen_tokens, etc.).
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=str(settings.hf_home),
            trust_remote_code=False,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=False,
                cache_dir=str(settings.hf_home),
                attn_implementation="eager",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=False,
                cache_dir=str(settings.hf_home),
                attn_implementation="eager",
            )

        if (
            load_adapter
            and settings.lora_adapter_path is not None
            and settings.lora_adapter_path.exists()
        ):
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(  # type: ignore[assignment]  # PeftModel compat
                    self.model,
                    str(settings.lora_adapter_path),
                )
                print(f"Loaded LoRA adapter from {settings.lora_adapter_path}")
            except Exception as e:
                print(f"Could not load LoRA adapter: {e}")

        self.model.eval()

    def generate(
        self,
        title: str,
        abstract: str,
        why: str,
        level: str = "grad",
    ) -> dict:
        from scholar.generation.prompts import get_prompt

        prompt = get_prompt(level=level, title=title, abstract=abstract, why=why)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(  # type: ignore[misc]  # PEFT model generate typing
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return _parse_output(str(generated))

    @classmethod
    def get_instance(cls, load_adapter: bool = True) -> "Phi3Generator":
        if cls._instance is None:
            cls._instance = cls(load_adapter=load_adapter)
        return cls._instance


def _parse_output(text: str) -> dict:
    summary_match = re.search(r"SUMMARY:\s*(.+?)(?=GLOSSARY:|WHY:|$)", text, re.DOTALL)
    glossary_match = re.search(r"GLOSSARY:\s*(.+?)(?=WHY:|$)", text, re.DOTALL)
    why_match = re.search(r"WHY:\s*(.+?)$", text, re.DOTALL)

    summary = summary_match.group(1).strip() if summary_match else text[:300].strip()

    glossary: list[str] = []
    if glossary_match:
        raw_glossary = glossary_match.group(1).strip()
        for line in raw_glossary.splitlines():
            line = line.strip()
            if line.startswith("-"):
                glossary.append(line[1:].strip())
            elif line:
                glossary.append(line)

    why = why_match.group(1).strip() if why_match else ""

    return {
        "summary": summary,
        "glossary": glossary,
        "why": why,
    }
