import os
import threading
import time
import json
import sys
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TQDM_MININTERVAL", "0.5")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("TQDM_DISABLE", "0")

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel
from PIL import Image


JSON_START = "<JSON>"
JSON_END = "</JSON>"

@dataclass
class ModelConfig:
    model_id: str = os.getenv("MODEL_ID", "google/gemma-3-27b-it")
    adapter_dir: str = os.getenv("ADAPTER_DIR", "")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    local_model_path: str = os.getenv("GEMMA_MODEL_PATH", "")

    # single-GPU setup
    device_map: str = os.getenv("DEVICE_MAP", "cuda")  # <- key change
    load_in_4bit: bool = os.getenv("LOAD_IN_4BIT", "0") == "1"



CONFIG = ModelConfig()

class Heartbeat:
    def __init__(self, label: str, interval: int = 10):
        self.label = label
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        start = time.time()
        while not self._stop_event.is_set():
            elapsed = int(time.time() - start)
            print(
                f"[Heartbeat] {self.label}... elapsed={elapsed}s",
                flush=True,
            )
            self._stop_event.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1)



class Gemma3Generator:
    def __init__(self, config: ModelConfig = CONFIG) -> None:
        self.config = config

        print(f"[Gemma3Generator] PID={os.getpid()} initializing...", flush=True)

        base_model_id = self.config.local_model_path or self.config.model_id
        use_local_only = bool(self.config.local_model_path)

        print(f"[Gemma3Generator] Base model: {base_model_id}", flush=True)
        print(f"[Gemma3Generator] adapter_dir: {self.config.adapter_dir or '(none)'}", flush=True)
        print(f"[Gemma3Generator] local_files_only: {use_local_only}", flush=True)
        print(f"[Gemma3Generator] device_map: {self.config.device_map}", flush=True)
        print(f"[Gemma3Generator] load_in_4bit: {self.config.load_in_4bit}", flush=True)

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        print("[Gemma3Generator] Loading processor...", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            self.config.adapter_dir or base_model_id,
            use_fast=False,
            local_files_only=use_local_only,
            token=hf_token,
        )

        print("[Gemma3Generator] Processor loaded.", flush=True)


        hb = Heartbeat("Loading Gemma3 model weights", interval=15)
        hb.start()

        try:
            model_kwargs = dict(
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                local_files_only=use_local_only,
                low_cpu_mem_usage=True,
                device_map=self.config.device_map if torch.cuda.is_available() else None,
                token=hf_token,
            )

            if self.config.load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                print("[Gemma3Generator] Loading model in 4-bit mode using bitsandbytes...", flush=True)

            self.base_model = Gemma3ForConditionalGeneration.from_pretrained(
                base_model_id,
                **model_kwargs,
            )

        finally:
            hb.stop()

        print("[Gemma3Generator] Model weights loaded.", flush=True)

        self.model = self.base_model
        self.model.eval()

        print("[Gemma3Generator] Model ready.", flush=True)

    @staticmethod
    def _load_images(
        image_paths: Optional[List[str]],
    ) -> Optional[Union[Image.Image, List[Image.Image]]]:
        if not image_paths:
            return None
        pil_images: List[Image.Image] = [Image.open(p).convert("RGB") for p in image_paths]
        return pil_images[0] if len(pil_images) == 1 else pil_images

    def _build_inputs(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        images = self._load_images(image_paths)
        text_prompt = prompt

        if images is not None:
            boi_token = getattr(self.processor, "boi_token", None)
            if boi_token is not None:
                num_images = 1 if isinstance(images, Image.Image) else len(images)
                existing_tokens = text_prompt.count(boi_token)
                if num_images > 0:
                    if existing_tokens == 0:
                        injected = " ".join([boi_token] * num_images)
                        text_prompt = f"{injected}\n{text_prompt}"
                    elif existing_tokens != num_images:
                        print(
                            f"[Gemma3Generator] Warning: prompt has {existing_tokens} image token(s) "
                            f"but {num_images} image(s) were provided; ignoring images."
                        )
                        images = None

        if images is not None:
            inputs = self.processor(
                text=text_prompt,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor(
                text=text_prompt,
                return_tensors="pt",
                padding=True,
            )


        if torch.cuda.is_available():
            try:
                first_param_device = next(self.model.parameters()).device
                inputs = {k: (v.to(first_param_device) if isinstance(v, torch.Tensor) else v)
                          for k, v in inputs.items()}
            except StopIteration:
                pass

        return inputs
    
    @staticmethod
    def extract_first_json_block(raw: str) -> str:
        if raw is None:
            raise ValueError("No text provided")

        s = raw.strip()

        
        s = s.replace("<end_of_turn>", "").strip()

        if "```" in s:
            parts = s.split("```")

            for i in range(1, len(parts), 2):
                chunk = parts[i]
                # drop optional language header (e.g. "json\n")
                if "\n" in chunk:
                    first_line, rest = chunk.split("\n", 1)
                    if first_line.strip().lower() in {"json", "application/json"}:
                        chunk = rest
                chunk = chunk.strip()

                if chunk.startswith("{") or chunk.startswith("["):
                    s = chunk
                    break


        start = None
        for i, ch in enumerate(s):
            if ch in "{[":
                start = i
                break
        if start is None:
            raise ValueError("No JSON start found")

        opening = s[start]
        closing = "}" if opening == "{" else "]"


        depth = 0
        in_str = False
        esc = False

        for j in range(start, len(s)):
            ch = s[j]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            
            if ch == '"':
                in_str = True
                continue

            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return s[start : j + 1]

 
            if opening == "{":
                if ch == "[":                
                    pass
            else:
                if ch == "{":
                    pass

        raise ValueError("Unterminated JSON (no matching closing brace/bracket)")

    def generate(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
        skip_special_tokens: bool = True,
        strip_prompt: bool = True,
        strip_json_markers: bool = True,) -> str:
        inputs = self._build_inputs(prompt, image_paths=image_paths)
        effective_max_new_tokens = max_new_tokens or self.config.max_new_tokens

        tok = self.processor.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=effective_max_new_tokens,
                pad_token_id=pad_id,
            )

        if strip_prompt:
            input_len = inputs["input_ids"].shape[1]
            gen_only_ids = generated_ids[:, input_len:]
        else:
            gen_only_ids = generated_ids

        text = self.processor.batch_decode(
            gen_only_ids,
            skip_special_tokens=skip_special_tokens,
        )[0]

        try:
            json_str = self.extract_first_json_block(text)
            return json_str   # or return json.loads(json_str) if you want a dict
        except ValueError:
            return text  # fallback if model didn't produce JSON at all


    
