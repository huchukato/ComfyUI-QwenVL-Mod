# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import gc
import hashlib
import json
import platform
from enum import Enum
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from AILab_QwenVL import (
    ATTENTION_MODES,
    HF_TEXT_MODELS,
    HF_VL_MODELS,
    PROMPT_CACHE,
    get_cache_key,
    get_alternative_cache_key,
    save_prompt_cache,
    QwenVLBase,
    Quantization,
    TOOLTIPS,
)

# Simple global variable to store last generated prompt
LAST_SAVED_PROMPT = None

NODE_DIR = Path(__file__).parent
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"

DEFAULT_STYLES = {
    "📝 Enhance": "Expand and enrich this prompt with vivid visual context:",
    "📝 Refine": "Polish this prompt for clarity and precise AI interpretation:",
    "📝 Creative Rewrite": "Rewrite this prompt imaginatively while preserving intent:",
    "📝 Detailed Visual": "Turn this prompt into a highly detailed visual description:",
    "📝 Artistic Style": "Describe this prompt in artistic language suitable for image generation:",
    "📝 Technical Specs": "Convert this prompt into clear technical parameters:",
}


def _load_prompt_styles() -> dict[str, str]:
    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwen_text = data.get("qwen_text") or {}
        styles = qwen_text.get("styles") or {}
        if isinstance(styles, dict) and styles:
            resolved = {
                name: entry.get("system_prompt", "")
                for name, entry in styles.items()
                if isinstance(entry, dict) and entry.get("system_prompt")
            }
            if resolved:
                return resolved
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] Prompt style load failed: {exc}")
    return DEFAULT_STYLES


PROMPT_STYLES = _load_prompt_styles()


class AILab_QwenVL_PromptEnhancer(QwenVLBase):
    # Custom icon for QwenVL-Mod
    ICON = "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"
    STYLES = PROMPT_STYLES

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ENHANCED_OUTPUT",)
    FUNCTION = "process"
    CATEGORY = "🔷QwenVL-Mod/QwenVL"

    def __init__(self):
        super().__init__()
        self.text_model = None
        self.text_tokenizer = None
        self.text_signature = None

    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_TEXT_MODELS.keys()) + [name for name in HF_VL_MODELS.keys() if name not in HF_TEXT_MODELS]
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        styles = list(cls.STYLES.keys())
        preferred_style = "📝 Enhance"
        default_style = preferred_style if preferred_style in styles else (styles[0] if styles else "📝 Enhance")
        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["use_torch_compile"]}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": TOOLTIPS["device"]}),
                "prompt_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt text to enhance. Leave blank to just emit the preset instruction."}),
                "enhancement_style": (styles, {"default": default_style}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 256, "min": 32, "max": 1024}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep the last generated prompt instead of creating a new one"}),
            }
        }

    def process(
        self,
        model_name,
        quantization,
        attention_mode,
        use_torch_compile,
        device,
        prompt_text,
        enhancement_style,
        custom_system_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        keep_last_prompt=False,
    ):
        global LAST_SAVED_PROMPT
        
        # Simple keep last prompt logic
        if keep_last_prompt:
            print(f"[QwenVL PromptEnhancer HF] Keep last prompt enabled - using last saved prompt")
            if LAST_SAVED_PROMPT:
                print(f"[QwenVL PromptEnhancer HF] Using last prompt: {LAST_SAVED_PROMPT[:50]}...")
                return (LAST_SAVED_PROMPT,)
            else:
                print(f"[QwenVL PromptEnhancer HF] No previous prompt found, returning empty")
                return ("",)
        
        # Always generate when keep last prompt is disabled
        print(f"[QwenVL PromptEnhancer HF] Keep last prompt disabled - generating new prompt")
        
        base_instruction = custom_system_prompt.strip() or self.STYLES.get(
            enhancement_style,
            next(iter(self.STYLES.values()), ""),
        )
        user_prompt = prompt_text.strip() or "Describe a scene vividly."
        merged_prompt = f"{user_prompt}\n\n{base_instruction}".strip()
        if model_name in HF_TEXT_MODELS:
            enhanced = self._invoke_text(
                model_name,
                quantization,
                device,
                merged_prompt,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty,
                keep_model_loaded,
                seed,
            )
        else:
            enhanced = self._invoke_qwen(
                model_name,
                quantization,
                attention_mode,
                use_torch_compile,
                device,
                merged_prompt,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty,
                keep_model_loaded,
                seed,
            )
        
        # Save the generated prompt for future bypass mode
        LAST_SAVED_PROMPT = enhanced.strip()
        print(f"[QwenVL PromptEnhancer HF] Saved prompt for bypass mode: {LAST_SAVED_PROMPT[:50]}...")
        
        return (enhanced.strip(),)

    def _invoke_qwen(
        self,
        model_name,
        quantization,
        attention_mode,
        use_torch_compile,
        device,
        prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
    ):
        output = self.run(
            model_name=model_name,
            quantization=quantization,
            preset_prompt="🪄 Prompt Refine & Expand",
            custom_prompt=prompt,
            image=None,
            video=None,
            frame_count=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            repetition_penalty=repetition_penalty,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            attention_mode=attention_mode,
            use_torch_compile=use_torch_compile,
            device=device,
        )
        return output[0]

    def _load_text_model(self, model_name, quantization, device_choice):
        info = HF_TEXT_MODELS.get(model_name, {})
        repo_id = info.get("repo_id")
        if not repo_id:
            raise ValueError(f"[QwenVL] Missing repo_id for text model: {model_name}")

        if device_choice == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        else:
            device = device_choice

        if quantization == Quantization.Q4:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == Quantization.Q8:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_cfg = None

        signature = (repo_id, quantization, device)
        if self.text_model is not None and self.text_signature == signature:
            return

        self.text_model = None
        self.text_tokenizer = None
        self.text_signature = None

        load_kwargs = {}
        if quant_cfg:
            load_kwargs["quantization_config"] = quant_cfg
        else:
            load_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

        print(f"[QwenVL] Loading text model {model_name} ({quantization})")
        self.text_tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        self.text_model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True, **load_kwargs).eval()
        self.text_model.to(device)
        self.text_signature = signature

    def _invoke_text(
        self,
        model_name,
        quantization,
        device,
        prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
    ):
        self._load_text_model(model_name, quantization, device)

        if device == "auto":
            device_choice = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        else:
            device_choice = device

        inputs = self.text_tokenizer(prompt, return_tensors="pt").to(device_choice)
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "eos_token_id": self.text_tokenizer.eos_token_id,
            "pad_token_id": self.text_tokenizer.eos_token_id,
        }
        outputs = self.text_model.generate(**inputs, **kwargs)
        decoded = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        prefix = self.text_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        result = decoded[len(prefix) :].strip()
        
        # Cache the generated text
        PROMPT_CACHE[cache_key] = {
            "text": result,
            "timestamp": None,  # PromptEnhancer doesn't have CUDA events
            "model": model_name,
            "preset": style,
            "seed": seed,
            "image_hash": None,  # PromptEnhancer doesn't use images
            "video_hash": None   # PromptEnhancer doesn't use videos
        }
        save_prompt_cache()  # Save cache to file
        
        print(f"[QwenVL PromptEnhancer HF] Cached new prompt for seed {seed}: {cache_key[:8]}...")

        if not keep_model_loaded:
            self.text_model = None
            self.text_tokenizer = None
            self.text_signature = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result

NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_PromptEnhancer": AILab_QwenVL_PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_PromptEnhancer": "QwenVL-Mod Prompt Enhancer",
}
