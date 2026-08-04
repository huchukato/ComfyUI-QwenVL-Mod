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

from AILab_OutputCleaner import OutputCleanConfig, clean_model_output, prompt_output_guard

from AILab_QwenVL import (
    ATTENTION_MODES,
    HF_TEXT_MODELS,
    HF_VL_MODELS,
    PROMPT_CACHE,
    ensure_cuda_vram_headroom,
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
CUSTOM_ONLY_STYLE = "✍️ Custom Only (no preset)"

DEFAULT_STYLES = {
    "📝 Enhance": "Write one production-ready prompt paragraph in the same language as the user. Expand the idea with concrete subject, action, environment, lighting, camera, composition, color, texture, mood, and style details. Output only the final prompt paragraph.",
    "📝 Refine": "Write one polished prompt paragraph in the same language as the user. Preserve the core intent, remove redundancy and contradiction, and add useful visual specificity for subject, scene, lighting, camera perspective, composition, palette, texture, and atmosphere. Output only the final prompt paragraph.",
    "📝 Creative Rewrite": "Write one fresh, imaginative prompt paragraph in the same language as the user. Preserve the core intent while adding cohesive cinematic atmosphere, gesture, micro-details, color relationships, light interaction, materials, depth, and composition. Output only the final prompt paragraph.",
    "📝 Detailed Visual": "Write one highly detailed visual prompt paragraph in the same language as the user. Include subject traits, pose, expression, wardrobe, materials, foreground, midground, background, lighting source and direction, palette, contrast, lens feel, depth of field, framing, and final aesthetic style. Output only the final prompt paragraph.",
    "📝 Artistic Style": "Write one artistic prompt paragraph in the same language as the user. Build a coherent visual direction with mood, palette, shape language, contrast, material feel, camera perspective, composition rhythm, and fitting style references. Output only the final prompt paragraph.",
    "📝 Technical Specs": "Write one clear technical photography or cinematography prompt paragraph in the same language as the user. Include camera distance, angle, lens feel, aperture or depth of field, focus target, lighting type and direction, color temperature, contrast, framing, background separation, texture rendering, and final image style. Output only the final prompt paragraph.",
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
PROMPT_STYLES = {CUSTOM_ONLY_STYLE: "", **PROMPT_STYLES}


class AILab_QwenVL_PromptEnhancer(QwenVLBase):
    STYLES = PROMPT_STYLES

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ENHANCED_OUTPUT",)
    FUNCTION = "process"
    CATEGORY = "QwenVL-Mod/QwenVL"

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
                "max_tokens": ("INT", {"default": 1024, "min": 32, "max": 16384}),
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

        is_custom_only = enhancement_style == CUSTOM_ONLY_STYLE
        style_instruction = "" if is_custom_only else self.STYLES.get(
            enhancement_style,
            next(iter(self.STYLES.values()), ""),
        ).strip()
        custom_instruction = custom_system_prompt.strip()
        base_instruction = "\n\n".join(part for part in (custom_instruction, style_instruction) if part)
        if not base_instruction and is_custom_only:
            raise ValueError("custom_system_prompt is required when using Custom Only (no preset).")
        base_instruction = "\n\n".join(part for part in (base_instruction, prompt_output_guard()) if part)
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

        # BnB needs a GPU and cannot tolerate a post-load .to(device).
        if quant_cfg is not None and device == "cpu":
            print("[QwenVL] ⚠️  BitsAndBytes requires a CUDA/ROCm GPU — falling back to FP32 on CPU")
            quant_cfg = None

        signature = (repo_id, quantization, device)
        if self.text_model is not None and self.text_signature == signature:
            ensure_cuda_vram_headroom("QwenVL PromptEnhancer HF", min_free_gb=1.0, min_free_ratio=0.08)
            return

        self.text_model = None
        self.text_tokenizer = None
        self.text_signature = None

        load_kwargs = {"trust_remote_code": True}
        if quant_cfg is not None:
            load_kwargs["quantization_config"] = quant_cfg
            # accelerate must dispatch BnB weights straight to the target GPU.
            load_kwargs["device_map"] = device if device.startswith("cuda") else "auto"
        else:
            load_kwargs["dtype"] = torch.float16 if device == "cuda" else torch.float32

        print(f"[QwenVL] Loading text model {model_name} ({quantization})")
        self.text_tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        self.text_model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs).eval()
        # Only move non-BnB models — BnB is already dispatched via device_map.
        if quant_cfg is None:
            self.text_model.to(device)
        ensure_cuda_vram_headroom("QwenVL PromptEnhancer HF", min_free_gb=1.0, min_free_ratio=0.08)
        # Detect architecture from loaded model config
        hf_model_type = getattr(self.text_model.config, "model_type", None)
        self.is_qwen35 = hf_model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_vl") if hf_model_type else "qwen3.5-" in model_name.lower()
        if self.is_qwen35:
            print(f"[QwenVL] Qwen3.5 detected (model_type={hf_model_type}): Will disable thinking in chat template.")
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
        ensure_cuda_vram_headroom("QwenVL PromptEnhancer HF", min_free_gb=1.0, min_free_ratio=0.08)

        if device == "auto":
            device_choice = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        else:
            device_choice = device

        messages = [{"role": "user", "content": prompt}]
        is_qwen35 = getattr(self, "is_qwen35", False)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}

        # Inject the disable thinking kwargs for HF Transformers correctly
        if is_qwen35:
            template_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            formatted_prompt = self.text_tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception:
            # Fallback to raw prompt if the tokenizer lacks a chat template
            formatted_prompt = prompt

        inputs = self.text_tokenizer(formatted_prompt, return_tensors="pt").to(device_choice)
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "eos_token_id": self.text_tokenizer.eos_token_id,
            "pad_token_id": self.text_tokenizer.eos_token_id,
        }

        # Optional: Apply seed for generation reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        outputs = self.text_model.generate(**inputs, **kwargs)

        # Strip out the input tokens to get just the generated response
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        result = self.text_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        result = clean_model_output(result, OutputCleanConfig(mode="prompt")) or result

        # Cache the generated text
        # PROMPT_CACHE[cache_key] = {
        #     "text": result,
        #     "timestamp": None,  # PromptEnhancer doesn't have CUDA events
        #     "model": model_name,
        #     "preset": style,
        #     "seed": seed,
        #     "image_hash": None,  # PromptEnhancer doesn't use images
        #     "video_hash": None   # PromptEnhancer doesn't use videos
        # }
        # save_prompt_cache()  # Save cache to file

        # print(f"[QwenVL PromptEnhancer HF] Cached new prompt for seed {seed}: {cache_key[:8]}...")

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
