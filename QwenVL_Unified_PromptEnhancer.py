# ComfyUI-QwenVL-Mod — unified HF/GGUF prompt enhancer node.
# One node selects between Transformers (HF) and llama.cpp (GGUF) backends via a
# backend dropdown and a prefixed model list. Old separate nodes are kept for
# backward compatibility.

import AILab_QwenVL as _hf_base
import AILab_QwenVL_PromptEnhancer as _hf
from qwenvl_presets import TEXT_STYLE_NAMES, DURATION_OPTIONS, DEFAULT_DURATION

try:
    import AILab_QwenVL_GGUF_PromptEnhancer as _gguf
except Exception as _gguf_err:
    _gguf = None
    print(f"[QwenVL] GGUF prompt enhancer unavailable: {_gguf_err}")

HF_TEXT_MODELS = getattr(_hf_base, "HF_TEXT_MODELS", {})
HF_VL_MODELS = getattr(_hf_base, "HF_VL_MODELS", {})
Quantization = _hf_base.Quantization
ATTENTION_MODES = _hf_base.ATTENTION_MODES
CAMERA_TAG_OPTIONS = _hf_base.CAMERA_TAG_OPTIONS
CAMERA_TAG_TOOLTIP = _hf_base.CAMERA_TAG_TOOLTIP
STYLE_TAG_OPTIONS = _hf_base.STYLE_TAG_OPTIONS
STYLE_TAG_TOOLTIP = _hf_base.STYLE_TAG_TOOLTIP
TOOLTIPS = _hf_base.TOOLTIPS
PROMPT_STYLES = getattr(_hf, "PROMPT_STYLES", {})

HF_PREFIX = "HF: "
GGUF_PREFIX = "GGUF: "


def _hf_model_list():
    return list(HF_TEXT_MODELS.keys()) + [n for n in HF_VL_MODELS.keys() if n not in HF_TEXT_MODELS]


def _gguf_model_list():
    if _gguf is None:
        return []
    catalog = _gguf.AILab_QwenVL_GGUF_PromptEnhancer.load_gguf_models()
    return sorted(list((catalog.get("models") or {}).keys()))


def _combined_model_list():
    models = [f"{GGUF_PREFIX}{m}" for m in _gguf_model_list()]
    models.extend(f"{HF_PREFIX}{m}" for m in _hf_model_list())
    return models or ["(no models found)"]


def _default_model():
    models = _combined_model_list()
    for m in models:
        if m.startswith(GGUF_PREFIX):
            return m
    return models[0]


class QwenVL_Unified_PromptEnhancer:
    """Single prompt enhancer node backed by HF Transformers or GGUF llama.cpp."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ENHANCED_OUTPUT",)
    FUNCTION = "process"
    CATEGORY = "🔮 QwenVL-Mod"

    def __init__(self):
        self._hf = _hf.AILab_QwenVL_PromptEnhancer()
        self._gguf = _gguf.AILab_QwenVL_GGUF_PromptEnhancer() if _gguf else None

    @classmethod
    def INPUT_TYPES(cls):
        styles = list(TEXT_STYLE_NAMES) or list(PROMPT_STYLES.keys())
        preferred_style = "Enhance"
        default_style = preferred_style if preferred_style in styles else (styles[0] if styles else "Enhance")
        gguf_note = "" if _gguf else " (GGUF models hidden — llama-cpp-python not installed)"
        return {
            "required": {
                "model_name": (_combined_model_list(), {"default": _default_model(), "tooltip": f"HF models are prefixed with 'HF: ', GGUF models with 'GGUF: '. The prefix selects the backend.{gguf_note}"}),
                "prompt_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt text to enhance. Leave blank to just emit the preset instruction."}),
                "enhancement_style": (styles, {"default": default_style}),
                "camera_tag": (CAMERA_TAG_OPTIONS, {"default": "None", "tooltip": CAMERA_TAG_TOOLTIP}),
                "style_tag": (STYLE_TAG_OPTIONS, {"default": "None", "tooltip": STYLE_TAG_TOOLTIP}),
                "max_tokens": ("INT", {"default": 8192, "min": 32, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "english_output": ("BOOLEAN", {"default": False, "tooltip": "Force final output in English using translation prompt (GGUF backend only)."}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": TOOLTIPS.get("device", "")}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS.get("quantization", "") + " (HF backend only)"}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS.get("attention_mode", "") + " (HF backend only)"}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS.get("use_torch_compile", "") + " (HF backend only)"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep model loaded in memory for faster repeated inference (uses more VRAM)."}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep the last generated prompt instead of creating a new one"}),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "Skip Qwen model loading and return prompt_text directly. Use when the chat already generated the final prompt — saves VRAM and inference time."}),
                "duration": (DURATION_OPTIONS, {"default": DEFAULT_DURATION, "tooltip": "Clip length for duration-aware styles (MiniMax/LTX/Wan). Ignored by generic styles."}),
            }
        }

    def process(
        self,
        model_name,
        prompt_text,
        enhancement_style,
        camera_tag,
        style_tag,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        english_output,
        device,
        quantization,
        attention_mode,
        use_torch_compile,
        keep_model_loaded,
        seed,
        keep_last_prompt=False,
        passthrough=False,
        duration=DEFAULT_DURATION,
    ):
        if model_name.startswith(GGUF_PREFIX):
            if self._gguf is None:
                raise ImportError(
                    "GGUF prompt enhancer unavailable — llama-cpp-python is not "
                    "installed. Local install: `pip install llama-cpp-python`."
                )
            return self._gguf.process(
                model_name=model_name[len(GGUF_PREFIX):],
                prompt_text=prompt_text,
                preset_system_prompt=enhancement_style,
                camera_tag=camera_tag,
                style_tag=style_tag,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                english_output=english_output,
                device=device,
                keep_model_loaded=keep_model_loaded,
                seed=seed,
                keep_last_prompt=keep_last_prompt,
                passthrough=passthrough,
                duration=duration,
            )

        if model_name.startswith(HF_PREFIX):
            m = model_name[len(HF_PREFIX):]
        else:
            m = model_name
        return self._hf.process(
            model_name=m,
            quantization=quantization,
            attention_mode=attention_mode,
            use_torch_compile=use_torch_compile,
            device=device,
            prompt_text=prompt_text,
            enhancement_style=enhancement_style,
            camera_tag=camera_tag,
            style_tag=style_tag,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            keep_model_loaded=keep_model_loaded,
            seed=seed,
            keep_last_prompt=keep_last_prompt,
            passthrough=passthrough,
            duration=duration,
        )


NODE_CLASS_MAPPINGS = {
    "QwenVL_Unified_PromptEnhancer": QwenVL_Unified_PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL_Unified_PromptEnhancer": "✍🏻 QwenVL Unified Prompt Enhancer (HF / GGUF)",
}
