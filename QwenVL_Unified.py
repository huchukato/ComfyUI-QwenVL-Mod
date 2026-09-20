# ComfyUI-QwenVL-Mod — unified HF/GGUF vision-language node.
# One node selects between Transformers (HF) and llama.cpp (GGUF) backends via a
# backend dropdown and a prefixed model list. Old separate nodes are kept for
# backward compatibility.

import torch

import AILab_QwenVL as _hf
import AILab_QwenVL_GGUF as _gguf


HF_VL_MODELS = getattr(_hf, "HF_VL_MODELS", {})
GGUF_VL_CATALOG = getattr(_gguf, "GGUF_VL_CATALOG", {})
PRESET_PROMPTS = getattr(_hf, "PRESET_PROMPTS") or ["Describe this image in detail."]
Quantization = _hf.Quantization
ATTENTION_MODES = _hf.ATTENTION_MODES
CAMERA_TAG_OPTIONS = _hf.CAMERA_TAG_OPTIONS
CAMERA_TAG_TOOLTIP = _hf.CAMERA_TAG_TOOLTIP
TOOLTIPS = _hf.TOOLTIPS


HF_PREFIX = "HF: "
GGUF_PREFIX = "GGUF: "


def _combined_model_list():
    hf_models = sorted(HF_VL_MODELS.keys())
    gguf_all = GGUF_VL_CATALOG.get("models") or {}
    gguf_models = sorted([k for k, e in gguf_all.items() if (e or {}).get("mmproj_filename")])
    models = []
    if hf_models:
        models.extend([f"{HF_PREFIX}{m}" for m in hf_models])
    if gguf_models:
        models.extend([f"{GGUF_PREFIX}{m}" for m in gguf_models])
    if not models:
        models = ["(no models found)"]
    return models


def _default_model():
    models = _combined_model_list()
    for m in models:
        if m.startswith(GGUF_PREFIX):
            return m
    return models[0]


class QwenVL_Unified:
    """Single node that runs a Qwen-VL model on HF Transformers or GGUF."""

    def __init__(self):
        self._hf = _hf.AILab_QwenVL_Advanced()
        self._gguf = _gguf.AILab_QwenVL_GGUF_Advanced()

    @classmethod
    def INPUT_TYPES(cls):
        prompts = PRESET_PROMPTS or ["Describe this image in detail."]
        preferred = "🖼️ Detailed Description"
        default_prompt = preferred if preferred in prompts else prompts[0]
        return {
            "required": {
                "backend": (["HF Transformers", "GGUF llama.cpp"], {"default": "GGUF llama.cpp", "tooltip": "Backend engine. The model_name prefix determines actual routing; keep them aligned."}),
                "model_name": (_combined_model_list(), {"default": _default_model(), "tooltip": "HF models are prefixed with 'HF: ', GGUF models with 'GGUF: '."}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS.get("preset_prompt", "")}),
                "camera_tag": (CAMERA_TAG_OPTIONS, {"default": "None", "tooltip": CAMERA_TAG_TOOLTIP}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS.get("custom_prompt", "")}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 8192, "tooltip": TOOLTIPS.get("max_tokens", "")}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS.get("keep_model_loaded", "")}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS.get("seed", "")}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep the last generated prompt instead of creating a new one"}),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "Skip Qwen model loading and return custom_prompt directly."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "First reference image (single image). For R2VA this is Picture 1."}),
                "image2": ("IMAGE", {"tooltip": "Second reference image (single image). For R2VA this is Picture 2."}),
                "video": ("IMAGE", {"tooltip": "Video frames input. Use frame_count to control how many frames are sampled."}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": TOOLTIPS.get("frame_count", "")}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "QwenVL-Mod"

    def process(
        self,
        backend,
        model_name,
        preset_prompt,
        camera_tag,
        custom_prompt,
        max_tokens,
        keep_model_loaded,
        seed,
        keep_last_prompt,
        passthrough=False,
        image=None,
        image2=None,
        video=None,
        frame_count=16,
    ):
        if model_name.startswith(GGUF_PREFIX):
            m = model_name[len(GGUF_PREFIX):]
            return self._gguf.process(
                model_name=m,
                device="auto",
                preset_prompt=preset_prompt,
                camera_tag=camera_tag,
                custom_prompt=custom_prompt,
                max_tokens=max_tokens,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.0,
                frame_count=frame_count,
                ctx=32768,
                n_batch=512,
                gpu_layers=-1,
                image_max_tokens=4096,
                top_k=20,
                pool_size=4194304,
                keep_model_loaded=keep_model_loaded,
                seed=seed,
                keep_last_prompt=keep_last_prompt,
                passthrough=passthrough,
                image=image,
                image2=image2,
                video=video,
            )

        if model_name.startswith(HF_PREFIX):
            m = model_name[len(HF_PREFIX):]
        else:
            m = model_name
        return self._hf.process(
            model_name=m,
            quantization=Quantization.FP16.value,
            attention_mode="auto",
            use_torch_compile=False,
            device="auto",
            preset_prompt=preset_prompt,
            camera_tag=camera_tag,
            custom_prompt=custom_prompt,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            num_beams=1,
            repetition_penalty=1.0,
            keep_model_loaded=keep_model_loaded,
            seed=seed,
            keep_last_prompt=keep_last_prompt,
            passthrough=passthrough,
            image=image,
            image2=image2,
            video=video,
            frame_count=frame_count,
        )


class QwenVL_Unified_Advanced(QwenVL_Unified):
    """Unified node exposing all backend-specific knobs."""

    @classmethod
    def INPUT_TYPES(cls):
        prompts = PRESET_PROMPTS or ["Describe this image in detail."]
        preferred = "🖼️ Detailed Description"
        default_prompt = preferred if preferred in prompts else prompts[0]
        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list
        return {
            "required": {
                "backend": (["HF Transformers", "GGUF llama.cpp"], {"default": "GGUF llama.cpp"}),
                "model_name": (_combined_model_list(), {"default": _default_model(), "tooltip": "HF models are prefixed with 'HF: ', GGUF models with 'GGUF: '."}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS.get("preset_prompt", "")}),
                "camera_tag": (CAMERA_TAG_OPTIONS, {"default": "None", "tooltip": CAMERA_TAG_TOOLTIP}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS.get("custom_prompt", "")}),
                "device": (device_options, {"default": "auto", "tooltip": TOOLTIPS.get("device", "")}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 8192, "tooltip": TOOLTIPS.get("max_tokens", "")}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS.get("keep_model_loaded", "")}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS.get("seed", "")}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep the last generated prompt instead of creating a new one"}),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "Skip Qwen model loading and return custom_prompt directly."}),
                # HF-specific
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS.get("quantization", "")}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS.get("attention_mode", "")}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS.get("use_torch_compile", "")}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": TOOLTIPS.get("num_beams", "")}),
                # GGUF-specific
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": TOOLTIPS.get("frame_count", "")}),
                "ctx": ("INT", {"default": 32768, "min": 1024, "max": 262144, "step": 512}),
                "n_batch": ("INT", {"default": 512, "min": 64, "max": 32768, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
                "image_max_tokens": ("INT", {"default": 4096, "min": 256, "max": 1024000, "step": 256}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "First reference image (single image). For R2VA this is Picture 1."}),
                "image2": ("IMAGE", {"tooltip": "Second reference image (single image). For R2VA this is Picture 2."}),
                "video": ("IMAGE", {"tooltip": "Video frames input. Use frame_count to control how many frames are sampled."}),
            },
        }

    def process(
        self,
        backend,
        model_name,
        preset_prompt,
        camera_tag,
        custom_prompt,
        device,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        keep_model_loaded,
        seed,
        keep_last_prompt,
        passthrough,
        quantization,
        attention_mode,
        use_torch_compile,
        num_beams,
        frame_count,
        ctx,
        n_batch,
        gpu_layers,
        image_max_tokens,
        top_k,
        pool_size,
        image=None,
        image2=None,
        video=None,
    ):
        if model_name.startswith(GGUF_PREFIX):
            m = model_name[len(GGUF_PREFIX):]
            return self._gguf.process(
                model_name=m,
                device=device,
                preset_prompt=preset_prompt,
                camera_tag=camera_tag,
                custom_prompt=custom_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                frame_count=frame_count,
                ctx=ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                image_max_tokens=image_max_tokens,
                top_k=top_k,
                pool_size=pool_size,
                keep_model_loaded=keep_model_loaded,
                seed=seed,
                keep_last_prompt=keep_last_prompt,
                passthrough=passthrough,
                image=image,
                image2=image2,
                video=video,
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
            preset_prompt=preset_prompt,
            camera_tag=camera_tag,
            custom_prompt=custom_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            keep_model_loaded=keep_model_loaded,
            seed=seed,
            keep_last_prompt=keep_last_prompt,
            passthrough=passthrough,
            image=image,
            image2=image2,
            video=video,
            frame_count=frame_count,
        )


NODE_CLASS_MAPPINGS = {
    "QwenVL_Unified": QwenVL_Unified,
    "QwenVL_Unified_Advanced": QwenVL_Unified_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL_Unified": "🧠 QwenVL Unified (HF / GGUF)",
    "QwenVL_Unified_Advanced": "🧠 QwenVL Unified Advanced (HF / GGUF)",
}
