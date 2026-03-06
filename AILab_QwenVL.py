# ComfyUI-QwenVL
# This custom node integrates the Qwen-VL series, including the latest Qwen3-VL models,
# including Qwen2.5-VL and the latest Qwen3-VL, to enable advanced multimodal AI for text generation,
# image understanding, and video analysis.
#
# Models License Notice:
# - Qwen3-VL: Apache-2.0 License (https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
# - Qwen2.5-VL: Apache-2.0 License (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import gc
import json
import platform
from enum import Enum
from pathlib import Path
import hashlib

import numpy as np
import psutil
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

# SageAttention support
try:
    from sageattention.core import (
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

# Global cache for generated prompts
PROMPT_CACHE = {}
CACHE_FILE = Path(__file__).parent / "prompt_cache.json"

# Simple global variable to store last generated prompt
LAST_SAVED_PROMPT = None

def load_prompt_cache():
    """Load prompt cache from file"""
    global PROMPT_CACHE
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                PROMPT_CACHE = json.load(f)
                print(f"[QwenVL] Loaded {len(PROMPT_CACHE)} cached prompts")
    except Exception as e:
        print(f"[QwenVL] Failed to load prompt cache: {e}")
        PROMPT_CACHE = {}

def save_prompt_cache():
    """Save prompt cache to file"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(PROMPT_CACHE, f, indent=2)
    except Exception as e:
        print(f"[QwenVL] Failed to save prompt cache: {e}")

def get_cache_key(model_name, preset_prompt, custom_prompt, image_hash=None, video_hash=None, seed=None):
    """Generate cache key from inputs"""
    key_data = {
        "model": model_name,
        "preset": preset_prompt,
        "custom": custom_prompt.strip() if custom_prompt else "",
        "image": image_hash,
        "video": video_hash,
        "seed": seed  # Always include seed to ensure proper caching behavior
    }
    # Create deterministic hash
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def get_alternative_cache_key(model_name, preset_prompt, custom_prompt, image_hash=None, video_hash=None, seed=None, module_name="QwenVL"):
    """Generate alternative cache key for fixed seed mode to find random prompts"""
    # Only for fixed seed mode (when user wants consistent prompts)
    # We consider any seed that the user keeps fixed as "fixed seed mode"
    
    print(f"[{module_name} DEBUG] Searching through cache for model={model_name}, preset={preset_prompt}")
    
    # Try to find any cached prompt with same model/preset/custom/image but different seed
    for cached_key, cached_data in PROMPT_CACHE.items():
        cached_model = cached_data.get("model")
        cached_preset = cached_data.get("preset") 
        cached_seed = cached_data.get("seed")
        
        print(f"[{module_name} DEBUG] Checking entry: model={cached_model}, preset={cached_preset}, seed={cached_seed}")
        
        if (cached_model == model_name and 
            cached_preset == preset_prompt and
            cached_seed != seed):  # Different seed
            
            # Generate the cache key that would have been created for this cached data
            # to check if image/video hashes match
            cached_image_hash = cached_data.get("image_hash")
            cached_video_hash = cached_data.get("video_hash")
            
            print(f"[{module_name} DEBUG] Found potential match with hashes: image={cached_image_hash}, video={cached_video_hash}")
            
            # If the cached data doesn't have hash info, try to match by other criteria
            if cached_image_hash is None and cached_video_hash is None:
                # Fallback: if both current and cached have no image/video, consider it a match
                if image_hash is None and video_hash is None:
                    print(f"[{module_name} DEBUG] Match found (no images/videos)!")
                    return cached_key
            else:
                # Match if hashes are the same (including None)
                if cached_image_hash == image_hash and cached_video_hash == video_hash:
                    print(f"[{module_name} DEBUG] Match found (hashes match)!")
                    return cached_key
    print(f"[{module_name} DEBUG] No alternative cache found")
    return None

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image with memory optimization"""
    if tensor is None:
        return None
    try:
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        # More aggressive memory management
        if tensor.is_floating_point():
            # Scale to 0-255 range more efficiently
            tensor = tensor.clamp(0, 1)
        
        # Reduce sample size for memory efficiency
        array = tensor.cpu().numpy()
        if tensor.numel() > 0:
            # Take smaller sample for hash to save memory
            sample_size = min(50, tensor.numel() // 4)  # Reduced from 100
            sample_pixels = array.flatten()[:sample_size].tolist() if array.size > 0 else []
        else:
            sample_pixels = []
        
        content = f"{tensor.shape}_{tensor.dtype}_{sample_pixels[:10]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except:
        return None

def get_image_hash(image):
    """Generate hash for image tensor"""
    if image is None:
        return None
    try:
        # Use image tensor properties for hash
        shape = str(image.shape)
        dtype = str(image.dtype)
        # Sample a few pixels for content hash (avoid full tensor for performance)
        if len(image.shape) >= 3:
            sample_pixels = image.flatten()[:100].tolist() if image.numel() > 0 else []
        else:
            sample_pixels = image.flatten().tolist() if image.numel() > 0 else []
        
        content = f"{shape}_{dtype}_{sample_pixels[:10]}"  # Limit sample size
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except:
        return None

def get_video_hash(video):
    """Generate hash for video tensor (same as image)"""
    return get_image_hash(video)

# Load cache on module import
load_prompt_cache()
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

# Export memory functions for external use
__all__ = ['PROMPT_CACHE', 'get_cache_key', 'get_alternative_cache_key', 'save_prompt_cache', 'get_image_hash', 'get_video_hash', 'check_pytorch_memory', 'set_pytorch_memory_fraction', 'get_device_info', 'tensor_to_pil', 'get_video_hash', 'enforce_memory', 'quantization_config', 'ensure_model', 'resolve_attention_mode', 'flash_attn_available', 'normalize_device_choice', 'load_model_configs', 'HF_VL_MODELS', 'HF_TEXT_MODELS', 'HF_ALL_MODELS', 'SYSTEM_PROMPTS', 'PRESET_PROMPTS', 'TOOLTIPS', 'Quantization', 'ATTENTION_MODES', 'NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

import folder_paths

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "hf_models.json"
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"
HF_VL_MODELS: dict[str, dict] = {}
HF_TEXT_MODELS: dict[str, dict] = {}
HF_ALL_MODELS: dict[str, dict] = {}
SYSTEM_PROMPTS = {}
PRESET_PROMPTS: list[str] = ["Describe this image in detail."]

TOOLTIPS = {
    "model_name": "Pick the Qwen-VL checkpoint. First run downloads weights into models/LLM/Qwen-VL, so leave disk space.",
    "quantization": "Precision vs VRAM. FP16 gives the best quality if memory allows; 8-bit suits 8–16 GB GPUs; 4-bit fits 6 GB or lower but is slower.",
    "attention_mode": "auto tries SageAttention → FlashAttention 2 → SDPA in order. SDPA is stable and recommended. Only override when debugging attention backends.",
    "preset_prompt": "Built-in instruction describing how Qwen-VL should analyze the media input.",
    "custom_prompt": "Additional user input that gets combined with the preset template. Leave empty to use only the template.",
    "max_tokens": "Maximum number of new tokens to decode. Larger values yield longer answers but consume more time and memory.",
    "keep_model_loaded": "Keeps the model resident in VRAM/RAM after the run so the next prompt skips loading.",
    "seed": "Seed controlling sampling and frame picking; reuse it to reproduce results.",
    "use_torch_compile": "Enable torch.compile('reduce-overhead') on supported CUDA/Torch 2.1+ builds for extra throughput after the first compile.",
    "device": "Choose where to run the model: auto, cpu, mps, or cuda:x for multi-GPU systems.",
    "temperature": "Sampling randomness when num_beams == 1. 0.2–0.4 is focused, 0.7+ is creative.",
    "top_p": "Nucleus sampling cutoff when num_beams == 1. Lower values keep only top tokens; 0.9–0.95 allows more variety.",
    "num_beams": "Beam-search width. Values >1 disable temperature/top_p and trade speed for more stable answers.",
    "repetition_penalty": "Values >1 (e.g., 1.1–1.3) penalize repeated phrases; 1.0 leaves logits untouched.",
    "frame_count": "Number of frames extracted from video inputs before prompting Qwen-VL. More frames provide context but cost time.",
}

class Quantization(str, Enum):
    Q4 = "4-bit (VRAM-friendly)"
    Q8 = "8-bit (Balanced)"
    FP16 = "None (FP16)"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Unsupported quantization: {value}")

ATTENTION_MODES = ["auto", "sage", "flash_attention_2", "sdpa"]

# Debug: Check SageAttention availability
print(f"[QwenVL Debug] SAGE_ATTENTION_AVAILABLE: {SAGE_ATTENTION_AVAILABLE}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"[QwenVL Debug] CUDA capability: {major}.{minor}")
else:
    print("[QwenVL Debug] CUDA not available")
print(f"[QwenVL Debug] Final ATTENTION_MODES: {ATTENTION_MODES}")

# Temporarily show sage option even if not available for testing
print("[QwenVL] NOTE: SageAttention option shown for testing. Install with: pip install sageattention")

def load_model_configs():
    global HF_VL_MODELS, HF_TEXT_MODELS, HF_ALL_MODELS, SYSTEM_PROMPTS, PRESET_PROMPTS
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        if "hf_vl_models" in data or "hf_text_models" in data:
            HF_VL_MODELS = data.get("hf_vl_models") or {}
            HF_TEXT_MODELS = data.get("hf_text_models") or {}
        else:
            HF_VL_MODELS = {k: v for k, v in data.items() if not k.startswith("_")}
            HF_TEXT_MODELS = {}
        SYSTEM_PROMPTS = data.get("_system_prompts", {})
        PRESET_PROMPTS = data.get("_preset_prompts", PRESET_PROMPTS)
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")
        HF_VL_MODELS = {}
        HF_TEXT_MODELS = {}
        HF_ALL_MODELS = {}
        SYSTEM_PROMPTS = {}
    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwenvl_prompts = data.get("qwenvl") or {}
        preset_override = data.get("_preset_prompts") or []
        if isinstance(qwenvl_prompts, dict) and qwenvl_prompts:
            SYSTEM_PROMPTS = qwenvl_prompts
        if isinstance(preset_override, list) and preset_override:
            PRESET_PROMPTS = preset_override
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] System prompts load failed: {exc}")
    custom = NODE_DIR / "custom_models.json"
    if custom.exists():
        try:
            with open(custom, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            custom_vl = data.get("hf_vl_models") or {}
            custom_text = data.get("hf_text_models") or {}
            legacy = data.get("hf_models", {}) or data.get("models", {})
            if isinstance(custom_vl, dict) and custom_vl:
                HF_VL_MODELS.update(custom_vl)
                print(f"[QwenVL] Loaded {len(custom_vl)} custom VL models")
            if isinstance(custom_text, dict) and custom_text:
                HF_TEXT_MODELS.update(custom_text)
                print(f"[QwenVL] Loaded {len(custom_text)} custom text models")
            if isinstance(legacy, dict) and legacy:
                HF_VL_MODELS.update(legacy)
                print(f"[QwenVL] Loaded {len(legacy)} custom legacy models")
        except Exception as exc:
            print(f"[QwenVL] custom_models.json skipped: {exc}")
    HF_ALL_MODELS = dict(HF_VL_MODELS)
    HF_ALL_MODELS.update(HF_TEXT_MODELS)

if not HF_ALL_MODELS:
    load_model_configs()

def check_pytorch_memory():
    """Check current PyTorch memory settings and allow user to set fraction"""
    try:
        import torch
        print(f"[QwenVL] PyTorch {torch.__version__}")
        print(f"[QwenVL] CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            current_fraction = torch.cuda.get_per_process_memory_fraction()
            print(f"[QwenVL] Current Memory Fraction: {current_fraction:.3f} ({current_fraction*100:.1f}% of GPU)")
            print(f"[QwenVL] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Allow user to set new fraction
            try:
                prompt = f"[QwenVL] Enter new memory fraction (0.1-0.9"
                if current_fraction is not None:
                    prompt += f", current={current_fraction:.3f})"
                new_fraction = float(input(prompt + ": ") or current_fraction)
                if 0.1 <= new_fraction <= 0.9:
                    torch.cuda.set_per_process_memory_fraction(new_fraction)
                    print(f"[QwenVL] ✅ Memory fraction set to: {new_fraction:.3f} ({new_fraction*100:.1f}%)")
                else:
                    print(f"[QwenVL] ❌ Invalid fraction. Must be between 0.1 and 0.9")
            except KeyboardInterrupt:
                print(f"[QwenVL] ✅ Keeping current fraction: {current_fraction:.3f}")
            except Exception as e:
                print(f"[QwenVL] ❌ Error setting fraction: {e}")
        else:
            print("[QwenVL] CUDA not available")
    except Exception as e:
        print(f"[QwenVL] Error checking PyTorch: {e}")

def set_pytorch_memory_fraction(fraction):
    """Set PyTorch memory fraction if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            if 0.1 <= fraction <= 0.9:
                torch.cuda.set_per_process_memory_fraction(fraction)
                print(f"[QwenVL] Memory fraction set to: {fraction:.3f} ({fraction*100:.1f}%)")
                return True
            else:
                print(f"[QwenVL] Invalid fraction: {fraction:.3f}. Must be between 0.1 and 0.9")
                return False
        else:
            print("[QwenVL] CUDA not available")
            return False
    except Exception as e:
        print(f"[QwenVL] Error: {e}")
        return False

def get_device_info():
    gpu = {"available": False, "total_memory": 0, "free_memory": 0}
    device_type = "cpu"
    recommended = "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = total - allocated - reserved
        
        gpu = {
            "available": True,
            "total_memory": total,
            "allocated_memory": allocated,
            "reserved_memory": reserved, 
            "free_memory": free,
        }
        device_type = "nvidia_gpu"
        recommended = "cuda"
        
        # Detailed memory debugging
        print(f"[QwenVL] GPU Memory Debug:")
        print(f"  Total VRAM: {total / 1024**3:.2f} GB")
        print(f"  Allocated: {allocated / 1024**3:.2f} GB")  
        print(f"  Reserved: {reserved / 1024**3:.2f} GB")
        print(f"  Free: {free / 1024**3:.2f} GB")
        print(f"  Model requires: 0.74 GB")
        print(f"  Available ratio: {(free / total) * 100:.1f}%")
        
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type = "apple_silicon"
        recommended = "mps"
        gpu = {"available": True, "total_memory": 0, "free_memory": 0}
    sys_mem = psutil.virtual_memory()
    return {
        "gpu": gpu,
        "system_memory": {
            "total": sys_mem.total / 1024**3,
            "available": sys_mem.available / 1024**3,
        },
        "device_type": device_type,
        "recommended_device": recommended,
    }

def normalize_device_choice(device: str) -> str:
    device = (device or "auto").strip()
    if device == "auto":
        return "auto"

    if device.isdigit():
        device = f"cuda:{int(device)}"

    if device == "cuda":
        if not torch.cuda.is_available():
            print("[QwenVL] CUDA requested but not available, falling back to CPU")
            return "cpu"
        return "cuda"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[QwenVL] CUDA requested but not available, falling back to CPU")
            return "cpu"
        if ":" in device:
            try:
                device_idx = int(device.split(":", 1)[1])
                if device_idx >= torch.cuda.device_count():
                    print(f"[QwenVL] CUDA device {device_idx} not available, using cuda:0")
                    return "cuda:0"
            except (ValueError, IndexError):
                print(f"[QwenVL] Invalid CUDA device format '{device}', using cuda:0")
                return "cuda:0"
        return device

    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            print("[QwenVL] MPS requested but not available, falling back to CPU")
            return "cpu"
        return "mps"

    return device

def flash_attn_available():
    if not torch.cuda.is_available():
        return False

    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False

    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False

    try:
        import importlib.metadata as importlib_metadata
        _ = importlib_metadata.version("flash_attn")
    except Exception:
        return False

    return True

def sage_attn_available():
    """Check if SageAttention is available and GPU supports it."""
    if not SAGE_ATTENTION_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    return True


def get_sage_attention_config():
    """Get the appropriate SageAttention kernel based on GPU architecture."""
    if not sage_attn_available():
        return None, None, None

    major, minor = torch.cuda.get_device_capability()
    arch_code = major * 10 + minor

    attn_func = None
    pv_accum_dtype = "fp32"

    if arch_code >= 120:  # Blackwell
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda
        print(f"[QwenVL] SageAttention: Using SM120 (Blackwell) FP8 kernel")
    elif arch_code >= 90:  # Hopper
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda_sm90
        print(f"[QwenVL] SageAttention: Using SM90 (Hopper) FP8 kernel")
    elif arch_code == 89:  # Ada Lovelace
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda
        print(f"[QwenVL] SageAttention: Using SM89 (Ada) FP8 kernel")
    elif arch_code >= 80:  # Ampere
        pv_accum_dtype = "fp32"
        attn_func = sageattn_qk_int8_pv_fp16_cuda
        print(f"[QwenVL] SageAttention: Using SM80+ (Ampere) FP16 kernel")
    else:
        print(f"[QwenVL] SageAttention not supported on SM{arch_code}")
        return None, None, None

    return attn_func, "per_warp", pv_accum_dtype

def is_fp8_model(model_name: str) -> bool:
    """Check if model name indicates it's a pre-quantized FP8 model."""
    fp8_indicators = ["-fp8", "_fp8", "-FP8", "_FP8"]
    return any(indicator in model_name for indicator in fp8_indicators)

def resolve_attention_mode(mode, force_sdpa=False):
    """Resolve attention mode with fallback logic.

    Args:
        mode: The requested attention mode
        force_sdpa: If True, always return SDPA (for FP8/BnB models)
    """
    if force_sdpa:
        return "sdpa"

    if mode == "sdpa":
        return "sdpa"
    if mode == "sage":
        if sage_attn_available():
            return "sage"
        print("[QwenVL] SageAttention forced but unavailable, falling back to SDPA")
        return "sdpa"
    if mode == "flash_attention_2":
        if flash_attn_available():
            return "flash_attention_2"
        print("[QwenVL] Flash-Attn forced but unavailable, falling back to SDPA")
        return "sdpa"

    # Auto mode: try sage → flash → sdpa
    if sage_attn_available():
        print("[QwenVL] Auto mode: Using SageAttention")
        return "sage"
    if flash_attn_available():
        print("[QwenVL] Auto mode: Using Flash Attention 2")
        return "flash_attention_2"
    print("[QwenVL] Auto mode: Using SDPA")
    return "sdpa"

def ensure_model(model_name):
    info = HF_ALL_MODELS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")
    repo_id = info["repo_id"]

    # Use ComfyUI's multi-path system if available
    llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    if llm_paths:
        models_dir = Path(llm_paths[0]) / "Qwen-VL"
    else:
        # Fallback to default behavior
        models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"

    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / repo_id.split("/")[-1]

    # ✅ If already downloaded (has weights), use local without calling snapshot_download
    if target.exists() and target.is_dir():
        if any(target.glob("*.safetensors")) or any(target.glob("*.bin")):
            return str(target)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        ignore_patterns=["*.md", ".git*"],
    )
    return str(target)

def enforce_memory(model_name, quantization, device_info):
    info = HF_ALL_MODELS.get(model_name, {})
    requirements = info.get("vram_requirement", {})
    mapping = {
        Quantization.Q4: requirements.get("4bit", 0),
        Quantization.Q8: requirements.get("8bit", 0),
        Quantization.FP16: requirements.get("full", 0),
    }
    needed = mapping.get(quantization, 0)
    if not needed:
        return quantization  # Always return quantization
    
    if device_info["recommended_device"] in {"cpu", "mps"}:
        needed *= 1.5
        available = device_info["system_memory"]["available"]
    else:
        available = device_info["gpu"]["free_memory"]
    
    # More conservative memory management
    if needed * 1.5 > available:  # More conservative threshold
        if quantization == Quantization.FP16:
            print("[QwenVL] ⚠️  Auto-switch to 8-bit due to VRAM pressure")
            return Quantization.Q8
        if quantization == Quantization.Q8:
            print("[QwenVL] ⚠️  Auto-switch to 4-bit due to VRAM pressure")
            return Quantization.Q4
        raise RuntimeError(f"Insufficient memory for {quantization.value} mode. Required: {needed * 1.5:.1f}GB, Available: {available / 1024**3:.1f}GB")
    
    # Conservative memory check with safety margin
    if needed * 1.3 > available:  # Reduced from 1.5 to 1.3
        if quantization == Quantization.FP16:
            print("[QwenVL] ⚠️  Auto-switch to 8-bit due to VRAM pressure")
            return Quantization.Q8
        if quantization == Quantization.Q8:
            print("[QwenVL] ⚠️  Auto-switch to 4-bit due to VRAM pressure")
            return Quantization.Q4
        print(f"[QwenVL] Memory pressure detected. Consider: 1) Use 4-bit quantization, 2) Reduce max_tokens below 1024, 3) Close other applications")
        return quantization  # Always return quantization

def quantization_config(model_name, quantization):
    info = HF_ALL_MODELS.get(model_name, {})
    if info.get("quantized"):
        return None, None
    if quantization == Quantization.Q4:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return cfg, None
    if quantization == Quantization.Q8:
        return BitsAndBytesConfig(load_in_8bit=True), None
    return None, torch.float16 if torch.cuda.is_available() else torch.float32

class QwenVLBase:
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        print(f"[QwenVL] Node on {self.device_info['device_type']}")

    def clear(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(
        self,
        model_name,
        quant_value,
        attention_mode,
        use_compile,
        device_choice,
        keep_model_loaded,
    ):
        quant = Quantization.from_value(quant_value)  # Skip enforce_memory for now
        
        # Safety check: ensure quant is not None
        if quant is None:
            print(f"[QwenVL] Invalid quantization value: {quant_value}, falling back to FP16")
            quant = Quantization.FP16
        
        # Check if BitsAndBytes quantization is being used
        is_bnb_quantization = quant in [Quantization.Q4, Quantization.Q8]
        
        # Check if this is a pre-quantized FP8 model
        is_prequantized_fp8 = is_fp8_model(model_name) or HF_ALL_MODELS.get(model_name, {}).get("quantized", False)
        
        # Determine if we need to force SDPA (for FP8 or BitsAndBytes models)
        force_sdpa = is_prequantized_fp8 or is_bnb_quantization
        
        # Resolve attention mode with force_sdpa flag
        attn_impl = resolve_attention_mode(attention_mode, force_sdpa=force_sdpa)
        
        # Additional info messages for forced SDPA
        if force_sdpa and attention_mode in ["auto", "sage", "flash_attention_2"]:
            if is_prequantized_fp8:
                print("[QwenVL] FP8 model detected - forcing SDPA attention")
            elif is_bnb_quantization:
                print("[QwenVL] BitsAndBytes quantization detected - forcing SDPA attention")
        
        print(f"[QwenVL] Attention backend selected: {attn_impl}")
        
        device_requested = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        device = normalize_device_choice(device_requested)
        signature = (model_name, quant.value, attn_impl, device, use_compile)
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            return
        self.clear()
        model_path = ensure_model(model_name)
        quant_config, dtype = quantization_config(model_name, quant)
        
        # Handle attention mode for loading
        # SageAttention requires loading with SDPA first, then patching
        actual_attn_impl = attn_impl
        if attn_impl == "sage":
            actual_attn_impl = "sdpa"
        
        # MEMORY DEBUGGING: Check memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[QwenVL] 📊 Memory BEFORE load - Allocated: {allocated_before:.1f}GB, Reserved: {reserved_before:.1f}GB, Total: {total_memory:.1f}GB")
            
            # Check what's using memory
            if allocated_before > 2.0:
                print(f"[QwenVL] ⚠️  WARNING: {allocated_before:.1f}GB already allocated before loading!")
        
        # DEBUG: Print quantization config details
        print(f"[QwenVL] 🔍 DEBUG - Model: {model_name}, Quant: {quant}, Quant config: {quant_config}")
        print(f"[QwenVL] 🔍 DEBUG - Device: {device}, Dtype: {dtype}")
        print(f"[QwenVL] 🔍 DEBUG - Attention impl: {actual_attn_impl}")
        
        # Quantization is back - enforce_memory was the problem
        
        load_kwargs = {
            "device_map": "cpu",  # Force CPU loading to test
            "dtype": torch.float16,  # Use dtype instead of deprecated torch_dtype
            "attn_implementation": actual_attn_impl,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
            
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
            
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        
        # Move to GPU manually if loading was on CPU
        if device != "cpu" and torch.cuda.is_available():
            print(f"[QwenVL] 🔄 Moving model from CPU to {device}...")
            try:
                self.model = self.model.to(device)
                print(f"[QwenVL] ✅ Model moved to {device}")
            except Exception as e:
                print(f"[QwenVL] ❌ Failed to move to GPU: {e}")
                print("[QwenVL] Keeping model on CPU (will be very slow)")
        
        # Apply SageAttention patching if needed
        if attn_impl == "sage":
            try:
                from sageattention_patch import set_sage_attention
                set_sage_attention(self.model)
                print("[QwenVL] SageAttention patching applied successfully")
            except Exception as e:
                print(f"[QwenVL] SageAttention patching failed: {e}")
                print("[QwenVL] Falling back to SDPA attention")
                # Model is already loaded with SDPA, so we can continue
        
        # MEMORY CLEANUP: Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[QwenVL] GPU memory after load: {torch.cuda.memory_allocated() / 1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
                
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        if use_compile and device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[QwenVL] torch.compile enabled")
            except Exception as exc:
                print(f"[QwenVL] torch.compile skipped: {exc}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.current_signature = signature

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @torch.no_grad()
    def generate(
        self,
        prompt_text,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
    ):
        # Memory optimization: clear cache before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        conversation = [{"role": "user", "content": []}]
        if image is not None:
            conversation[0]["content"].append({"type": "image", "image": self.tensor_to_pil(image)})
        if video is not None:
            frames = [self.tensor_to_pil(frame) for frame in video]
            if len(frames) > frame_count:
                idx = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
                frames = [frames[i] for i in idx]
            if frames:
                conversation[0]["content"].append({"type": "video", "video": frames})
        conversation[0]["content"].append({"type": "text", "text": prompt_text})
        
        # Optimize chat template for memory efficiency
        chat = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process images/videos more efficiently
        images = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
        video_frames = [frame for item in conversation[0]["content"] if item["type"] == "video" for frame in item["video"]]
        videos = [video_frames] if video_frames else None
        
        # Use smaller batch size for memory efficiency
        processed = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
        
        # Move to device more efficiently
        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)
        
        # Memory-efficient generation parameters
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False
            
        # Generate with memory monitoring
        try:
            outputs = self.model.generate(**model_inputs, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            # Clear memory and retry with reduced parameters
            print(f"[QwenVL] OOM detected, clearing memory and retrying...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Retry with smaller max_tokens
            reduced_tokens = max(256, max_tokens // 2)
            kwargs["max_new_tokens"] = reduced_tokens
            print(f"[QwenVL] Retrying with reduced tokens: {reduced_tokens}")
            
            try:
                outputs = self.model.generate(**model_inputs, **kwargs)
            except torch.cuda.OutOfMemoryError:
                print(f"[QwenVL] OOM still occurs with {reduced_tokens} tokens, falling back to CPU")
                # Fallback to CPU if GPU still OOM
                device_backup = model_inputs["input_ids"].device
                for key in model_inputs:
                    if torch.is_tensor(model_inputs[key]):
                        model_inputs[key] = model_inputs[key].cpu()
                outputs = self.model.generate(**model_inputs, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        input_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return text.strip()

    def run(self, model_name, quantization, preset_prompt, custom_prompt, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty, seed, keep_model_loaded, attention_mode, use_torch_compile, device, keep_last_prompt=False):
        torch.manual_seed(seed)
        
        global LAST_SAVED_PROMPT
        
        # Simple keep last prompt logic
        if keep_last_prompt:
            print(f"[QwenVL] Keep last prompt enabled - using last saved prompt")
            if LAST_SAVED_PROMPT:
                print(f"[QwenVL] Using last prompt: {LAST_SAVED_PROMPT[:50]}...")
                return (LAST_SAVED_PROMPT,)
            else:
                print(f"[QwenVL] No previous prompt found, returning empty")
                return ("",)
        
        # Always generate when keep last prompt is disabled
        print(f"[QwenVL] Keep last prompt disabled - generating new prompt")
        
        prompt_template = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        
        # Generate cache key with all inputs including seed
        image_hash = get_image_hash(image)
        video_hash = get_video_hash(video)
        cache_key = get_cache_key(model_name, preset_prompt, custom_prompt, image_hash, video_hash, seed)
        
        # Check cache first (only for random mode)
        if cache_key in PROMPT_CACHE:
            cached_text = PROMPT_CACHE[cache_key].get("text", "")
            if cached_text:
                print(f"[QwenVL] Using cached prompt for seed {seed}: {cache_key[:8]}...")
                return (cached_text,)
        
        if custom_prompt and custom_prompt.strip():
            # Combine user input with template - custom prompt first for priority
            prompt = f"{custom_prompt.strip()}\n\n{prompt_template}"
        else:
            prompt = prompt_template
            
        self.load_model(
            model_name,
            quantization,
            attention_mode,
            use_torch_compile,
            device,
            keep_model_loaded,
        )
        try:
            text = self.generate(
                prompt,
                image,
                video,
                frame_count,
                max_tokens,
                temperature,
                top_p,
                num_beams,
                repetition_penalty,
            )
            
            # Cache the generated text
            PROMPT_CACHE[cache_key] = {
                "text": text,
                "timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else None,
                "model": model_name,
                "preset": preset_prompt,
                "seed": seed,
                "image_hash": image_hash,
                "video_hash": video_hash
            }
            save_prompt_cache()  # Save cache to file
            
            print(f"[QwenVL] Cached new prompt for seed {seed}: {cache_key[:8]}...")
            
            # Save the generated prompt for future bypass mode
            LAST_SAVED_PROMPT = text
            print(f"[QwenVL] Saved prompt for bypass mode: {text[:50]}...")
            
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()

class AILab_QwenVL(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = PRESET_PROMPTS or ["Describe this image in detail."]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]
        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": TOOLTIPS["max_tokens"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"] + "\n\n💡 Cache Info: Prompts are cached automatically. Use the same inputs (model, preset, custom prompt, image/video) to reuse cached prompts and avoid regeneration.\n\n🔒 Fixed Seed Mode: Set seed = 1 to ignore image/video changes and only use text-based caching. Perfect for keeping the same prompt regardless of media input variations."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🔷 QwenVL-Mod/QwenVL"

    def process(self, model_name, quantization, preset_prompt, custom_prompt, attention_mode, max_tokens, keep_model_loaded, seed, keep_last_prompt=False, image=None, video=None):
        return self.run(model_name, quantization, preset_prompt, custom_prompt, image, video, 16, max_tokens, 0.6, 0.9, 1, 1.2, seed, keep_model_loaded, attention_mode, False, "auto", keep_last_prompt)

class AILab_QwenVL_Advanced(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = PRESET_PROMPTS or ["Describe this image in detail."]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["use_torch_compile"]}),
                "device": (device_options, {"default": "auto", "tooltip": TOOLTIPS["device"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 4096, "tooltip": TOOLTIPS["max_tokens"]}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "tooltip": TOOLTIPS["temperature"]}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": TOOLTIPS["top_p"]}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": TOOLTIPS["num_beams"]}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "tooltip": TOOLTIPS["repetition_penalty"]}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": TOOLTIPS["frame_count"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"] + "\n\n💡 Cache Info: Prompts are cached automatically. Use same inputs (model, preset, custom prompt, image/video) to reuse cached prompts and avoid regeneration.\n\n🔒 Fixed Seed Mode: Set seed = 1 to ignore image/video changes and only use text-based caching. Perfect for keeping the same prompt regardless of media input variations."}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep last generated prompt instead of creating a new one"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🔷 QwenVL-Mod/QwenVL"

    def process(self, model_name, quantization, attention_mode, use_torch_compile, device, preset_prompt, custom_prompt, max_tokens, temperature, top_p, num_beams, repetition_penalty, frame_count, keep_model_loaded, seed, keep_last_prompt, image=None, video=None):
        return self.run(model_name, quantization, preset_prompt, custom_prompt, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty, seed, keep_model_loaded, attention_mode, use_torch_compile, device, keep_last_prompt)

NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL": AILab_QwenVL,
    "AILab_QwenVL_Advanced": AILab_QwenVL_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL": "QwenVL-Mod",
    "AILab_QwenVL_Advanced": "QwenVL-Mod (Advanced)",
}
