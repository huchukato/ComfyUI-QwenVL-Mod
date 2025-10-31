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

import torch
import time
import json
import platform
import psutil
import numpy as np
from PIL import Image
from enum import Enum
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
import folder_paths
import gc

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "config.json"
MODEL_CONFIGS = {}
SYSTEM_PROMPTS = {}

def load_model_configs():
    global MODEL_CONFIGS, SYSTEM_PROMPTS
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            MODEL_CONFIGS = json.load(f)
            SYSTEM_PROMPTS = MODEL_CONFIGS.get("_system_prompts", {})
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {CONFIG_PATH}")
        MODEL_CONFIGS, SYSTEM_PROMPTS = {}, {}
    except json.JSONDecodeError:
        print(f"Error: Failed to parse configuration file.")
        MODEL_CONFIGS, SYSTEM_PROMPTS = {}, {}

    # --- user-defined custom models  ---
    custom_path = NODE_DIR / "custom_models.json"
    if custom_path.exists():
        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                custom_data = json.load(f) or {}

            user_models = custom_data.get("hf_models", {}) or custom_data.get("models", {})

            if user_models:
                MODEL_CONFIGS.update(user_models)
                print(f"[QwenVL] ✅ Loaded {len(user_models)} user-defined models from custom_models.json.")
            else:
                print("[QwenVL] ⚠️ custom_models.json found but no valid model entries.")
        except Exception as e:
            print(f"[QwenVL] ⚠️ Failed to load custom_models.json → {e}")
    else:
        print("[QwenVL] ℹ️ No custom_models.json found, skipping user-defined models.")

if not MODEL_CONFIGS:
    load_model_configs()

class Quantization(str, Enum):
    Q4_BIT = "4-bit (VRAM-friendly)"
    Q8_BIT = "8-bit (Balanced)"
    NONE = "None (FP16)"
    @classmethod
    def get_values(cls): return [item.value for item in cls]

def get_model_info(model_name: str) -> dict:
    return MODEL_CONFIGS.get(model_name, {})

def get_device_info() -> dict:
    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / 1024**3
        gpu_info = {"available": True, "total_memory": total_mem, "free_memory": total_mem - (torch.cuda.memory_allocated(0) / 1024**3)}
    else:
        gpu_info = {"available": False, "total_memory": 0, "free_memory": 0}

    sys_mem = psutil.virtual_memory()
    sys_mem_info = {"total": sys_mem.total / 1024**3, "available": sys_mem.available / 1024**3}

    device_info = {"gpu": gpu_info, "system_memory": sys_mem_info, "device_type": "cpu", "recommended_device": "cpu", "memory_sufficient": True, "warning_message": ""}

    if platform.system() == "Darwin" and platform.processor() == "arm":
        device_info.update({"device_type": "apple_silicon", "recommended_device": "mps"})
        if sys_mem_info["total"] < 16:
            device_info.update({"memory_sufficient": False, "warning_message": "Apple Silicon memory is less than 16GB, performance may be affected."})
    elif gpu_info["available"]:
        device_info.update({"device_type": "nvidia_gpu", "recommended_device": "cuda"})
        if gpu_info["total_memory"] < 8:
            device_info.update({"memory_sufficient": False, "warning_message": "GPU VRAM is less than 8GB, performance may be degraded."})
    return device_info

def check_memory_requirements(model_name: str, quantization: str, device_info: dict) -> str:
    model_info = get_model_info(model_name)
    vram_req = model_info.get("vram_requirement", {})
    quant_map = {Quantization.Q4_BIT: vram_req.get("4bit", 0), Quantization.Q8_BIT: vram_req.get("8bit", 0), Quantization.NONE: vram_req.get("full", 0)}
    
    base_memory = quant_map.get(quantization, 0)
    device = device_info["recommended_device"]
    use_cpu_mps = device in ["cpu", "mps"]
    
    required_mem = base_memory * (1.5 if use_cpu_mps else 1.0)
    available_mem = device_info["system_memory"]["available"] if use_cpu_mps else device_info["gpu"]["free_memory"]
    mem_type = "System RAM" if use_cpu_mps else "GPU VRAM"

    if required_mem * 1.2 > available_mem:
        print(f"Warning: Insufficient {mem_type} ({available_mem:.2f}GB available). Lowering quantization...")
        if quantization == Quantization.NONE: return Quantization.Q8_BIT
        if quantization == Quantization.Q8_BIT: return Quantization.Q4_BIT
        raise RuntimeError(f"Insufficient {mem_type} even for 4-bit quantization.")
    return quantization

def check_flash_attention() -> bool:
    try:
        import flash_attn
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return major >= 8
    except ImportError: return False
    return False

class ImageProcessor:
    def to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

class ModelDownloader:
    def __init__(self, configs):
        self.configs = configs
        self.models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def ensure_model_available(self, model_name):
        model_info = self.configs.get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found in configuration.")

        repo_id = model_info['repo_id']
        model_folder_name = repo_id.split('/')[-1]
        model_path = self.models_dir / model_folder_name
        
        print(f"Ensuring model '{model_name}' is available at {model_path}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", ".git*"]
        )
        print(f"Model '{model_name}' is ready.")
        return str(model_path)

class AILab_QwenVL_Advanced:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.current_device = None
        self.device_info = get_device_info()
        self.downloader = ModelDownloader(MODEL_CONFIGS)
        self.image_processor = ImageProcessor()
        print(f"QwenVL Node Initialized. Device: {self.device_info['device_type']}")
        if not self.device_info["memory_sufficient"]:
            print(f"Warning: {self.device_info['warning_message']}")

    def clear_model_resources(self):
        if self.model is not None:
            print("Releasing model resources...")
            del self.model, self.processor, self.tokenizer
            self.model = self.processor = self.tokenizer = None
            self.current_model_name = self.current_quantization = self.current_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_name: str, quantization_str: str, device: str = "auto"):
        effective_device = self.device_info["recommended_device"] if device == "auto" else device
        
        if self.model is not None and self.current_model_name == model_name and self.current_quantization == quantization_str and self.current_device == effective_device:
            return

        self.clear_model_resources()

        model_info = get_model_info(model_name)
        if model_info.get("quantized"):
            if self.device_info["gpu"]["available"]:
                major, minor = torch.cuda.get_device_capability()
                cc = major + minor / 10
                if cc < 8.9:
                    raise ValueError(
                        f"FP8 models require a GPU with Compute Capability 8.9 or higher (e.g., RTX 4090). "
                        f"Your GPU's capability is {cc}. Please select a non-FP8 model."
                    )

        model_path = self.downloader.ensure_model_available(model_name)
        adjusted_quantization = check_memory_requirements(model_name, quantization_str, self.device_info)
        
        quant_config, load_dtype = None, torch.float16
        if not get_model_info(model_name).get("quantized", False):
            if adjusted_quantization == Quantization.Q4_BIT:
                quant_config, load_dtype = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True), None
            elif adjusted_quantization == Quantization.Q8_BIT:
                quant_config, load_dtype = BitsAndBytesConfig(load_in_8bit=True), None

        device_map = "auto"
        if effective_device == "cuda" and torch.cuda.is_available(): device_map = {"": 0}

        load_kwargs = {"device_map": device_map, "torch_dtype": load_dtype, "attn_implementation": "flash_attention_2" if check_flash_attention() else "sdpa", "use_safetensors": True}
        if quant_config: load_kwargs["quantization_config"] = quant_config

        print(f"Loading model '{model_name}'...")
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.current_model_name, self.current_quantization, self.current_device = model_name, quantization_str, effective_device
        print("Model loaded successfully.")

    @classmethod
    def INPUT_TYPES(cls):
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]
        preset_prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image in detail."])

        return {
            "required": {
                "model_name": (model_names, {"default": default_model}),
                "quantization": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "preset_prompt": (preset_prompts, {"default": preset_prompts[2]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "If provided, this will override the preset prompt."}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": { "image": ("IMAGE",), "video": ("IMAGE",) }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    @torch.no_grad()
    def process(self, model_name, quantization, preset_prompt, max_tokens, temperature, top_p, repetition_penalty, num_beams, frame_count, device, seed, custom_prompt="", image=None, video=None, keep_model_loaded=True):
        start_time = time.time()
        torch.manual_seed(seed)
        
        try:
            self.load_model(model_name, quantization, device)
            effective_device = self.current_device
            
            prompt_text = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
            if custom_prompt and custom_prompt.strip():
                prompt_text = custom_prompt.strip()
            
            conversation = [{"role": "user", "content": []}]
            
            if image is not None:
                conversation[0]["content"].append({"type": "image", "image": self.image_processor.to_pil(image)})
            
            if video is not None:
                video_frames = [Image.fromarray((frame.cpu().numpy() * 255).astype(np.uint8)) for frame in video]
                if len(video_frames) > frame_count:
                    indices = np.linspace(0, len(video_frames) - 1, frame_count, dtype=int)
                    sampled_frames = [video_frames[i] for i in indices]
                else:
                    sampled_frames = video_frames

                if sampled_frames and len(sampled_frames) == 1:
                    sampled_frames.append(sampled_frames[0])
                    
                if sampled_frames:
                    conversation[0]["content"].append({"type": "video", "video": sampled_frames})

            conversation[0]["content"].append({"type": "text", "text": prompt_text})

            text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            
            pil_images = [item['image'] for item in conversation[0]['content'] if item['type'] == 'image']
            video_frames_list = [frame for item in conversation[0]['content'] if item['type'] == 'video' for frame in item['video']]
            videos_arg = [video_frames_list] if video_frames_list else None
            
            inputs = self.processor(text=text_prompt, images=pil_images if pil_images else None, videos=videos_arg, return_tensors="pt")
            model_inputs = {k: v.to(effective_device) for k, v in inputs.items() if torch.is_tensor(v)}

            stop_tokens = [self.tokenizer.eos_token_id]
            if hasattr(self.tokenizer, 'eot_id'): stop_tokens.append(self.tokenizer.eot_id)

            gen_kwargs = {"max_new_tokens": max_tokens, "repetition_penalty": repetition_penalty, "num_beams": num_beams, "eos_token_id": stop_tokens, "pad_token_id": self.tokenizer.pad_token_id}
            if num_beams > 1:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})

            outputs = self.model.generate(**model_inputs, **gen_kwargs)
            input_ids_len = model_inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(outputs[0, input_ids_len:], skip_special_tokens=True)
            
            print(f"Generation finished in {time.time() - start_time:.2f} seconds.")
            return (text.strip(),)

        except (ValueError, RuntimeError) as e:
            error_message = f"ERROR: {str(e)}"
            print(error_message)
            return (error_message,)
        finally:
            if not keep_model_loaded: self.clear_model_resources()

class AILab_QwenVL(AILab_QwenVL_Advanced):
    @classmethod
    def INPUT_TYPES(cls):
        model_names = [name for name in MODEL_CONFIGS.keys() if not name.startswith('_')]
        default_model = model_names[4] if len(model_names) > 4 else model_names[0]
        preset_prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image in detail."])

        return {
            "required": {
                "model_name": (model_names, {"default": default_model}),
                "quantization": (list(Quantization.get_values()), {"default": Quantization.NONE}),
                "preset_prompt": (preset_prompts, {"default": preset_prompts[2]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "If provided, this will override the preset prompt."}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": { "image": ("IMAGE",), "video": ("IMAGE",) }
        }

    FUNCTION = "process_standard"
    
    def process_standard(self, model_name, quantization, preset_prompt, max_tokens, seed, custom_prompt="", image=None, video=None, keep_model_loaded=True):
        return self.process(
            model_name=model_name,
            quantization=quantization,
            preset_prompt=preset_prompt,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            num_beams=1,
            frame_count=16,
            device="auto",
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            keep_model_loaded=keep_model_loaded,
            seed=seed
        )

NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL": AILab_QwenVL,
    "AILab_QwenVL_Advanced": AILab_QwenVL_Advanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL": "QwenVL",
    "AILab_QwenVL_Advanced": "QwenVL (Advanced)",
}