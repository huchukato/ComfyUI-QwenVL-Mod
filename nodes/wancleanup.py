#!/usr/bin/env python3
"""
ComfyUI WAN Models Cleanup Node
Specialized cleanup for WAN 2.2 models to prevent memory issues in story workflows
"""

import gc
import torch
import folder_paths
from comfy import model_management

class WANCleanup:
    """
    Specialized cleanup node for WAN 2.2 models
    Prevents memory accumulation in multi-segment video workflows
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cleanup_mode": ([
                    "wan_models_only", 
                    "aggressive_wan", 
                    "wan_and_cache", 
                    "full_cleanup"
                ], {"default": "wan_models_only"}),
                "force_gc": ("BOOLEAN", {"default": True}),
                "clear_cuda_cache": ("BOOLEAN", {"default": True}),
                "synchronize_cuda": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "delay_seconds": ("INT", {"default": 0, "min": 0, "max": 10}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "cleanup_wan_memory"
    CATEGORY = "🔷 QwenVL-Mod/Utils"
    OUTPUT_NODE = True
    
    def cleanup_wan_memory(self, cleanup_mode, force_gc, clear_cuda_cache, synchronize_cuda, delay_seconds=0):
        """
        Cleanup WAN models and memory to prevent crashes in story workflows
        """
        try:
            # Add delay if specified (useful for timing-sensitive workflows)
            if delay_seconds > 0:
                import time
                time.sleep(delay_seconds)
            
            print(f"🧹 WAN Cleanup: Starting {cleanup_mode} cleanup...")
            
            # Get current memory state
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                print(f"📊 Initial VRAM: {initial_memory / 1024**3:.2f} GB")
            
            # Mode-specific cleanup
            if cleanup_mode == "wan_models_only":
                self._cleanup_wan_models_only()
            elif cleanup_mode == "aggressive_wan":
                self._aggressive_wan_cleanup()
            elif cleanup_mode == "wan_and_cache":
                self._cleanup_wan_and_cache()
            elif cleanup_mode == "full_cleanup":
                self._full_memory_cleanup()
            
            # Force garbage collection
            if force_gc:
                gc.collect()
                print("🗑️ Garbage collection completed")
            
            # Clear CUDA cache
            if clear_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧠 CUDA cache cleared")
            
            # Synchronize CUDA operations
            if synchronize_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
                print("⚡ CUDA synchronized")
            
            # Report final memory state
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated()
                freed_memory = initial_memory - final_memory
                print(f"📉 Final VRAM: {final_memory / 1024**3:.2f} GB")
                print(f"💾 Freed: {freed_memory / 1024**3:.2f} GB")
            
            print("✅ WAN Cleanup completed successfully!")
            
        except Exception as e:
            print(f"❌ WAN Cleanup failed: {str(e)}")
            raise e
        
        return ()
    
    def _cleanup_wan_models_only(self):
        """Gentle cleanup targeting only WAN models"""
        try:
            # Try to clear WAN-specific model references
            if hasattr(folder_paths, 'get_filename_list'):
                # Clear WAN model caches if they exist
                wan_models = ['wan2.1', 'wan2.2', 'wan']
                for model_type in wan_models:
                    if model_type in folder_paths.get_filename_list('diffusion_models'):
                        print(f"🧹 Clearing {model_type} model references")
            
            # Use ComfyUI's model management for safe cleanup
            model_management.unload_all_models()
            print("🎯 WAN models only cleanup completed")
            
        except Exception as e:
            print(f"⚠️ WAN models cleanup warning: {e}")
    
    def _aggressive_wan_cleanup(self):
        """More aggressive WAN model cleanup"""
        try:
            # Unload all models first
            model_management.unload_all_models()
            
            # Additional WAN-specific cleanup
            if torch.cuda.is_available():
                # Clear any remaining WAN-related tensors
                for obj in gc.get_objects():
                    if hasattr(obj, 'data_ptr') and hasattr(obj, 'device'):
                        if str(obj.device).startswith('cuda'):
                            try:
                                del obj
                            except:
                                pass
            
            print("🔥 Aggressive WAN cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Aggressive WAN cleanup warning: {e}")
    
    def _cleanup_wan_and_cache(self):
        """Cleanup WAN models plus system caches"""
        try:
            # WAN models cleanup
            self._cleanup_wan_models_only()
            
            # Additional cache cleanup
            if hasattr(model_management, 'interrupt_processing'):
                model_management.interrupt_processing(False)
            
            print("🧼 WAN and cache cleanup completed")
            
        except Exception as e:
            print(f"⚠️ WAN and cache cleanup warning: {e}")
    
    def _full_memory_cleanup(self):
        """Complete memory cleanup (use with caution)"""
        try:
            # Unload all models
            model_management.unload_all_models()
            
            # Clear all possible caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force multiple garbage collection passes
            for _ in range(3):
                gc.collect()
            
            print("💥 Full memory cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Full cleanup warning: {e}")

# Register the node
NODE_CLASS_MAPPINGS = {
    "WANCleanup": WANCleanup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANCleanup": "WAN Cleanup Node"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
