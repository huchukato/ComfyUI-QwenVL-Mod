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
                "input": ("*",),  # Any input to allow connection
                "cleanup_mode": ([
                    "Cache Only",
                    "Text Encoder",
                    "Full Cleanup",
                    "T2V + QwenVL Fix"
                ], {"default": "Cache Only"}),
            }
        }
    
    RETURN_TYPES = ("*",)  # Pass through the input
    RETURN_NAMES = ("output",)
    FUNCTION = "cleanup_wan_memory"
    CATEGORY = "🔷 QwenVL-Mod/Utils"
    OUTPUT_NODE = True
    
    def cleanup_wan_memory(self, input, cleanup_mode):
        """
        Cleanup WAN models and memory to prevent crashes in story workflows
        """
        try:
            print(f"🧹 WAN Cleanup: Starting {cleanup_mode} cleanup...")
            
            # Get current memory state
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                print(f"📊 Initial VRAM: {initial_memory / 1024**3:.2f} GB")
            
            # Mode-specific cleanup
            if cleanup_mode == "Cache Only":
                self._cache_only()
            elif cleanup_mode == "Text Encoder":
                self._text_encoder()
            elif cleanup_mode == "Full Cleanup":
                self._full_cleanup()
            elif cleanup_mode == "T2V + QwenVL Fix":
                self._t2v_qwen_fix()
            
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
        
        return (input,)  # Pass through the input
    
    def _cache_only(self):
        """Simple cache cleanup only"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("  Cache cleared")
            
        except Exception as e:
            print(f"⚠️ Cache cleanup warning: {e}")
    
    def _text_encoder(self):
        """Targeted cleanup for WAN text encoder"""
        try:
            if torch.cuda.is_available():
                for i in range(3):
                    torch.cuda.empty_cache()
                    if i == 1:
                        torch.cuda.synchronize()
                    print(f"  Text encoder clear {i+1}/3")
                
                print("  Text encoder cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Text encoder cleanup warning: {e}")
    
    def _full_cleanup(self):
        """Complete cleanup - unload all models"""
        try:
            if torch.cuda.is_available():
                try:
                    print("  Attempting full model unload...")
                    model_management.unload_all_models()
                    print("  All models unloaded successfully")
                except Exception as unload_error:
                    print(f"  Model unload failed: {unload_error}")
                    print("  Continuing with cache cleanup only...")
                
                for _ in range(5):
                    torch.cuda.empty_cache()
                
                torch.cuda.synchronize()
                gc.collect()
                print("  Full cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Full cleanup warning: {e}")
    
    def _t2v_qwen_fix(self):
        """Special fix for T2V + QwenVL conflict - uses Easy Use method"""
        try:
            if torch.cuda.is_available():
                try:
                    print("  Attempting model unload (Easy Use method)...")
                    model_management.unload_all_models()
                    print("  Models unloaded successfully")
                except Exception as unload_error:
                    print(f"  Model unload failed: {unload_error}")
                    print("  Continuing with cache cleanup only...")
                
                for i in range(3):
                    torch.cuda.empty_cache()
                    if i == 1:
                        torch.cuda.synchronize()
                    print(f"  T2V residue clear {i+1}/3")
                
                try:
                    temp_tensor = torch.randn(1500, 1500, device='cuda')
                    del temp_tensor
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("  Memory pressure applied")
                except:
                    pass
                
                torch.cuda.synchronize()
                gc.collect()
                print("  T2V + QwenVL fix completed")
            
        except Exception as e:
            print(f"⚠️ T2V + QwenVL fix warning: {e}")

# Register the node
NODE_CLASS_MAPPINGS = {
    "WANCleanup": WANCleanup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANCleanup": "WAN Cleanup Node"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
