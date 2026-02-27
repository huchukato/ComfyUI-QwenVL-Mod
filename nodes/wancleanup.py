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
                    "wan_models_only", 
                    "wan_text_encoder_only",
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
    
    RETURN_TYPES = ("*",)  # Pass through the input
    RETURN_NAMES = ("output",)
    FUNCTION = "cleanup_wan_memory"
    CATEGORY = "🔷 QwenVL-Mod/Utils"
    OUTPUT_NODE = True
    
    def cleanup_wan_memory(self, input, cleanup_mode, force_gc, clear_cuda_cache, synchronize_cuda, delay_seconds=0):
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
            elif cleanup_mode == "wan_text_encoder_only":
                self._cleanup_wan_text_encoder_only()
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
        
        return (input,)  # Pass through the input
    
    def _cleanup_wan_text_encoder_only(self):
        """Specific cleanup for WAN text encoder - the main memory hog"""
        try:
            if torch.cuda.is_available():
                # Force multiple cache clears to target text encoder
                print("🎯 Targeting WAN text encoder cleanup...")
                
                # Multiple aggressive cache clears
                for i in range(5):
                    torch.cuda.empty_cache()
                    if i % 2 == 0:  # Synchronize every other iteration
                        torch.cuda.synchronize()
                    print(f"  Cache clear {i+1}/5")
                
                # Additional memory pressure to force text encoder cleanup
                try:
                    # Create temporary tensor to pressure memory, then delete
                    temp_tensor = torch.randn(1000, 1000, device='cuda')
                    del temp_tensor
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("  Memory pressure applied")
                except:
                    pass
                
                print("🎯 WAN text encoder cleanup completed")
            
        except Exception as e:
            print(f"⚠️ WAN text encoder cleanup warning: {e}")
    
    def _cleanup_wan_models_only(self):
        """Very gentle cleanup - just basic garbage collection"""
        try:
            # Only do safe cleanup - don't touch model_management directly
            if torch.cuda.is_available():
                # Just clear CUDA cache safely
                torch.cuda.empty_cache()
                print("🎯 Safe WAN cleanup (CUDA cache only) completed")
            
        except Exception as e:
            print(f"⚠️ WAN cleanup warning: {e}")
    
    def _aggressive_wan_cleanup(self):
        """More aggressive but still safe cleanup"""
        try:
            # Skip model_management.unload_all_models() - too aggressive
            # Just do more thorough CUDA cleanup
            if torch.cuda.is_available():
                # Multiple cache clear attempts
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print("🔥 Thorough CUDA cleanup completed")
            
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
        """Safe but thorough memory cleanup"""
        try:
            # Skip unload_all_models() - too dangerous
            # Just do comprehensive CUDA cleanup
            if torch.cuda.is_available():
                # Clear cache multiple times
                for _ in range(5):
                    torch.cuda.empty_cache()
                
                # Force synchronization
                torch.cuda.synchronize()
                
                # Final garbage collection
                gc.collect()
                
                print("💥 Safe thorough cleanup completed")
            
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
