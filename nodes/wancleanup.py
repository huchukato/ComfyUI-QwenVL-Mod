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
                    "Gentle Cleanup",
                    "After T2V Use",
                    "Before WAN Load", 
                    "After WAN Use",
                    "Full Memory Reset"
                ], {"default": "Gentle Cleanup"}),
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
            if cleanup_mode == "Gentle Cleanup":
                self._gentle_cleanup()
            elif cleanup_mode == "After T2V Use":
                self._after_t2v_use()
            elif cleanup_mode == "Before WAN Load":
                self._before_wan_load()
            elif cleanup_mode == "After WAN Use":
                self._after_wan_use()
            elif cleanup_mode == "Full Memory Reset":
                self._full_memory_reset()
            
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
    
    def _after_t2v_use(self):
        """Targeted cleanup after T2V use - prepares for QwenVL prompt generation"""
        try:
            print("🎯 After T2V Use: Cleaning T2V residues for QwenVL...")
            
            if torch.cuda.is_available():
                # Gentle but thorough cleanup targeting T2V residues
                for i in range(3):
                    torch.cuda.empty_cache()
                    if i == 1:  # Synchronize in middle
                        torch.cuda.synchronize()
                    print(f"  T2V residue clear {i+1}/3")
                
                # Light memory pressure to force T2V cleanup
                try:
                    temp_tensor = torch.randn(1500, 1500, device='cuda')
                    del temp_tensor
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("  T2V memory pressure applied")
                except:
                    pass
                
                # Final synchronization
                torch.cuda.synchronize()
                print("🎯 After T2V Use cleanup completed")
            
        except Exception as e:
            print(f"⚠️ After T2V Use cleanup warning: {e}")
    
    def _before_wan_load(self):
        """Aggressive cleanup before loading WAN model - includes delay and full options"""
        try:
            print("🚀 Before WAN Load: Preparing memory for WAN model...")
            
            # Add delay for QwenVL to fully unload
            import time
            time.sleep(2)
            print("  ⏱️ Delay for QwenVL unload completed")
            
            if torch.cuda.is_available():
                # Multiple aggressive cache clears
                for i in range(5):
                    torch.cuda.empty_cache()
                    if i % 2 == 0:
                        torch.cuda.synchronize()
                    print(f"  Cache clear {i+1}/5")
                
                # Memory pressure to force cleanup
                try:
                    temp_tensor = torch.randn(2000, 2000, device='cuda')
                    del temp_tensor
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("  Memory pressure applied")
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
            print("🚀 Before WAN Load cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Before WAN Load cleanup warning: {e}")
    
    def _after_wan_use(self):
        """Targeted cleanup after WAN use - focuses on text encoder"""
        try:
            print("🎯 After WAN Use: Cleaning WAN text encoder...")
            
            if torch.cuda.is_available():
                # Multiple cache clears targeting text encoder
                for i in range(3):
                    torch.cuda.empty_cache()
                    if i == 1:  # Synchronize in middle
                        torch.cuda.synchronize()
                    print(f"  Cache clear {i+1}/3")
                
                print("🎯 After WAN Use cleanup completed")
            
        except Exception as e:
            print(f"⚠️ After WAN Use cleanup warning: {e}")
    
    def _gentle_cleanup(self):
        """Very gentle cleanup - safe option"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 Gentle cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Gentle cleanup warning: {e}")
    
    def _full_memory_reset(self):
        """Comprehensive memory reset between segments"""
        try:
            if torch.cuda.is_available():
                # Multiple cache clears
                for _ in range(7):
                    torch.cuda.empty_cache()
                
                # Synchronization
                torch.cuda.synchronize()
                
                # Garbage collection
                gc.collect()
                
                print("💥 Full memory reset completed")
            
        except Exception as e:
            print(f"⚠️ Full memory reset warning: {e}")

# Register the node
NODE_CLASS_MAPPINGS = {
    "WANCleanup": WANCleanup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANCleanup": "WAN Cleanup Node"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
