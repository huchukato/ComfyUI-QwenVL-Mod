"""
Simple Show Text node for displaying text in ComfyUI
Alternative to ComfyUI-Custom-Scripts show_text
"""

class ShowTextNode:
    """
    Display text in ComfyUI interface and pass it through
    For use with SetNode and other text processing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text input to display and pass through"}),
            },
            "optional": {
                "title": ("STRING", {"multiline": False, "default": "Text Output", "tooltip": "Title for the display"}),
                "show_length": ("BOOLEAN", {"default": True, "tooltip": "Show character count"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show_text"
    CATEGORY = "🔷QwenVL-Mod/Utils"
    OUTPUT_NODE = True
    
    def show_text(self, text, title="Text Output", show_length=True):
        """
        Display text and pass it through for SetNode chaining
        """
        if not text:
            text = ""
        
        # Display logic handled by ComfyUI UI
        # Just return the text for chaining
        
        return (text,)

# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ShowTextNode": ShowTextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShowTextNode": "Show Text"
}
