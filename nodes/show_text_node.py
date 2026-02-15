"""
Simple Show Text node for displaying text in ComfyUI
Alternative to ComfyUI-Custom-Scripts show_text
"""

class ShowTextNode:
    """
    Display text in ComfyUI interface
    Simple alternative to show_text from ComfyUI-Custom-Scripts
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Text to display"}),
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
        Display text with optional title and character count
        """
        if not text:
            text = ""
        
        # Prepare display text
        display_text = text
        
        if show_length:
            char_count = len(text)
            line_count = len(text.split('\n'))
            display_text = f"{text}\n\n---\nCharacters: {char_count} | Lines: {line_count}"
        
        # Return the text for potential chaining
        return (text,)

# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ShowTextNode": ShowTextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShowTextNode": "Show Text"
}
