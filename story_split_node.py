"""
Custom node for splitting WAN 2.2 Story prompts
Handles various separators automatically
"""

class StorySplitNode:
    """
    Splits WAN 2.2 Story output into 4 separate prompts
    Handles different separators: \n, \n\n, \n\n\n
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "auto_split": ("BOOLEAN", {"default": True}),
                "custom_delimiter": ("STRING", {"multiline": False, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_1", "prompt_2", "prompt_3", "prompt_4", "debug_info")
    FUNCTION = "split_story"
    CATEGORY = "Custom/Story"
    
    def split_story(self, text, auto_split=True, custom_delimiter=""):
        """
        Split story text into 4 prompts
        """
        if not text:
            return ("", "", "", "", "Empty input")
        
        text = text.strip()
        
        if auto_split and not custom_delimiter:
            # Try different separators automatically
            if '\n\n\n' in text:
                prompts = text.split('\n\n\n')
            elif '\n\n' in text:
                prompts = text.split('\n\n')
            elif '\n' in text:
                prompts = text.split('\n')
            else:
                # No clear separator, split by paragraphs
                prompts = [p for p in text.split('\n') if p.strip()]
        else:
            # Use custom delimiter
            delimiter = custom_delimiter if custom_delimiter else '\n\n'
            prompts = text.split(delimiter)
        
        # Clean up prompts
        clean_prompts = []
        for prompt in prompts:
            clean_prompt = prompt.strip()
            if clean_prompt and not clean_prompt.startswith("Prompt") and "content describing" not in clean_prompt:
                clean_prompts.append(clean_prompt)
        
        # Ensure we have exactly 4 prompts
        while len(clean_prompts) < 4:
            clean_prompts.append("")
        
        result_prompts = clean_prompts[:4]
        
        # Debug info
        debug = f"Found {len(prompts)} parts, cleaned to {len(clean_prompts)} prompts"
        
        return (
            result_prompts[0], 
            result_prompts[1], 
            result_prompts[2], 
            result_prompts[3], 
            debug
        )

# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "StorySplitNode": StorySplitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StorySplitNode": "Story Split Node"
}
