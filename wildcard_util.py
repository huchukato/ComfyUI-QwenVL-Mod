"""Shared TagForge wildcard expansion helper.

Finds WildcardLoader regardless of the TagForge install dir name and expands
__wildcard__ tokens. Returns the input unchanged when TagForge is absent.
"""

import sys


def wildcard_loader():
    try:
        for mod in list(sys.modules.values()):
            mod_name = (getattr(mod, "__name__", "") or "").lower()
            if mod is not None and "tagforge" in mod_name and mod_name.endswith("wildcards") and hasattr(mod, "WildcardLoader"):
                return mod.WildcardLoader
    except Exception:
        pass
    try:
        import importlib
        for name in ("ComfyUI-TagForge.py.wildcards", "ComfyUI_TagForge.py.wildcards", "comfyui_tagforge.py.wildcards"):
            try:
                module = importlib.import_module(name)
                if hasattr(module, "WildcardLoader"):
                    return module.WildcardLoader
            except Exception:
                continue
    except Exception:
        pass
    return None


def expand_wildcard_tokens(text):
    if not text or "__" not in text:
        return text
    loader = wildcard_loader()
    if loader is not None:
        try:
            expanded = loader.process(text)
            if expanded:
                return expanded
        except Exception:
            pass
    return text
