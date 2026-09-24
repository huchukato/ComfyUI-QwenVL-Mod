"""
{
  "name": "QwenVL-Mod: Enhanced Vision-Language",
  "description": "Enhanced QwenVL node with Flash Attention 2, WAN 2.2 video generation, free abliterated models, and comprehensive NSFW support. Advanced fork with major improvements over original for stable multimodal AI workflows.",
  "author": "huchukato",
  "version": "2.8.14",
  "url": "https://github.com/huchukato/ComfyUI-QwenVL-Mod",
  "category": "image"
}
"""

import asyncio
import html
import importlib.util
import os
import sys

from aiohttp import web
import folder_paths
from server import PromptServer

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

# ──────────────────────────────────────────────────────────────────────────────
# Download log endpoint: serves last 30 lines of model download log at /dlstatus
# Works on the same port as ComfyUI (8188), no extra port exposure needed.
# ──────────────────────────────────────────────────────────────────────────────
_DL_LOGS = [
    "/var/log/minimax-h3-models.log",
    "/var/log/ltx-models.log",
    "/var/log/wan-models.log",
]

@PromptServer.instance.routes.get("/dlstatus")
async def _dlstatus(request):
    for log_path in _DL_LOGS:
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()[-30:]
                return web.Response(text="".join(lines), content_type="text/plain")
            except Exception:
                pass
    return web.Response(text="Download log not yet available...", content_type="text/plain")

_DLSTATUS_PAGE = """<!DOCTYPE html>
<html>
<head><meta http-equiv="refresh" content="3"></head>
<body><pre style="font-family:monospace;font-size:12px;white-space:pre-wrap;">%s</pre></body>
</html>"""


@PromptServer.instance.routes.get("/dlstatus.html")
async def _dlstatus_html(request):
    for log_path in _DL_LOGS:
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path, "r") as f:
                tail = f.readlines()[-30:]
        except Exception:
            continue
        escaped = html.escape("".join(tail))
        return web.Response(text=_DLSTATUS_PAGE % escaped, content_type="text/html")
    return web.Response(text="Download log not yet available...", content_type="text/plain")

def load_modules_from_directory(directory):
    for file in os.listdir(directory):
        if file.endswith(".py"):
            file_path = os.path.join(directory, file)
            module_name = os.path.basename(file)[:-3]
            if module_name == os.path.basename(__file__)[:-3]:
                continue

            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"Error loading module {module_name}: {e}")

load_modules_from_directory(current_dir)

# Also load from nodes subdirectory
nodes_dir = os.path.join(current_dir, "nodes")
if os.path.exists(nodes_dir):
    load_modules_from_directory(nodes_dir)

from chat_service import CHAT_RUNTIME, list_output_images


@PromptServer.instance.routes.get("/qwenvl/chat/models")
async def _chat_models(request):
    return web.json_response(CHAT_RUNTIME.models())


@PromptServer.instance.routes.get("/qwenvl/chat/assets")
async def _chat_assets(request):
    return web.json_response(await asyncio.to_thread(list_output_images, folder_paths.get_output_directory()))


@PromptServer.instance.routes.post("/qwenvl/chat")
async def _chat(request):
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("request body must be an object")
        options = data.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("options must be an object")
        result = await asyncio.to_thread(
            CHAT_RUNTIME.chat,
            data.get("backend"),
            data.get("model"),
            data.get("messages"),
            data.get("graph"),
            options,
            data.get("images"),
            data.get("video"),
            data.get("directives"),
        )
        return web.json_response(result)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    except Exception as error:
        print(f"[QwenVL Chat] {error}")
        return web.json_response({"error": "Chat generation failed. Check the ComfyUI console."}, status=500)


@PromptServer.instance.routes.post("/qwenvl/chat/unload")
async def _chat_unload(request):
    try:
        data = await request.json()
        backend = data.get("backend", "all") if isinstance(data, dict) else "all"
        unloaded = await asyncio.to_thread(CHAT_RUNTIME.unload, backend)
        return web.json_response({"status": "success", "unloaded": unloaded})
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)


NODE_CLASS_MAPPINGS = dict(sorted(NODE_CLASS_MAPPINGS.items(), key=lambda x: NODE_DISPLAY_NAME_MAPPINGS.get(x[0], x[0])))
NODE_DISPLAY_NAME_MAPPINGS = dict(sorted(NODE_DISPLAY_NAME_MAPPINGS.items(), key=lambda x: x[1]))

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]
