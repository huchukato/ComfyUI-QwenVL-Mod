# ComfyUI-QwenVL — Livepeer Agent render node (hackathon branch)
# Calls the Livepeer Agent MCP raw profile (deterministic dispatch) via
# stateless JSON-RPC over HTTP. No extra dependencies required.

import base64
import io
import json
import os
import re
import time
import urllib.request
import uuid

import torch
from PIL import Image

import folder_paths

try:
    from comfy_api.latest import InputImpl
    _VIDEO_FROM_FILE = getattr(InputImpl, "VideoFromFile", None)
except Exception:
    try:
        from comfy_api.input_impl import VideoFromFile as _VIDEO_FROM_FILE
    except Exception:
        _VIDEO_FROM_FILE = None

MCP_ENDPOINT = "https://agent.livepeer.org/api/mcp/raw"
POLL_INTERVAL_S = 6
DEFAULT_TIMEOUT_S = 480

# Video capabilities verified on the network at build time. "auto" resolves to
# minimax-h3-i2v / minimax-h3-t2v depending on whether a source image is given.
CAPABILITY_OPTIONS = [
    "auto",
    "minimax-h3-i2v",
    "minimax-h3-t2v",
    "kling-o3-i2v",
    "kling-o3-t2v",
    "kling-v3-turbo-i2v",
    "kling-v3-turbo-t2v",
    "kling-v3-turbo-pro-i2v",
    "kling-v3-turbo-pro-t2v",
    "seedance-i2v",
    "seedance-i2v-fast",
    "seedance-mini-i2v",
    "seedance-25-i2v",
    "seedance-25-t2v",
    "seedance-mini-t2v",
    "ltx-i2v",
    "ltx-q-i2v",
    "ltx-25-i2v-fast",
    "ltx-25-i2v-pro",
    "ltx-t2v",
    "ltx-q-t2v",
    "ltx-25-t2v-fast",
    "ltx-25-t2v-pro",
    "veo-i2v",
    "veo-t2v",
    "pixverse-i2v",
    "pixverse-t2v",
    "ray-32-i2v",
    "ray-32-t2v",
    "flux-3-i2v",
    "flux-3-t2v",
    "flux-3-draft-i2v",
    "flux-3-draft-t2v",
    "cosmos-3-i2v",
    "grok-imagine-video-t2v",
    "animatediff-t2v",
]


def _mcp_call(tool, arguments, api_key="", endpoint=MCP_ENDPOINT, timeout=120):
    """Stateless MCP tools/call against the Livepeer Agent HTTP transport."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    # Streamable-HTTP may answer with SSE frames; unwrap them if so.
    if body.lstrip().startswith("event:") or "\ndata:" in body or body.lstrip().startswith("data:"):
        data_lines = [l[5:].strip() for l in body.splitlines() if l.startswith("data:")]
        body = data_lines[-1] if data_lines else "{}"
    envelope = json.loads(body)
    if "error" in envelope:
        raise RuntimeError(f"MCP error {envelope['error'].get('code')}: {envelope['error'].get('message')}")
    result = envelope.get("result") or {}
    if result.get("isError"):
        sc = result.get("structuredContent") or {}
        msg = (sc.get("error") or {}).get("message") or "".join(
            c.get("text", "") for c in result.get("content", [])
        ) or "unknown tool error"
        raise RuntimeError(f"Livepeer tool '{tool}' failed: {msg}")
    sc = result.get("structuredContent")
    if isinstance(sc, dict) and sc:
        return sc
    for chunk in result.get("content", []):
        text = chunk.get("text", "")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    return {"text": "".join(c.get("text", "") for c in result.get("content", []))}


def _tensor_to_jpeg_b64(image, max_side=1536, max_bytes=2_500_000):
    """ComfyUI IMAGE tensor (B,H,W,C float 0-1) -> base64 JPEG under max_bytes."""
    frame = image[0] if image.dim() == 4 else image
    arr = (frame.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    pil = Image.fromarray(arr)
    quality = 90
    while True:
        if max(pil.size) > max_side:
            ratio = max_side / max(pil.size)
            pil = pil.resize((max(1, int(pil.width * ratio)), max(1, int(pil.height * ratio))), Image.LANCZOS)
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= max_bytes or (quality <= 55 and max_side <= 768):
            return base64.b64encode(data).decode("ascii")
        quality -= 10
        if quality < 55:
            quality = 55
            max_side = max_side // 2


def _extract_url(payload):
    for key in ("url", "video_url", "output_url", "asset_url", "result_url"):
        if isinstance(payload.get(key), str) and payload[key].startswith("http"):
            return payload[key]
    for key in ("result", "output", "asset", "video", "media"):
        sub = payload.get(key)
        if isinstance(sub, dict):
            found = _extract_url(sub)
            if found:
                return found
    text = payload.get("text", "")
    match = re.search(r"https://\S+", text)
    return match.group(0).rstrip(').,]"\'') if match else None


def _resolve_capability(capability, custom, has_image):
    if custom and custom.strip():
        return custom.strip()
    if capability != "auto":
        return capability
    return "minimax-h3-i2v" if has_image else "minimax-h3-t2v"


class AILab_LivepeerRender:
    """Send a shot prompt to the Livepeer Agent network and get a video back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Shot-native English video prompt: camera movement, framing, subject action, lighting, pacing. The chat writes this automatically."}),
                "capability": (CAPABILITY_OPTIONS, {"default": "auto", "tooltip": "Model on the Livepeer network. 'auto' picks minimax-h3-i2v with an image input, minimax-h3-t2v without."}),
                "custom_capability": ("STRING", {"default": "", "tooltip": "Optional exact capability name overriding the dropdown (from list_capabilities)."}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "tooltip": "Video duration in seconds. Billed per second."}),
                "resolution": (["default", "768P", "2K", "1080p", "720p"], {"default": "default", "tooltip": "Resolution tier when the model supports it (MiniMax H3: 768P/2K)."}),
                "aspect_ratio": (["auto", "16:9", "9:16", "1:1", "3:2", "2:3", "4:3", "3:4", "2.35:1"], {"default": "auto", "tooltip": "Passed through when the capability accepts it."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "tooltip": "-1 = random take. A fixed seed re-renders the same take for A/B iterations."}),
                "timeout_s": ("INT", {"default": DEFAULT_TIMEOUT_S, "min": 60, "max": 1800, "tooltip": "Max seconds to wait for the render. MiniMax H3 p95 is ~300s; the server aborts at ~450s."}),
                "filename_prefix": ("STRING", {"default": "Livepeer/", "tooltip": "Output filename prefix in the ComfyUI output folder."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Reference/first frame. Enables i2v (animate). Uploaded to Livepeer storage first."}),
                "end_image": ("IMAGE", {"tooltip": "Optional last keyframe for transition-capable models."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Optional Daydream sk_... key. Empty uses the hackathon demo balance."}),
                "extra_params": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional JSON object merged into the capability inputs (e.g. {\"guidance_scale\": 7})."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "url", "report")
    FUNCTION = "run"
    CATEGORY = "QwenVL-Mod"
    OUTPUT_NODE = True

    def run(self, prompt, capability, custom_capability, duration, resolution, aspect_ratio, seed, timeout_s, filename_prefix, image=None, end_image=None, api_key="", extra_params=""):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Livepeer render needs a non-empty prompt")

        cap = _resolve_capability(capability, custom_capability, image is not None)
        t0 = time.time()

        source_url = None
        if image is not None:
            b64 = _tensor_to_jpeg_b64(image)
            try:
                up = _mcp_call("upload", {"data": b64, "mime_type": "image/jpeg", "kind": "image", "filename": "frame.jpg"}, api_key)
                source_url = _extract_url(up)
            except Exception:
                source_url = None
            if not source_url:
                source_url = f"data:image/jpeg;base64,{b64}"

        inputs = {"duration": int(duration)}
        if resolution != "default":
            inputs["resolution"] = resolution
        if aspect_ratio != "auto":
            inputs["aspect_ratio"] = aspect_ratio
        if seed >= 0:
            inputs["seed"] = int(seed)
        if end_image is not None:
            eb64 = _tensor_to_jpeg_b64(end_image)
            try:
                eup = _mcp_call("upload", {"data": eb64, "mime_type": "image/jpeg", "kind": "image", "filename": "end.jpg"}, api_key)
                inputs["end_image_url"] = _extract_url(eup) or f"data:image/jpeg;base64,{eb64}"
            except Exception:
                inputs["end_image_url"] = f"data:image/jpeg;base64,{eb64}"
        if extra_params.strip():
            try:
                extra = json.loads(extra_params)
                if isinstance(extra, dict):
                    inputs.update(extra)
            except ValueError as e:
                raise ValueError(f"extra_params is not valid JSON: {e}")

        args = {"capability": cap, "prompt": prompt, "inputs": inputs, "async": True}
        if source_url:
            args["source_url"] = source_url

        submit = _mcp_call("run_capability", args, api_key, timeout=180)
        url = _extract_url(submit)
        job_id = submit.get("job_id") or submit.get("id")
        status = submit.get("status", "")

        deadline = t0 + int(timeout_s)
        while not url and job_id and time.time() < deadline:
            if status in ("failed", "cancelled", "canceled", "error"):
                break
            time.sleep(POLL_INTERVAL_S)
            poll = _mcp_call("get_create_media", {"job_id": job_id}, api_key, timeout=60)
            status = poll.get("status", status)
            url = _extract_url(poll)
            if status == "done":
                break
            if status in ("failed", "cancelled", "canceled", "error"):
                err = poll.get("error") or poll.get("detail") or status
                raise RuntimeError(f"Livepeer job {job_id} {status}: {err}")

        if not url:
            raise RuntimeError(f"Livepeer job {job_id or '?'} produced no URL before timeout ({timeout_s}s)")

        ext = ".mp4"
        m = re.search(r"\.(mp4|webm|mov|png|jpe?g|webp)(?:\?|$)", url, re.I)
        if m:
            ext = "." + m.group(1).lower().replace("jpeg", "jpg")
        dl_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        req = urllib.request.Request(url, headers=dl_headers)
        with urllib.request.urlopen(req, timeout=300) as resp:
            blob = resp.read()

        out_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), 0, 0)
        file_name = f"{fname}_{counter:05}_.{ext.lstrip('.')}"
        path = os.path.join(out_dir, file_name)
        with open(path, "wb") as f:
            f.write(blob)

        elapsed = round(time.time() - t0, 1)
        report = {
            "capability": cap,
            "job_id": job_id,
            "url": url,
            "duration_s": duration,
            "elapsed_s": elapsed,
            "cost_usd": submit.get("cost_usd") or submit.get("cost_usd_estimated") or submit.get("cost"),
            "file": os.path.join(subfolder, file_name) if subfolder else file_name,
            "bytes": len(blob),
            "model_note": submit.get("model_note"),
        }
        video_out = _VIDEO_FROM_FILE(path) if _VIDEO_FROM_FILE and ext in (".mp4", ".webm", ".mov") else None
        return {
            "ui": {"images": [{"filename": file_name, "subfolder": subfolder, "type": "output"}], "animated": (True,)},
            "result": (video_out, url, json.dumps(report, indent=2)),
        }


NODE_CLASS_MAPPINGS = {
    "AILab_LivepeerRender": AILab_LivepeerRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_LivepeerRender": "🌐 Livepeer Agent Render",
}
