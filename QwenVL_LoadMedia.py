# ComfyUI-QwenVL-Mod — hybrid media loader
# Lists images AND videos from input/ and output/ in one picker.
# Image files -> IMAGE output; video files -> VIDEO output plus one
# extracted frame on the IMAGE output (see frame_index).

import os

import numpy as np
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

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
_VIDEO_EXT = (".mp4", ".webm", ".mov")
_MAX_FILES = 2000

_SOURCES = (("input", folder_paths.get_input_directory),
            ("output", folder_paths.get_output_directory))


def _media_files():
    """Relative paths of image/video files in input/ and output/, tagged."""
    files = []
    for tag, get_dir in _SOURCES:
        root_dir = get_dir()
        if not os.path.isdir(root_dir):
            continue
        for base, _, names in os.walk(root_dir):
            for name in sorted(names):
                if name.lower().endswith(_IMAGE_EXT + _VIDEO_EXT):
                    rel = os.path.relpath(os.path.join(base, name), root_dir)
                    files.append(f"{rel} [{tag}]")
                    if len(files) >= _MAX_FILES:
                        return files
    return files


def _resolve(name):
    """'clip.mp4 [output]' or uploaded 'clip.mp4' -> (abs path, tag)."""
    for tag, get_dir in _SOURCES:
        suffix = f" [{tag}]"
        if name.endswith(suffix):
            path = os.path.join(get_dir(), name[: -len(suffix)])
            if os.path.isfile(path):
                return path, tag
            return None, None
    # Untagged name (e.g. set by the upload widget): input/ wins, then output/
    for tag, get_dir in _SOURCES:
        path = os.path.join(get_dir(), name)
        if os.path.isfile(path):
            return path, tag
    return None, None


def _pil_to_tensor(pil):
    arr = np.asarray(pil.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class QwenVL_LoadMedia:
    """Load an image or a video from input//output/ in a single node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media": (_media_files() or ["<no media found>"], {"image_upload": True, "video_upload": True}),
                "frame_index": ("INT", {"default": 0, "min": -1, "max": 10000, "tooltip": "For videos: which frame to emit on the IMAGE output. 0 = first, -1 = last."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "STRING")
    RETURN_NAMES = ("image", "video", "path")
    FUNCTION = "load"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, media, frame_index=0):
        path, _ = _resolve(media)
        return (path, os.path.getmtime(path)) if path else media

    def load(self, media, frame_index=0):
        path, tag = _resolve(media)
        if not path:
            raise ValueError(f"media file not found: {media}")
        lower = path.lower()

        if lower.endswith(_IMAGE_EXT):
            root = dict(input=folder_paths.get_input_directory(), output=folder_paths.get_output_directory())[tag]
            return {
                "ui": {"images": [{
                    "filename": os.path.basename(path),
                    "subfolder": os.path.dirname(os.path.relpath(path, root)),
                    "type": tag,
                }]},
                "result": (_pil_to_tensor(Image.open(path)), None, path),
            }

        if lower.endswith(_VIDEO_EXT):
            if _VIDEO_FROM_FILE is None:
                raise RuntimeError("VIDEO input type not available in this ComfyUI version")
            video = _VIDEO_FROM_FILE(path)
            frame_tensor = None
            try:
                frames = video.get_components().images
                if frames is not None and frames.shape[0]:
                    index = frame_index if frame_index >= 0 else frames.shape[0] - 1
                    frame_tensor = frames[min(index, frames.shape[0] - 1)].unsqueeze(0)
            except Exception:
                frame_tensor = None
            root = dict(input=folder_paths.get_input_directory(), output=folder_paths.get_output_directory())[tag]
            return {
                "ui": {"images": [{
                    "filename": os.path.basename(path),
                    "subfolder": os.path.dirname(os.path.relpath(path, root)),
                    "type": tag,
                }], "animated": (True,)},
                "result": (frame_tensor, video, path),
            }

        raise ValueError(f"unsupported media type: {media}")


NODE_CLASS_MAPPINGS = {
    "QwenVL_LoadMedia": QwenVL_LoadMedia,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL_LoadMedia": "🎞️ QwenVL Load Media (image+video)",
}
