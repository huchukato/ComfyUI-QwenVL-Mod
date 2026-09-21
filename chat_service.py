import base64
import gc
import io
import json
import re
import sys
import threading
from pathlib import Path

from PIL import Image


MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 12000
MAX_GRAPH_NODES = 200
MAX_CHAT_IMAGES = 3
MAX_VIDEO_FRAMES = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_ACTIONS = {"set_widget_value", "set_node_mode", "queue_workflow"}
MINIMAX_I2VA_BINDING = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."

BASE_SYSTEM_PROMPT = """You are Qwen Workflow Assistant inside ComfyUI. Answer the user and, only when requested, control the open workflow using the supplied snapshot.
LANGUAGE: "message" and choice labels must mirror the LATEST user message language. Workflow prompt text must be English.
OUTPUT: return exactly one JSON object, no text outside it:
{"message":"reply","actions":[{"type":"set_widget_value","node_id":1,"widget":"steps","value":25},{"type":"queue_workflow"}],"choices":[{"label":"A","send":"text for A"}]}
Allowed actions: set_widget_value, set_node_mode (bypass/enable only), queue_workflow. Never invent node IDs or widget names. Never emit code, shell, filesystem, JS, or connection actions.
When asked to generate/run/queue, apply every required widget update and include queue_workflow in the same response. Never ask "execute now?" after queue_workflow. If you only set a prompt without queue_workflow, do not claim execution started.
Choices are top-level, only when truly ambiguous. Always close braces.
---
WORKFLOW TARGETS (pick the FIRST matching case for the node you are controlling):
1. MiniMax H3 video sampler (exposes unet_name + preset_prompt + passthrough):
   - If the request says "use Native", "Config C", "use 10Eros", "Config A" etc., FIRST update the sampler widgets, then write only the action into the "prompt" widget.
   - Config mapping:
     * Native / Config C → unet_name="minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors", steps=20, sampler_name="res_multistep", scheduler="simple", shift_video=12, shift_audio=3
     * 10Eros / Config A → unet_name="10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors", steps=8, sampler_name="euler", scheduler="simple", shift_video=6, shift_audio=3
     * Turbo LoRA / Config B → steps=8, sampler_name="euler", scheduler="simple", shift_video=6, shift_audio=3 (LoRA toggle is manual)
   - If a different duration is requested, set "value_1" to that number of seconds.
   - For the prompt: remove config words and duration. Write only a short English action description.
   - EXAMPLE: user says "generate a 5s video, use Native: rhythmic hip sway, subtle back and forth"
     Actions: value_1=5; unet_name=minimax...; steps=20; sampler_name=res_multistep; shift_video=12; shift_audio=3; prompt="rhythmic hip sway, subtle back and forth"; passthrough=false; queue_workflow.
   - EXAMPLE: user says "use 10Eros: slow caressing on thigh, static camera"
     Actions: unet_name=10Eros...; steps=8; sampler_name=euler; shift_video=6; prompt="slow caressing on thigh, static camera"; passthrough=false; queue_workflow.
   - NEVER copy "use Native", "use 10Eros", "generate", "5s video" into the prompt widget.
   - NEVER describe the image yourself (clothes, face, room, light); the inner QwenVL model will see the image and describe it. You only provide the action.
2. Livepeer Render node (type contains "Livepeer", exposes capability + duration):
   - Write one English shot-native prompt into its "prompt" widget. Update capability, duration, aspect_ratio to match the request.
   - For images select an image capability from the dropdown (flux-schnell, flux-dev, etc.); for video select a video capability. Keep custom_capability empty unless the user names a model not in the dropdown.
   - NEVER bypass the Livepeer render node or the media loader.
3. Passthrough node (has passthrough but NO image pixels): write the complete final English prompt into the actual prompt widget and set passthrough=true.
4. Direct prompt: write the complete final English prompt into the generation widget.
When you set a text/prompt widget, repeat the new value verbatim in "message" so the user sees it.
Only write prompts into real generation widgets (prompt/custom_prompt/prompt_text). NEVER write into display nodes like ShowText, easy showAnything, or MarkdownNote."""

_LT = chr(60)
_GT = chr(62)
THINK_OPEN = _LT + "think" + _GT
THINK_CLOSE = _LT + "/think" + _GT
THINKING_INSTRUCTION = " Put your step-by-step reasoning inside " + THINK_OPEN + "..." + THINK_CLOSE + " tags, then return the JSON object after the closing " + THINK_CLOSE + " tag."
NO_THINKING_INSTRUCTION = " Do not output reasoning tags; return only the JSON object."

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + NO_THINKING_INSTRUCTION


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    result = []
    for item in messages[-MAX_MESSAGES:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            raise ValueError("invalid chat message")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > MAX_MESSAGE_CHARS:
            raise ValueError("invalid chat message content")
        result.append({"role": item["role"], "content": content.strip()})
    return result


def validate_graph(graph):
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError("invalid workflow snapshot")
    nodes = graph["nodes"]
    if len(nodes) > MAX_GRAPH_NODES:
        raise ValueError(f"workflow exceeds {MAX_GRAPH_NODES} nodes")
    result = []
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), (int, str)):
            raise ValueError("invalid workflow node")
        widgets = node.get("widgets", [])
        if not isinstance(widgets, list):
            raise ValueError("invalid workflow widgets")
        safe_widgets = []
        for widget in widgets[:100]:
            if not isinstance(widget, dict) or not isinstance(widget.get("name"), str):
                continue
            value = widget.get("value")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                value = str(value)[:1000]
            options = widget.get("options") if isinstance(widget.get("options"), dict) else {}
            values = options.get("values")
            safe_widgets.append({
                "name": widget["name"][:200],
                "type": str(widget.get("type", ""))[:100],
                "value": value[:4000] if isinstance(value, str) else value,
                "options": {
                    "min": options.get("min") if isinstance(options.get("min"), (int, float)) else None,
                    "max": options.get("max") if isinstance(options.get("max"), (int, float)) else None,
                    "values": [str(item)[:500] for item in values[:200]] if isinstance(values, list) else None,
                },
            })
        result.append({
            "id": node["id"],
            "type": str(node.get("type", ""))[:200],
            "title": str(node.get("title", ""))[:200],
            "mode": node.get("mode", 0),
            "widgets": safe_widgets,
        })
    return {"nodes": result}


def validate_actions(actions):
    if not isinstance(actions, list):
        return []
    result = []
    for action in actions[:50]:
        if not isinstance(action, dict) or action.get("type") not in ALLOWED_ACTIONS:
            continue
        action_type = action["type"]
        if action_type == "queue_workflow":
            result.append({"type": action_type})
        elif action_type == "set_node_mode" and isinstance(action.get("node_id"), (int, str)) and action.get("mode") in {"bypass", "enable"}:
            result.append({"type": action_type, "node_id": action["node_id"], "mode": action["mode"]})
        elif action_type == "set_widget_value" and isinstance(action.get("node_id"), (int, str)) and isinstance(action.get("widget"), str) and len(action["widget"]) <= 200:
            value = action.get("value")
            if value is None or isinstance(value, (str, int, float, bool)):
                result.append({"type": action_type, "node_id": action["node_id"], "widget": action["widget"], "value": value})
    return result


def validate_choices(choices):
    if not isinstance(choices, list):
        return []
    result = []
    for item in choices[:4]:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        send = item.get("send")
        if isinstance(label, str) and isinstance(send, str) and 0 < len(label) <= 80 and 0 < len(send) <= 500:
            result.append({"label": label.strip(), "send": send.strip()})
    return result


def list_output_images(output_dir, limit=500):
    root = Path(output_dir).resolve()
    if not root.is_dir():
        return []
    images = []
    for path in root.rglob("*"):
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov"}:
                continue
            images.append((resolved.stat().st_mtime, resolved.relative_to(root).as_posix()))
        except (OSError, ValueError):
            continue
    images.sort(key=lambda item: item[0], reverse=True)
    return [f"{relative} [output]" for _, relative in images[:max(1, min(int(limit), 2000))]]


def validate_images(images, limit=MAX_CHAT_IMAGES):
    if not isinstance(images, list):
        return []
    result = []
    for item in images[:limit]:
        if not isinstance(item, str):
            continue
        try:
            data = base64.b64decode(item, validate=True)
        except (ValueError, TypeError):
            continue
        if len(data) > MAX_IMAGE_BYTES:
            continue
        result.append(data)
    return result


def _decode_images(image_data_list):
    result = []
    for data in image_data_list:
        try:
            image = Image.open(io.BytesIO(data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            result.append(image)
        except Exception:
            continue
    return result


def extract_thinking(text):
    text = (text or "").strip()
    thinking = ""
    pattern = re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        text = re.sub(pattern, "", text, count=1, flags=re.DOTALL).strip()
    return thinking, text


def parse_model_response(text):
    thinking, text = extract_thinking(text)
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            message = data.get("message", "")
            return {
                "thinking": thinking,
                "message": message if isinstance(message, str) else str(message),
                "actions": validate_actions(data.get("actions", [])),
                "choices": validate_choices(data.get("choices", [])),
                "parsed": True,
            }
    message_fallback = text or "The model returned an empty response."
    # Salvage the message field from malformed JSON (e.g. unclosed brackets).
    salvage = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if salvage:
        try:
            message_fallback = json.loads('"' + salvage.group(1) + '"')
        except (TypeError, json.JSONDecodeError):
            pass
    return {"thinking": thinking, "message": message_fallback, "actions": [], "choices": [], "parsed": False}


_COMMAND_FILLERS = {
    "a", "adesso", "ahead", "avanti", "bene", "certo", "con", "coda", "di", "favore", "gia", "già",
    "grazie", "i", "il", "in", "it", "la", "le", "lo", "now", "ok", "okay", "ora", "per", "perfetto",
    "please", "precedente", "previous", "pure", "same", "si", "sì", "stesso", "subito", "sure",
    "thanks", "the", "this", "un", "una", "va", "with", "yes",
}
_COMMAND_OBJECTS = {"generation", "generazione", "job", "prompt", "queue", "task", "video", "workflow"}
_COMMAND_VERBS = {
    "avvia", "avvialo", "avviare", "cambia", "change", "conferma", "confermo", "confirm", "crea",
    "create", "do", "edit", "esegui", "eseguilo", "eseguire", "execute", "fallo", "genera",
    "generalo", "generate", "go", "lancia", "lancialo", "launch", "metti", "mettilo", "modifica",
    "modify", "procedi", "proceed", "queue", "run", "start", "vai",
}
_CONFIRM_ONLY = {"bene", "certo", "confermo", "grazie", "ok", "okay", "perfetto", "si", "sì", "sure", "va", "yes"}
_PREFORMATTED_PROMPT_MARKERS = (
    "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music",
    "<picture", "for the target video", "[shot",
)


def _normalize_command(text):
    normalized = re.sub(r"[^\w\s]", "", (text or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _is_execution_only(text):
    """True when the message is only a run/confirm command with no scene content."""
    words = _normalize_command(text).split()
    if not words or len(words) > 8:
        return False
    if all(word in _CONFIRM_ONLY for word in words):
        return True
    vocabulary = _COMMAND_FILLERS | _COMMAND_OBJECTS | _COMMAND_VERBS
    return any(word in _COMMAND_VERBS for word in words[:2]) and all(word in vocabulary for word in words)


def _is_enhancer_instruction(value):
    """True when a model-written prompt action is a usable action directive —
    not a bare confirmation and not a pre-formatted preset prompt."""
    if not isinstance(value, str) or not value.strip() or len(value) > 1500:
        return False
    lowered = value.lower()
    if any(marker in lowered for marker in _PREFORMATTED_PROMPT_MARKERS):
        return False
    return not _is_execution_only(value)


def select_workflow_intent(messages):
    intent = None
    last_user = None
    for item in messages or []:
        if item.get("role") != "user" or not isinstance(item.get("content"), str):
            continue
        content = item["content"].strip()
        if not content:
            continue
        last_user = content
        if not _is_execution_only(content):
            intent = content
    return intent if intent is not None else (last_user or "")


_CONFIG_PHRASE = re.compile(
    r"\b(?:use|using|with|in|switch(?:ing)?\s+to|usa|metti|passa\s+a)\s+"
    r"(?:the\s+)?(?:native|10\s*eros(?:-max)?|eros|turbo(?:\s*lora)?|config\s*[abc]|sol[-\s]?attn)"
    r"(?:\s+(?:mode|config|preset|model|version))?",
    re.IGNORECASE,
)
_GENERATION_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:generate|creates?|makes?|render|animate|produce|do|genera|crea|fai)\s+"
    r"(?:a\s+|an\s+|the\s+|this\s+|me\s+|one\s+|un\s+|una\s+|il\s+)?\s*"
    r"\d*\s*(?:sec(?:ond)?s?|s)?\s*"
    r"(?:second\s+|new\s+)?(?:video|clip|animation|scene)?\s*"
    r"(?:of|with|showing|where|di|con)?[:,]?\s*",
    re.IGNORECASE,
)
_DURATION_MENTION = re.compile(r"\b\d+\s*(?:s|sec(?:ond)?s?|secondi?)\b(?:\s*(?:video|clip|animation))?", re.IGNORECASE)
_MEDIA_WORD = re.compile(r"\b(?:video|clip|animation|scene)\b", re.IGNORECASE)


def _clean_action_directive(text):
    """Strip routing keywords and generation meta from a directive so the
    enhancer only receives the scene action (e.g. 'use native' and
    'generate a 5s video' never reach the prompt widget). Returns an empty
    string when nothing but meta remains."""
    if not isinstance(text, str):
        return ""
    cleaned = _CONFIG_PHRASE.sub("", text).strip(" ,;:-").lstrip(".")
    cleaned = _GENERATION_PREFIX.sub("", cleaned, count=1).strip(" ,;:-").lstrip(".")
    cleaned = _GENERATION_PREFIX.sub("", cleaned, count=1).strip(" ,;:-").lstrip(".")
    cleaned = _DURATION_MENTION.sub("", cleaned)
    cleaned = _MEDIA_WORD.sub("", cleaned)
    cleaned = re.sub(r"\s*[,;:]\s*", ", ", cleaned)
    cleaned = re.sub(r"(?:,\s*){2,}", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).lstrip(" ,;:.-").rstrip(" ,;:-")
    return cleaned


def enforce_image_enhancer_routing(result, graph, messages, has_images):
    if not has_images or not any(action.get("type") == "queue_workflow" for action in result.get("actions", [])):
        return result
    candidates = []
    for node in graph.get("nodes", []):
        widgets = {widget.get("name") for widget in node.get("widgets", []) if isinstance(widget, dict)}
        prompt_widget = next((name for name in ("prompt", "custom_prompt", "prompt_text") if name in widgets), None)
        title = f'{node.get("title", "")} {node.get("type", "")}'.lower()
        if prompt_widget and {"preset_prompt", "passthrough"}.issubset(widgets) and "image to video" in title:
            candidates.append((node, prompt_widget))
    targeted = {
        str(action.get("node_id"))
        for action in result.get("actions", [])
        if action.get("type") == "set_widget_value" and action.get("widget") in {"prompt", "custom_prompt", "prompt_text", "preset_prompt", "passthrough"}
    }
    selected = [(node, widget) for node, widget in candidates if str(node.get("id")) in targeted]
    if len(selected) != 1:
        if len(candidates) != 1:
            return result
        selected = candidates
    node, prompt_widget = selected[0]
    node_id = node.get("id")
    intent = _clean_action_directive(select_workflow_intent(messages))
    actions = result.get("actions", [])
    prompt_action = next((action for action in actions if str(action.get("node_id")) == str(node_id) and action.get("widget") in {"prompt", "custom_prompt", "prompt_text"}), None)
    if prompt_action is not None and _is_enhancer_instruction(prompt_action.get("value")):
        intent = _clean_action_directive(prompt_action["value"].strip())
    if not intent:
        # Only config keywords (e.g. "use native") — leave the existing prompt
        if prompt_action is not None:
            actions.remove(prompt_action)
    else:
        if prompt_action is None:
            queue_index = next((index for index, action in enumerate(actions) if action.get("type") == "queue_workflow"), len(actions))
            prompt_action = {"type": "set_widget_value", "node_id": node_id}
            actions.insert(queue_index, prompt_action)
        prompt_action.update({"widget": prompt_widget, "value": intent})
    passthrough_action = next((action for action in actions if str(action.get("node_id")) == str(node_id) and action.get("widget") == "passthrough"), None)
    if passthrough_action:
        passthrough_action["value"] = False
    else:
        queue_index = next((index for index, action in enumerate(actions) if action.get("type") == "queue_workflow"), len(actions))
        actions.insert(queue_index, {"type": "set_widget_value", "node_id": node_id, "widget": "passthrough", "value": False})
    if intent:
        result["message"] = f'{result.get("message", "").rstrip()}\n\nWorkflow enhancer instruction:\n{intent}'.strip()
    return result


def enforce_image_reference_bindings(result, graph, has_images):
    if not has_images:
        return result
    nodes = {str(node.get("id")): node for node in graph.get("nodes", [])}
    # Use the passthrough value that the action set will write, not the snapshot's
    # old widget value, so a fresh passthrough=false does not get a binding line.
    passthrough_values = {
        str(action.get("node_id")): action.get("value")
        for action in result.get("actions", [])
        if action.get("type") == "set_widget_value" and action.get("widget") == "passthrough"
    }
    amended = []
    for action in result.get("actions", []):
        if action.get("type") != "set_widget_value" or action.get("widget") not in {"prompt", "custom_prompt", "prompt_text"}:
            continue
        value = action.get("value")
        node = nodes.get(str(action.get("node_id")))
        if not isinstance(value, str) or not node or MINIMAX_I2VA_BINDING in value:
            continue
        widgets = {widget.get("name"): widget.get("value") for widget in node.get("widgets", []) if isinstance(widget, dict)}
        preset = str(widgets.get("preset_prompt", ""))
        title = f'{node.get("title", "")} {node.get("type", "")}'.lower()
        passthrough = passthrough_values.get(str(action.get("node_id")), widgets.get("passthrough") is True)
        if "minimax h3 nsfw (" not in preset.lower() or "image to video" not in title or not passthrough:
            continue
        action["value"] = f"{MINIMAX_I2VA_BINDING}\n\n{value.lstrip()}"
        amended.append(action["value"])
    if amended:
        result["message"] = f'{result.get("message", "").rstrip()}\n\nFinal prompt sent to workflow:\n{amended[-1]}'.strip()
    return result


def _has_image_enhancer_target(graph):
    """True if the workflow exposes an image-to-video node that has a preset
    enhancer (preset_prompt + passthrough + a prompt widget). In that case the
    chat must not write a fully formatted video prompt itself."""
    for node in graph.get("nodes", []):
        widgets = {widget.get("name"): widget.get("value") for widget in node.get("widgets", []) if isinstance(widget, dict)}
        prompt_widget = next((name for name in ("prompt", "custom_prompt", "prompt_text") if name in widgets), None)
        if not prompt_widget or "preset_prompt" not in widgets or "passthrough" not in widgets:
            continue
        title = f'{node.get("title", "")} {node.get("type", "")}'.lower()
        if "image to video" in title:
            return True
    return False


def _preset_guides(graph, messages, has_images=False):
    """Collect prompt-writing guides for presets selected in the workflow's
    widgets or named in the last user message.

    When an image-enhancer target exists, the chat is only the first stage of a
    two-stage pipeline: it must write a concise action directive, while the
    inner QwenVL node applies the preset and formats the final prompt. Full
    format guides for video presets are therefore suppressed to avoid confusing
    the chat into outputting a complete MiniMax/LTX/Wan prompt."""
    module = sys.modules.get("AILab_QwenVL")
    guides = getattr(module, "SYSTEM_PROMPTS", None) or {}
    if not guides:
        return ""
    wanted = set()
    for node in graph.get("nodes", []):
        for widget in node.get("widgets", []):
            if isinstance(widget, dict) and widget.get("value") in guides:
                wanted.add(widget["value"])
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    wanted.update(name for name in guides if name in last_user)
    # A duration mention (e.g. "10 seconds", "10s", "10 secondi") selects the
    # preset variants for that length, e.g. "MiniMax H3 NSFW (10s)".
    for match in re.finditer(r"(\d+)\s*(?:s|sec|secondi|seconds)\b", last_user, re.IGNORECASE):
        wanted.update(name for name in guides if f"({match.group(1)}s)" in name)
    if not wanted:
        return ""
    # Suppress full-format video guides when the chat must only feed the enhancer.
    if has_images and _has_image_enhancer_target(graph):
        video_prefixes = ("🎬 MiniMax", "🎞️ MiniMax", "🔄 MiniMax", "🎥 LTX", "🔀 LTX", "🎵 LTX", "📹 Wan", "🔄 Wan", "📖 Wan")
        wanted = {name for name in wanted if not any(name.startswith(prefix) for prefix in video_prefixes)}
        if not wanted:
            return ""
    parts = "\n\n".join(f"### {name}\n{guides[name]}" for name in sorted(wanted))
    return (
        "\n\nPROMPT WRITING GUIDES - when writing or editing a prompt for a node "
        "associated with one of these presets, follow the corresponding guide "
        "exactly, including its required output format. If the user asks for a "
        "different clip duration than the one the workflow is set to, also set "
        "the preset widget to the matching duration variant (if one exists) and "
        "update the workflow's duration/frame-count widgets accordingly:\n" + parts
    )


CHAT_GUIDES_PATH = Path(__file__).resolve().parent / "AILab_System_Prompts.json"


def _load_chat_guides():
    try:
        data = json.loads(CHAT_GUIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    guides = data.get("_chat_guides")
    return guides if isinstance(guides, dict) else {}


def _chat_guides_for(graph, has_images=False):
    """Inject workflow-knowledge guides whose trigger text appears in the
    snapshot's node types, titles, or widget values."""
    haystacks = []
    for node in graph.get("nodes", []):
        haystacks.append(str(node.get("type", "")))
        haystacks.append(str(node.get("title", "")))
        for widget in node.get("widgets", []):
            if isinstance(widget, dict) and isinstance(widget.get("value"), str):
                haystacks.append(widget["value"][:200])
    hay = "\n".join(haystacks).lower()
    parts = []
    for name, entry in _load_chat_guides().items():
        if not isinstance(entry, dict):
            continue
        triggers = entry.get("trigger") or []
        if isinstance(triggers, str):
            triggers = [triggers]
        text = entry.get("text")
        if isinstance(text, str) and any(str(t).lower() in hay for t in triggers):
            parts.append(f"### {name}\n{text}")
    for node in graph.get("nodes", []):
        widgets = {widget.get("name"): widget.get("value") for widget in node.get("widgets", []) if isinstance(widget, dict)}
        prompt_widget = next((name for name in ("prompt", "custom_prompt", "prompt_text") if name in widgets), None)
        if not prompt_widget or "passthrough" not in widgets:
            continue
        title = f'{node.get("title", "")} {node.get("type", "")}'.lower()
        image_enhancer = has_images and "preset_prompt" in widgets and "image to video" in title
        if image_enhancer:
            parts.append(
                f'### Exact image-enhancer target\nImage pixels are provided. Node {node.get("id")} exposes "{prompt_widget}", "preset_prompt", and "passthrough". '
                f'Inspect the provided image pixels to understand how the requested action applies. The node currently selects preset "{widgets.get("preset_prompt", "")}". '
                f'IGNORE any full prompt-writing guide for that preset above: the inner QwenVL node will use it to build the final prompt. '
                f'You MUST set node {node.get("id")} widget "{prompt_widget}" to a concise English action directive derived from the latest substantive request (skip execute-only confirmations; never copy it verbatim; never add the "For the target video..." binding line; never write integrated_multimodal_description/sections). '
                f'set node {node.get("id")} widget "passthrough" to false, then queue. The inner QwenVL must analyze the image and create the final preset prompt.'
            )
        else:
            parts.append(
                f'### Exact passthrough target\nNode {node.get("id")} exposes both "{prompt_widget}" and "passthrough". '
                f'For generation, set node {node.get("id")} widget "{prompt_widget}" to the complete final English prompt, '
                f'set node {node.get("id")} widget "passthrough" to true, then queue. Do not omit either widget action.'
            )
    if not parts:
        return ""
    return (
        "\n\nWORKFLOW KNOWLEDGE - rules that apply to this workflow; follow them "
        "when choosing actions and values:\n" + "\n\n".join(parts)
    )


def build_prompt(messages, graph, enable_thinking=False, has_images=False, has_video=False):
    history = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)
    snapshot = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    instruction = BASE_SYSTEM_PROMPT + _preset_guides(graph, messages, has_images) + _chat_guides_for(graph, has_images)
    instruction += THINKING_INSTRUCTION if enable_thinking else NO_THINKING_INSTRUCTION
    if has_video:
        instruction += "\nVIDEO INPUT: sampled frames from a video clip are attached to the latest user message. Treat them as the clip itself when the user asks to review, critique, or refine a generated video."
    return f"{instruction}\n\nWORKFLOW SNAPSHOT:\n{snapshot}\n\nCONVERSATION:\n{history}\n\nJSON RESPONSE:"


_EXPLICIT_USE = re.compile(r"^\s*(?:use|usa)\s+([a-z0-9][\w.\-]*)[.:,;\s]\s*(.*)$", re.IGNORECASE | re.DOTALL)


def _capability_result(graph, capability, prompt, text):
    """Apply capability + prompt + duration/aspect-ratio + queue on the Livepeer
    render node. `prompt` is the scene text to write; `text` is the full user
    message used for duration/ratio/language detection."""
    for node in graph.get("nodes", []):
        title = f'{node.get("title", "")} {node.get("type", "")}'.lower()
        if "livepeer" not in title:
            continue
        widgets = {w.get("name"): w for w in node.get("widgets", []) if isinstance(w, dict)}
        if "capability" not in widgets or "prompt" not in widgets:
            continue
        options = (widgets["capability"].get("options") or {}).get("values") or []
        lookup = {str(o).lower(): o for o in options}
        actions = []
        if capability.lower() in lookup:
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "capability", "value": lookup[capability.lower()]})
            if widgets.get("custom_capability", {}).get("value"):
                actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "custom_capability", "value": ""})
        elif "-" in capability and capability.lower() != "auto" and "custom_capability" in widgets:
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "custom_capability", "value": capability})
        else:
            return None
        if prompt:
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "prompt", "value": prompt[:4000]})
        duration = re.search(r"\b(\d{1,2})\s*(?:sec(?:ond)?s?|s|secondi?)\b", text, re.IGNORECASE)
        if duration and "duration" in widgets:
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "duration", "value": max(3, min(15, int(duration.group(1))))})
        ratio = re.search(r"\b(16:9|9:16|1:1|3:2|2:3|4:3|3:4|2\.35:1)\b", text)
        if ratio and "aspect_ratio" in widgets:
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "aspect_ratio", "value": ratio.group(1)})
        actions.append({"type": "queue_workflow"})
        if re.match(r"^\s*(usa|passa|metti|fai|genera|crea)\b", text, re.IGNORECASE):
            message = f"⚙️ {capability} — workflow in coda."
        else:
            message = f"⚙️ {capability} — workflow queued."
        return {"message": message, "actions": actions, "choices": []}
    return None


def _explicit_capability_request(messages, graph):
    """Deterministic `use <capability> <prompt>` shortcut: when the workflow has
    a Livepeer render node, apply capability + prompt + queue locally without
    calling the chat model. Returns None when the name is not a Livepeer
    capability (e.g. "use native" is a MiniMax config) or no render node exists."""
    last_user = _last_user_message(messages)
    match = _EXPLICIT_USE.match(last_user)
    if not match:
        return None
    capability = match.group(1).rstrip(".:,;")
    prompt = (match.group(2) or "").strip()
    return _capability_result(graph, capability, prompt, last_user)


_CONFIG_TRIGGER = re.compile(
    r"\b(?:use|usa|switch\s+to|passa\s+a|metti|set|con)\s+(?:the\s+|il\s+|la\s+)?"
    r"(native|10\s*eros(?:[-\s]?max)?|turbo(?:\s*lora)?|config\s*[abc])\b",
    re.IGNORECASE,
)

_MINIMAX_CONFIGS = {
    "native": {
        "unet_needle": "fl2va_pruned",
        "unet_fallback": "minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors",
        "values": {"steps": 20, "sampler_name": "res_multistep", "scheduler": "simple", "shift_video": 12, "shift_audio": 3},
    },
    "10eros": {
        "unet_needle": "10eros",
        "unet_fallback": "10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors",
        "values": {"steps": 8, "sampler_name": "euler", "scheduler": "simple", "shift_video": 6, "shift_audio": 3},
    },
    "turbo": {
        "unet_needle": "fl2va_pruned",
        "unet_fallback": "minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors",
        "values": {"steps": 8, "sampler_name": "euler", "scheduler": "simple", "shift_video": 6, "shift_audio": 3},
        "lora_mode": "enable",
    },
}
# Turbo LoRA must be bypassed for Native/10Eros (fused in the 10Eros checkpoint)
_MINIMAX_LORA_MODE = {"native": "bypass", "10eros": "bypass", "turbo": "enable"}


def _last_user_message(messages):
    for item in reversed(messages or []):
        if item.get("role") == "user" and isinstance(item.get("content"), str) and item["content"].strip():
            return item["content"].strip()
    return ""


def _widget_options(widget):
    return (widget.get("options") or {}).get("values") or []


def _match_option(widget, needle):
    for option in _widget_options(widget):
        if needle in str(option).lower():
            return option
    return None


def _minimax_result(graph, config_key, text):
    """Apply a MiniMax H3 sampler config + optional cleaned scene directive +
    queue on the enhancer node. `text` is the raw user message (duration is
    parsed from it, the cleaned remainder becomes the prompt directive)."""
    for node in graph.get("nodes", []):
        widgets = {w.get("name"): w for w in node.get("widgets", []) if isinstance(w, dict)}
        if not {"unet_name", "preset_prompt", "passthrough"}.issubset(widgets):
            continue
        config = _MINIMAX_CONFIGS[config_key]
        actions = []
        unet_widget = widgets["unet_name"]
        if config["unet_needle"]:
            unet = _match_option(unet_widget, config["unet_needle"]) or config["unet_fallback"]
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "unet_name", "value": unet})
        for name, value in config["values"].items():
            widget = widgets.get(name)
            if widget is None:
                continue
            options = _widget_options(widget)
            if options and str(value) not in [str(o) for o in options]:
                match = _match_option(widget, str(value).lower())
                if match is None:
                    continue
                value = match
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": name, "value": value})
        duration = re.search(r"\b(\d{1,2})\s*(?:sec(?:ond)?s?|s|secondi?)\b", text, re.IGNORECASE)
        if duration:
            seconds = int(duration.group(1))
            if "value_1" in widgets:
                actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "value_1", "value": seconds})
            preset_match = _match_option(widgets["preset_prompt"], f"({seconds}s)")
            if preset_match:
                actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "preset_prompt", "value": preset_match})
        directive = _clean_action_directive(text)
        if directive and "prompt" in widgets:
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "prompt", "value": directive})
        lora_mode = config.get("lora_mode") or _MINIMAX_LORA_MODE.get(config_key)
        prefix = f'{node["id"]}:'
        if lora_mode:
            target_mode = 0 if lora_mode == "enable" else 4
            for inner in graph.get("nodes", []):
                inner_id = str(inner.get("id", ""))
                if not inner_id.startswith(prefix) or "lora" not in str(inner.get("type", "")).lower():
                    continue
                if inner.get("mode", 0) != target_mode:
                    actions.append({"type": "set_node_mode", "node_id": inner["id"], "mode": lora_mode})
        actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "passthrough", "value": False})
        actions.append({"type": "queue_workflow"})
        label = {"native": "Native", "10eros": "10Eros", "turbo": "Turbo LoRA"}[config_key]
        if re.match(r"^\s*(usa|passa|metti|fai|genera|crea)\b", text, re.IGNORECASE):
            message = f"⚙️ MiniMax H3 → {label}. Workflow in coda."
        else:
            message = f"⚙️ MiniMax H3 → {label}. Workflow queued."
        return {"message": message, "actions": actions, "choices": []}
    return None


def _explicit_minimax_request(messages, graph):
    """Deterministic MiniMax config switch ("use native/10Eros/turbo", "config A/B/C"):
    applies sampler widgets + duration + cleaned action locally, no LLM call."""
    last_user = _last_user_message(messages)
    trigger = _CONFIG_TRIGGER.search(last_user)
    if not trigger:
        return None
    raw = trigger.group(1).lower()
    if "native" in raw or raw.rstrip() == "config c":
        config_key = "native"
    elif "eros" in raw or raw.rstrip() == "config a":
        config_key = "10eros"
    elif "turbo" in raw or raw.rstrip() == "config b":
        config_key = "turbo"
    else:
        return None
    return _minimax_result(graph, config_key, last_user)


class ChatRuntime:
    def __init__(self):
        self._instances = {}
        self._lock = threading.Lock()

    def models(self):
        hf = sys.modules.get("AILab_QwenVL")
        gguf = sys.modules.get("AILab_QwenVL_GGUF")
        # HF_ALL_MODELS includes both VL and text-only models (Qwen3.5/3.8).
        hf_models = sorted((getattr(hf, "HF_ALL_MODELS", {}) or {}).keys()) if hf else []
        gguf_models = sorted(((getattr(gguf, "GGUF_VL_CATALOG", {}) or {}).get("models") or {}).keys()) if gguf else []
        return {"hf": hf_models, "gguf": gguf_models}

    def chat(self, backend, model_name, messages, graph, options, images=None, video=None, directives=None):
        messages = validate_messages(messages)
        graph = validate_graph(graph)
        images = validate_images(images or [])
        video = validate_images(video or [], MAX_VIDEO_FRAMES)
        directives = directives if isinstance(directives, dict) else {}
        sel_capability = str(directives.get("capability") or "auto")
        sel_config = str(directives.get("config") or "auto")
        sel_text = str(directives.get("text") or "")
        explicit = None
        if sel_capability != "auto":
            explicit = _capability_result(graph, sel_capability, sel_text, sel_text)
        if explicit is None and sel_config in _MINIMAX_CONFIGS:
            explicit = _minimax_result(graph, sel_config, sel_text)
        if explicit is None:
            explicit = _explicit_capability_request(messages, graph) or _explicit_minimax_request(messages, graph)
        if explicit is not None:
            return explicit
        available = self.models().get(backend)
        if available is None:
            raise ValueError("backend must be hf or gguf")
        if not available:
            raise ValueError(f"no models available for backend '{backend}'")
        if not model_name:
            model_name = available[0]
        elif model_name not in available:
            raise ValueError(f"unknown model '{model_name}'")
        enable_thinking = bool(options.get("thinking", False))
        options = dict(options)
        if enable_thinking:
            # Thinking consumes tokens before the JSON reply; a small budget
            # truncates the response after the reasoning and no JSON is emitted.
            options["max_tokens"] = max(int(options.get("max_tokens", 1024)), 4096)
        prompt = build_prompt(messages, graph, enable_thinking, bool(images), bool(video))
        with self._lock:
            text = self._generate(backend, model_name, prompt, options, enable_thinking, images, video)
        result = parse_model_response(text)
        if result.pop("parsed"):
            result = enforce_image_enhancer_routing(result, graph, messages, bool(images))
            return enforce_image_reference_bindings(result, graph, bool(images))
        # The model ignored the JSON protocol (e.g. a conversational preamble
        # like "Generating the prompt…" and then stopped). Retry once, echoing
        # the bad reply back with a strict reminder.
        retry_messages = messages + [
            {"role": "assistant", "content": (text or "")[:2000]},
            {"role": "user", "content": "Your reply was not a JSON object. Return ONLY the JSON object described in the system instructions — no preamble, no extra text."},
        ]
        retry_prompt = build_prompt(retry_messages, graph, enable_thinking, bool(images), bool(video))
        with self._lock:
            retry_text = self._generate(backend, model_name, retry_prompt, options, enable_thinking, images, video)
        retry_result = parse_model_response(retry_text)
        if retry_result.pop("parsed"):
            retry_result = enforce_image_enhancer_routing(retry_result, graph, messages, bool(images))
            return enforce_image_reference_bindings(retry_result, graph, bool(images))
        return result

    def _generate(self, backend, model_name, prompt, options, enable_thinking, images, video=None):
        if backend == "hf":
            return self._chat_hf(model_name, prompt, options, enable_thinking, images, video)
        return self._chat_gguf(model_name, prompt, options, enable_thinking, images, video)

    def _chat_hf(self, model_name, prompt, options, enable_thinking=False, images=None, video=None):
        module = sys.modules["AILab_QwenVL"]
        instance = self._instances.get("hf")
        if instance is None:
            instance = module.QwenVLBase()
            self._instances["hf"] = instance
        instance.load_model(
            model_name,
            options.get("quantization", module.Quantization.Q8.value),
            options.get("attention_mode", "auto"),
            False,
            options.get("device", "auto"),
            True,
        )
        pil_images = _decode_images(images or [])
        image = pil_images[0] if len(pil_images) > 0 else None
        image2 = pil_images[1] if len(pil_images) > 1 else None
        pil_frames = _decode_images(video or [])
        return instance.generate(
            prompt, image, image2, len(pil_frames) or 1,
            int(options.get("max_tokens", 1024)),
            float(options.get("temperature", 0.2)),
            float(options.get("top_p", 0.9)),
            1,
            float(options.get("repetition_penalty", 1.05)),
            model_name=model_name,
            video=pil_frames or None,
            enable_thinking=enable_thinking,
        )

    def _chat_gguf(self, model_name, prompt, options, enable_thinking=False, images=None, video=None):
        module = sys.modules["AILab_QwenVL_GGUF"]
        instance = self._instances.get("gguf")
        if instance is None:
            instance = module.QwenVLGGUFBase()
            self._instances["gguf"] = instance
        instance._load_model(
            model_name,
            options.get("device", "auto"),
            options.get("ctx"),
            options.get("n_batch"),
            options.get("gpu_layers"),
            options.get("image_max_tokens"),
            options.get("top_k"),
            options.get("pool_size"),
        )
        # `images`/`video` arrive already validated to bytes by chat()
        images_b64 = [base64.b64encode(data).decode("ascii") for data in (images or []) + (video or [])]
        return instance._invoke(
            SYSTEM_PROMPT,
            prompt,
            images_b64,
            int(options.get("max_tokens", 1024)),
            float(options.get("temperature", 0.2)),
            float(options.get("top_p", 0.9)),
            float(options.get("repetition_penalty", 1.05)),
            int(options.get("seed", 1)),
            model_name,
            enable_thinking=enable_thinking,
        )

    def unload(self, backend="all"):
        if backend not in {"hf", "gguf", "all"}:
            raise ValueError("backend must be hf, gguf, or all")
        with self._lock:
            targets = list(self._instances) if backend == "all" else [backend]
            for target in targets:
                instance = self._instances.pop(target, None)
                if instance is not None:
                    instance.clear()
            gc.collect()
        return targets


CHAT_RUNTIME = ChatRuntime()
