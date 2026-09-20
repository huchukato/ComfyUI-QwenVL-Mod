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
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_ACTIONS = {"set_widget_value", "set_node_mode", "queue_workflow"}
MINIMAX_I2VA_BINDING = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."

BASE_SYSTEM_PROMPT = """You are Qwen Workflow Assistant inside ComfyUI. Answer the user and, only when requested, control the currently open workflow using the supplied snapshot.
The "message" text and choice labels MUST use the same language as the latest user message. Do not switch to English merely because workflow prompt text must be English.
If images or videos are loaded in the workflow inputs, their pixel content is also provided to you; refer to them when the user mentions "the image", "this image", or similar.
Return exactly one JSON object — no preamble, no text before or after it — with this schema:
{"message":"short answer to the user","actions":[{"type":"set_widget_value","node_id":1,"widget":"steps","value":25},{"type":"set_node_mode","node_id":2,"mode":"bypass"},{"type":"queue_workflow"}],"choices":[{"label":"option A","send":"the user message sent when option A is clicked"}]}
Allowed action types are set_widget_value, set_node_mode, and queue_workflow. set_node_mode accepts only bypass or enable. Never invent node IDs or widget names. Do not emit code, filesystem, shell, network, node creation, connection, deletion, or arbitrary JavaScript actions. If the request cannot be completed with the available actions, explain why in message and return an empty actions array.
When the user asks to generate N images or a batch of N, look for a "batch_size" or "batch" widget on the main generation node (the node with seed/steps/cfg/sampler_name — typically the sampler or the all-in-one generation node). Do NOT set batch_size on upscaler nodes (UpscalerTensorrt, LoadUpscalerTensorrtModel, UltimateSDUpscale, etc.) — that controls the upscaling batch, not the image count. If no batch_size exists on the generation node, explain that the workflow generates one image per run and ask if they want to queue it multiple times.
Final generated image and video prompts MUST be in English unless the user explicitly requests another prompt language. An instruction sent to an active inner preset enhancer must be a concise English action directive, because that enhancer analyzes the image and formats the final prompt. The surrounding assistant message must use the user's language.
When you set a text or prompt widget, repeat the complete new value verbatim inside message so the user can read it.
PROMPT ROUTING — inspect the widgets exposed on the SAME target node and apply the first matching case:
1. IMAGE + PRESET ENHANCER: when image pixels are provided and the image-to-video target exposes both "preset_prompt" and "passthrough", inspect the provided image pixels to understand the visible subject and how the requested motion applies, and read the target node's CURRENT "preset_prompt" value and its supplied PROMPT WRITING GUIDE. You MUST emit a set_widget_value action for the exposed prompt widget containing a concise English enhancer directive (1-3 sentences) derived from the latest substantive user request and appropriate for that selected preset: state only the requested motion/action and duration, optionally ending with "Preserve the reference image exactly and change only this action." Never copy the user message verbatim — always translate and clarify it. Skip execution-only follow-ups such as "Execute the video" and use the preceding descriptive request. Set "passthrough" to false. The inner QwenVL performs the full image analysis and formats the final prompt using the selected preset. Your pixel inspection is only for understanding the requested action: never describe or invent identity, appearance, clothing, environment, lighting, or framing, never add Picture reference lines, and never pre-format the final preset prompt.
2. PASSTHROUGH WITHOUT IMAGE: when no image pixels are provided and the target exposes "passthrough", write the complete final English prompt into the actual exposed prompt widget ("prompt", "custom_prompt", or "prompt_text") and set "passthrough" to true. A promoted outer "prompt" may feed an inner "custom_prompt"; always use the exposed name.
3. PRESET WITHOUT PASSTHROUGH: if the target exposes "preset_prompt" but not "passthrough", write a concise English intent into its exposed prompt widget so the inaccessible inner enhancer applies the preset.
4. DIRECT PROMPT: otherwise write the complete final English prompt into the actual exposed generation widget.
When the user asks you to draft or show a prompt without executing, write the fully formatted English preset prompt inside "message" for them to read and do not queue the workflow.
When the user asks to generate, run, execute, or queue, apply every required widget update and include queue_workflow in the same response. If queue_workflow is present, state that execution was started; NEVER ask whether to execute now and NEVER offer an execute choice. If you set a prompt but do not include queue_workflow, do not claim execution started: ask whether to execute and provide an execute choice.
Emit choices only when you genuinely need the user to pick between alternatives before acting (for example mutually exclusive generation modes). Put the question in message, give each choice a short label, and set send to the exact user message that should be sent back when the choice is clicked. Do not act on the ambiguous parameter until the user answers; omit choices when you can act directly. "choices" is a TOP-LEVEL field of the JSON object, a sibling of "message" and "actions" — never nest it inside an action object. Always close every bracket and brace.
When writing a prompt into a workflow, target the widget that actually feeds generation: the promoted "prompt" (or "custom_prompt"/"prompt_text") widget on the generation/subgraph node. NEVER write prompts into display/viewer nodes such as easy showAnything, ShowText, or MarkdownNote — they only preview text and change nothing."""

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
            if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            images.append((resolved.stat().st_mtime, resolved.relative_to(root).as_posix()))
        except (OSError, ValueError):
            continue
    images.sort(key=lambda item: item[0], reverse=True)
    return [f"{relative} [output]" for _, relative in images[:max(1, min(int(limit), 2000))]]


def validate_images(images):
    if not isinstance(images, list):
        return []
    result = []
    for item in images[:MAX_CHAT_IMAGES]:
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
    intent = select_workflow_intent(messages)
    actions = result.get("actions", [])
    prompt_action = next((action for action in actions if str(action.get("node_id")) == str(node_id) and action.get("widget") in {"prompt", "custom_prompt", "prompt_text"}), None)
    if prompt_action is not None and _is_enhancer_instruction(prompt_action.get("value")):
        intent = prompt_action["value"].strip()
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


def build_prompt(messages, graph, enable_thinking=False, has_images=False):
    history = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)
    snapshot = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    instruction = BASE_SYSTEM_PROMPT + _preset_guides(graph, messages, has_images) + _chat_guides_for(graph, has_images)
    instruction += THINKING_INSTRUCTION if enable_thinking else NO_THINKING_INSTRUCTION
    return f"{instruction}\n\nWORKFLOW SNAPSHOT:\n{snapshot}\n\nCONVERSATION:\n{history}\n\nJSON RESPONSE:"


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

    def chat(self, backend, model_name, messages, graph, options, images=None):
        messages = validate_messages(messages)
        graph = validate_graph(graph)
        images = validate_images(images or [])
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
        prompt = build_prompt(messages, graph, enable_thinking, bool(images))
        with self._lock:
            text = self._generate(backend, model_name, prompt, options, enable_thinking, images)
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
        retry_prompt = build_prompt(retry_messages, graph, enable_thinking, bool(images))
        with self._lock:
            retry_text = self._generate(backend, model_name, retry_prompt, options, enable_thinking, images)
        retry_result = parse_model_response(retry_text)
        if retry_result.pop("parsed"):
            retry_result = enforce_image_enhancer_routing(retry_result, graph, messages, bool(images))
            return enforce_image_reference_bindings(retry_result, graph, bool(images))
        return result

    def _generate(self, backend, model_name, prompt, options, enable_thinking, images):
        if backend == "hf":
            return self._chat_hf(model_name, prompt, options, enable_thinking, images)
        return self._chat_gguf(model_name, prompt, options, enable_thinking, images)

    def _chat_hf(self, model_name, prompt, options, enable_thinking=False, images=None):
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
        return instance.generate(
            prompt, image, image2, 1,
            int(options.get("max_tokens", 1024)),
            float(options.get("temperature", 0.2)),
            float(options.get("top_p", 0.9)),
            1,
            float(options.get("repetition_penalty", 1.05)),
            model_name=model_name,
            enable_thinking=enable_thinking,
        )

    def _chat_gguf(self, model_name, prompt, options, enable_thinking=False, images=None):
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
        valid = validate_images(images or [])
        images_b64 = [base64.b64encode(data).decode("ascii") for data in valid]
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
