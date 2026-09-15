import gc
import json
import re
import sys
import threading


MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 12000
MAX_GRAPH_NODES = 200
ALLOWED_ACTIONS = {"set_widget_value", "set_node_mode", "queue_workflow"}

SYSTEM_PROMPT = """You are Qwen Workflow Assistant inside ComfyUI. Answer the user and, only when requested, control the currently open workflow using the supplied snapshot.
Return exactly one JSON object with this schema:
{"message":"short answer to the user","actions":[{"type":"set_widget_value","node_id":1,"widget":"steps","value":25},{"type":"set_node_mode","node_id":2,"mode":"bypass"},{"type":"queue_workflow"}]}
Allowed action types are set_widget_value, set_node_mode, and queue_workflow. set_node_mode accepts only bypass or enable. Never invent node IDs or widget names. Do not emit code, filesystem, shell, network, node creation, connection, deletion, or arbitrary JavaScript actions. If the request cannot be completed with the available actions, explain why in message and return an empty actions array. Return JSON only."""


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


def parse_model_response(text):
    text = (text or "").strip()
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
                "message": message if isinstance(message, str) else str(message),
                "actions": validate_actions(data.get("actions", [])),
            }
    return {"message": text or "The model returned an empty response.", "actions": []}


def build_prompt(messages, graph):
    history = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)
    snapshot = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    return f"{SYSTEM_PROMPT}\n\nWORKFLOW SNAPSHOT:\n{snapshot}\n\nCONVERSATION:\n{history}\n\nJSON RESPONSE:"


class ChatRuntime:
    def __init__(self):
        self._instances = {}
        self._lock = threading.Lock()

    def models(self):
        hf = sys.modules.get("AILab_QwenVL")
        gguf = sys.modules.get("AILab_QwenVL_GGUF")
        hf_models = sorted((getattr(hf, "HF_VL_MODELS", {}) or {}).keys()) if hf else []
        gguf_models = sorted(((getattr(gguf, "GGUF_VL_CATALOG", {}) or {}).get("models") or {}).keys()) if gguf else []
        return {"hf": hf_models, "gguf": gguf_models}

    def chat(self, backend, model_name, messages, graph, options):
        messages = validate_messages(messages)
        graph = validate_graph(graph)
        available = self.models().get(backend)
        if available is None:
            raise ValueError("backend must be hf or gguf")
        if model_name not in available:
            raise ValueError("unknown model")
        prompt = build_prompt(messages, graph)
        with self._lock:
            if backend == "hf":
                text = self._chat_hf(model_name, prompt, options)
            else:
                text = self._chat_gguf(model_name, prompt, options)
        return parse_model_response(text)

    def _chat_hf(self, model_name, prompt, options):
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
        return instance.generate(
            prompt, None, None, 1,
            int(options.get("max_tokens", 1024)),
            float(options.get("temperature", 0.2)),
            float(options.get("top_p", 0.9)),
            1,
            float(options.get("repetition_penalty", 1.05)),
            model_name=model_name,
        )

    def _chat_gguf(self, model_name, prompt, options):
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
        return instance._invoke(
            SYSTEM_PROMPT,
            prompt,
            [],
            int(options.get("max_tokens", 1024)),
            float(options.get("temperature", 0.2)),
            float(options.get("top_p", 0.9)),
            float(options.get("repetition_penalty", 1.05)),
            int(options.get("seed", 1)),
            model_name,
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
