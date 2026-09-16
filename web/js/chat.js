import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const STORAGE_KEY = "qwenvl.chat.v1";
const DEFAULT_STATE = {
    backend: "gguf",
    model: "",
    unloadBeforeQueue: true,
    maxTokens: 1024,
    temperature: 0.2,
    thinking: false,
    messages: [],
};

let state = loadState();
let controller = null;
let elements = {};

function loadState() {
    try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
        return { ...DEFAULT_STATE, ...(saved || {}), messages: Array.isArray(saved?.messages) ? saved.messages.slice(-20) : [] };
    } catch {
        return { ...DEFAULT_STATE };
    }
}

function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...state, messages: state.messages.slice(-20) }));
}

function createElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
}

function setStatus(text, error = false) {
    if (!elements.status) return;
    elements.status.textContent = text;
    elements.status.classList.toggle("error", error);
}

function renderMessages() {
    if (!elements.messages) return;
    elements.messages.replaceChildren();
    if (!state.messages.length) {
        elements.messages.append(createElement("div", "qwen-chat-empty", "Chiedimi di analizzare o modificare i parametri del workflow aperto."));
        return;
    }
    for (const message of state.messages) {
        const row = createElement("div", `qwen-chat-message ${message.role}`);
        row.append(createElement("div", "qwen-chat-role", message.role === "user" ? "Tu" : "Qwen"));
        row.append(createElement("div", "qwen-chat-content", message.content));
        if (message.thinking) {
            const details = createElement("details", "qwen-chat-thinking");
            details.append(createElement("summary", "", "Pensiero"));
            const pre = createElement("pre", "", message.thinking);
            details.append(pre);
            row.append(details);
        }
        elements.messages.append(row);
    }
    elements.messages.scrollTop = elements.messages.scrollHeight;
}

function serializeValue(value) {
    if (value === null || ["string", "number", "boolean"].includes(typeof value)) return value;
    if (Array.isArray(value)) return value.slice(0, 20).map(serializeValue);
    return String(value ?? "").slice(0, 1000);
}

function snapshotGraph() {
    const nodes = (app.graph?._nodes || []).slice(0, 200).map((node) => ({
        id: node.id,
        type: node.type || node.comfyClass || "",
        title: node.title || "",
        mode: node.mode ?? 0,
        widgets: (node.widgets || []).slice(0, 100).filter((widget) => widget?.name).map((widget) => ({
            name: widget.name,
            type: widget.type || typeof widget.value,
            value: serializeValue(widget.value),
            options: widget.options ? {
                min: widget.options.min,
                max: widget.options.max,
                step: widget.options.step,
                values: Array.isArray(widget.options.values) ? widget.options.values.slice(0, 200).map(serializeValue) : undefined,
            } : undefined,
        })),
    }));
    return { nodes };
}

function findNode(nodeId) {
    return app.graph?.getNodeById?.(nodeId) || (app.graph?._nodes || []).find((node) => String(node.id) === String(nodeId));
}

function normalizeWidgetValue(widget, value) {
    const current = widget.value;
    if (typeof current === "number") {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) throw new Error(`${widget.name}: valore numerico non valido`);
        const min = Number(widget.options?.min);
        const max = Number(widget.options?.max);
        let result = numeric;
        if (Number.isFinite(min)) result = Math.max(min, result);
        if (Number.isFinite(max)) result = Math.min(max, result);
        return result;
    }
    if (typeof current === "boolean") {
        if (typeof value !== "boolean") throw new Error(`${widget.name}: valore booleano non valido`);
        return value;
    }
    const values = widget.options?.values;
    if (Array.isArray(values) && !values.some((item) => item === value)) throw new Error(`${widget.name}: opzione non disponibile`);
    if (typeof value !== "string" && value !== null) throw new Error(`${widget.name}: valore testuale non valido`);
    return value ?? "";
}

async function applyActions(actions) {
    const applied = [];
    const rejected = [];
    let shouldQueue = false;
    for (const action of Array.isArray(actions) ? actions : []) {
        if (action.type === "queue_workflow") {
            shouldQueue = true;
            continue;
        }
        const node = findNode(action.node_id);
        if (!node) {
            rejected.push(`nodo ${action.node_id} inesistente`);
            continue;
        }
        if (action.type === "set_widget_value") {
            const widget = (node.widgets || []).find((item) => item.name === action.widget);
            if (!widget) {
                rejected.push(`widget ${action.widget} inesistente nel nodo ${action.node_id}`);
                continue;
            }
            try {
                const value = normalizeWidgetValue(widget, action.value);
                const previousValue = widget.value;
                widget.value = value;
                widget.callback?.(value, app.canvas, node);
                node.onWidgetChanged?.(widget.name, value, previousValue, widget);
                node.setDirtyCanvas?.(true, true);
                applied.push(`nodo ${action.node_id}: ${widget.name} = ${String(value).slice(0, 120)}`);
            } catch (error) {
                rejected.push(error.message);
            }
        } else if (action.type === "set_node_mode" && ["bypass", "enable"].includes(action.mode)) {
            node.mode = action.mode === "bypass" ? 4 : 0;
            node.setDirtyCanvas?.(true, true);
            applied.push(`nodo ${action.node_id}: ${action.mode}`);
        } else {
            rejected.push(`azione ${action.type || "sconosciuta"} non consentita`);
        }
    }
    app.graph?.setDirtyCanvas?.(true, true);
    if (shouldQueue) {
        if (state.unloadBeforeQueue) {
            const response = await api.fetchApi("/qwenvl/chat/unload", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ backend: state.backend }),
            });
            if (!response.ok) throw new Error("Impossibile scaricare il modello chat dalla memoria");
        }
        await app.queuePrompt();
        applied.push("workflow aggiunto alla coda");
    }
    return { applied, rejected };
}

function setBusy(busy) {
    elements.send.disabled = busy;
    elements.stop.disabled = !busy;
    elements.input.disabled = busy;
    elements.backend.disabled = busy;
    elements.model.disabled = busy;
    elements.maxTokens.disabled = busy;
    elements.temperature.disabled = busy;
    elements.thinking.disabled = busy;
}

async function sendMessage() {
    const content = elements.input.value.trim();
    if (!content || controller) return;
    if (!state.model) {
        setStatus("Seleziona un modello Qwen.", true);
        return;
    }
    state.messages.push({ role: "user", content });
    state.messages = state.messages.slice(-20);
    elements.input.value = "";
    saveState();
    renderMessages();
    controller = new AbortController();
    setBusy(true);
    setStatus("Qwen sta analizzando il workflow…");
    try {
        const response = await api.fetchApi("/qwenvl/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
                backend: state.backend,
                model: state.model,
                messages: state.messages,
                graph: snapshotGraph(),
                options: {
                    max_tokens: state.maxTokens,
                    temperature: state.temperature,
                    top_p: 0.9,
                    repetition_penalty: 1.05,
                    quantization: "8-bit (Balanced)",
                    attention_mode: "auto",
                    device: "auto",
                    seed: Math.floor(Math.random() * 4294967294) + 1,
                    thinking: state.thinking,
                },
            }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `Errore HTTP ${response.status}`);
        const result = await applyActions(data.actions);
        let answer = data.message || "Operazione completata.";
        if (result.applied.length) answer += `\n\nApplicato:\n- ${result.applied.join("\n- ")}`;
        if (result.rejected.length) answer += `\n\nRifiutato:\n- ${result.rejected.join("\n- ")}`;
        state.messages.push({ role: "assistant", content: answer, thinking: data.thinking || "" });
        state.messages = state.messages.slice(-20);
        saveState();
        renderMessages();
        setStatus("Pronto");
    } catch (error) {
        if (error.name === "AbortError") setStatus("Attesa interrotta. L’inferenza backend potrebbe essere ancora in corso.", true);
        else setStatus(error.message || String(error), true);
    } finally {
        controller = null;
        setBusy(false);
    }
}

async function loadModels() {
    setStatus("Caricamento modelli…");
    try {
        const response = await api.fetchApi("/qwenvl/chat/models");
        const models = await response.json();
        state.availableModels = models;
        populateModels();
        setStatus("Pronto");
    } catch (error) {
        setStatus(`Modelli non disponibili: ${error.message}`, true);
    }
}

function populateModels() {
    const models = state.availableModels?.[state.backend] || [];
    elements.model.replaceChildren();
    for (const model of models) {
        const option = createElement("option", "", model);
        option.value = model;
        elements.model.append(option);
    }
    if (!models.includes(state.model)) state.model = models[0] || "";
    elements.model.value = state.model;
    saveState();
}

function buildSidebar(container) {
    container.replaceChildren();
    const style = createElement("style");
    style.textContent = `
        .qwen-chat { height:100%; display:flex; flex-direction:column; gap:8px; padding:10px; box-sizing:border-box; color:var(--fg-color); }
        .qwen-chat-controls { display:grid; grid-template-columns:auto 1fr; gap:6px 8px; align-items:center; }
        .qwen-chat-controls label { font-size:12px; white-space:nowrap; }
        .qwen-chat-controls input[type="checkbox"] { margin-right:6px; width:auto; }
        .qwen-chat select,.qwen-chat textarea,.qwen-chat input,.qwen-chat button { background:var(--comfy-input-bg,#222); color:inherit; border:1px solid var(--border-color,#555); border-radius:6px; padding:7px; }
        .qwen-chat-messages { flex:1; min-height:120px; overflow:auto; display:flex; flex-direction:column; gap:8px; }
        .qwen-chat-message { padding:8px; border-radius:8px; white-space:pre-wrap; overflow-wrap:anywhere; background:rgba(127,127,127,.12); }
        .qwen-chat-message.user { background:rgba(50,120,180,.22); }
        .qwen-chat-role { font-size:11px; font-weight:bold; opacity:.7; margin-bottom:4px; }
        .qwen-chat-content { user-select:text; }
        .qwen-chat-input { min-height:86px; resize:vertical; }
        .qwen-chat-actions { display:flex; gap:6px; }
        .qwen-chat-actions button { flex:1; cursor:pointer; }
        .qwen-chat-memory { display:flex; align-items:center; gap:6px; font-size:12px; }
        .qwen-chat-memory input { width:auto; }
        .qwen-chat-status { min-height:18px; font-size:12px; opacity:.8; }
        .qwen-chat-status.error { color:#ff7777; opacity:1; }
        .qwen-chat-empty { opacity:.6; padding:12px; text-align:center; }
    `;
    container.append(style);
    const root = createElement("div", "qwen-chat");
    const controls = createElement("div", "qwen-chat-controls");
    elements.backend = createElement("select");
    for (const [value, label] of [["gguf", "GGUF"], ["hf", "HF / Transformers"]]) {
        const option = createElement("option", "", label);
        option.value = value;
        elements.backend.append(option);
    }
    elements.backend.value = state.backend;
    elements.model = createElement("select");
    elements.maxTokens = createElement("input");
    elements.maxTokens.type = "number";
    elements.maxTokens.min = "64";
    elements.maxTokens.max = "8192";
    elements.maxTokens.value = String(state.maxTokens);
    elements.maxTokens.title = "Maximum response tokens";
    elements.temperature = createElement("input");
    elements.temperature.type = "number";
    elements.temperature.min = "0";
    elements.temperature.max = "2";
    elements.temperature.step = "0.1";
    elements.temperature.value = String(state.temperature);
    elements.temperature.title = "Temperature";
    elements.thinking = createElement("input");
    elements.thinking.type = "checkbox";
    elements.thinking.checked = state.thinking;
    elements.thinking.title = "Show model reasoning before the answer";
    controls.append(
        createElement("label", "", "Backend"), elements.backend,
        createElement("label", "", "Modello"), elements.model,
        createElement("label", "", "Max tokens"), elements.maxTokens,
        createElement("label", "", "Temperature"), elements.temperature,
        createElement("label", "", "Thinking"), elements.thinking,
    );
    elements.messages = createElement("div", "qwen-chat-messages");
    elements.input = createElement("textarea", "qwen-chat-input");
    elements.input.placeholder = "Es: imposta 25 step nel KSampler e avvia il workflow";
    const memory = createElement("label", "qwen-chat-memory");
    elements.unload = createElement("input");
    elements.unload.type = "checkbox";
    elements.unload.checked = state.unloadBeforeQueue;
    memory.append(elements.unload, document.createTextNode("Scarica Qwen prima di avviare il workflow"));
    const actions = createElement("div", "qwen-chat-actions");
    elements.send = createElement("button", "", "Invia");
    elements.stop = createElement("button", "", "Stop");
    elements.stop.disabled = true;
    elements.clear = createElement("button", "", "Nuova chat");
    actions.append(elements.send, elements.stop, elements.clear);
    elements.status = createElement("div", "qwen-chat-status", "Inizializzazione…");
    root.append(controls, elements.messages, elements.input, memory, actions, elements.status);
    container.append(root);
    elements.backend.addEventListener("change", () => {
        state.backend = elements.backend.value;
        state.model = "";
        populateModels();
    });
    elements.model.addEventListener("change", () => {
        state.model = elements.model.value;
        saveState();
    });
    elements.maxTokens.addEventListener("change", () => {
        state.maxTokens = Math.min(8192, Math.max(64, Number(elements.maxTokens.value) || 1024));
        elements.maxTokens.value = String(state.maxTokens);
        saveState();
    });
    elements.temperature.addEventListener("change", () => {
        state.temperature = Math.min(2, Math.max(0, Number(elements.temperature.value) || 0));
        elements.temperature.value = String(state.temperature);
        saveState();
    });
    elements.unload.addEventListener("change", () => {
        state.unloadBeforeQueue = elements.unload.checked;
        saveState();
    });
    elements.thinking.addEventListener("change", () => {
        state.thinking = elements.thinking.checked;
        saveState();
    });
    elements.send.addEventListener("click", sendMessage);
    elements.stop.addEventListener("click", () => controller?.abort());
    elements.clear.addEventListener("click", () => {
        state.messages = [];
        saveState();
        renderMessages();
        setStatus("Nuova conversazione");
    });
    elements.input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
    renderMessages();
    loadModels();
}

app.registerExtension({
    name: "QwenVL.WorkflowChat",
    async setup() {
        app.extensionManager.registerSidebarTab({
            id: "qwenvl-workflow-chat",
            icon: "pi pi-comments",
            title: "Qwen Chat",
            tooltip: "Qwen Workflow Assistant",
            type: "custom",
            render: buildSidebar,
        });
    },
});
