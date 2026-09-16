import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const STORAGE_KEY = "qwenvl.chat.v1";
const DEFAULT_GGUF_MODEL = "Qwen3.8-9B-heretic-uncensored.Q8_0.gguf";
const TRANSLATIONS = {
    en: {
        empty: "Ask me to analyze or modify the parameters of the open workflow.", user: "You", thinking: "Thinking",
        imagePriority: "Priority image for Qwen", remove: "Remove", invalidImage: "Select a valid image file",
        imageError: "Unable to process the image", attachedImage: "Attached image", selectModel: "Select a Qwen model.",
        loadingImages: "Loading workflow images…", analyzingAttachment: "Qwen is analyzing the attached image…",
        analyzingWorkflowImages: "Loaded {count} workflow image(s). Qwen is analyzing…", analyzingWorkflow: "Qwen is analyzing the workflow…",
        completed: "Operation completed.", applied: "Applied", rejected: "Rejected", ready: "Ready",
        aborted: "Request stopped. Backend inference may still be running.", loadingModels: "Loading models…",
        modelsUnavailable: "Models unavailable: {error}", backend: "Backend", model: "Model", maxTokens: "Max tokens",
        temperature: "Temperature", attach: "Attach image", send: "Send", repeat: "Repeat", repeatTitle: "Resend the latest user message",
        stop: "Stop", newChat: "New chat", initializing: "Initializing…", preparingImage: "Preparing image…",
        imageAttached: "Image attached: it will be used instead of workflow images", nothingToRepeat: "No message to repeat",
        newConversation: "New conversation", placeholder: "Example: set 25 steps in KSampler and run the workflow",
        unloadError: "Unable to unload the chat model from memory", workflowQueued: "workflow added to the queue",
        invalidNumber: "{widget}: invalid numeric value", invalidBoolean: "{widget}: invalid boolean value",
        unavailableOption: "{widget}: option is not available", invalidText: "{widget}: invalid text value",
        missingNode: "node {node} does not exist", missingWidget: "widget {widget} does not exist on node {node}",
        nodeValue: "node {node}: {widget} = {value}", nodeMode: "node {node}: {mode}",
        unsupportedAction: "action {action} is not allowed", unknown: "unknown",
        generateVideo: "Generate video", generateVideoSend: "Generate the video", assets: "ComfyUI Assets",
        assetsTitle: "Select a ComfyUI output", assetsLoading: "Loading output images…", assetsEmpty: "No output images found.",
        assetsError: "Unable to load ComfyUI Assets: {error}", close: "Close", selectTarget: "Select the Load Image (from Outputs) node",
        assetChatOnly: "Asset selected for Qwen. Add a Load Image (from Outputs) node to sync it with the workflow.",
        assetSynced: "Asset selected for Qwen and loaded into node {node}.",
    },
    it: {
        empty: "Chiedimi di analizzare o modificare i parametri del workflow aperto.", user: "Tu", thinking: "Pensiero",
        imagePriority: "Immagine prioritaria per Qwen", remove: "Rimuovi", invalidImage: "Seleziona un file immagine valido",
        imageError: "Impossibile elaborare l'immagine", attachedImage: "Immagine allegata", selectModel: "Seleziona un modello Qwen.",
        loadingImages: "Caricamento immagini del workflow…", analyzingAttachment: "Qwen sta analizzando l’immagine allegata…",
        analyzingWorkflowImages: "Caricate {count} immagine/i dal workflow. Qwen sta analizzando…", analyzingWorkflow: "Qwen sta analizzando il workflow…",
        completed: "Operazione completata.", applied: "Applicato", rejected: "Rifiutato", ready: "Pronto",
        aborted: "Attesa interrotta. L’inferenza backend potrebbe essere ancora in corso.", loadingModels: "Caricamento modelli…",
        modelsUnavailable: "Modelli non disponibili: {error}", backend: "Backend", model: "Modello", maxTokens: "Max tokens",
        temperature: "Temperatura", attach: "Allega immagine", send: "Invia", repeat: "Ripeti", repeatTitle: "Reinvia l'ultimo messaggio utente",
        stop: "Stop", newChat: "Nuova chat", initializing: "Inizializzazione…", preparingImage: "Preparazione dell’immagine…",
        imageAttached: "Immagine allegata: sarà usata al posto di quelle del workflow", nothingToRepeat: "Nessun messaggio da ripetere",
        newConversation: "Nuova conversazione", placeholder: "Es: imposta 25 step nel KSampler e avvia il workflow",
        unloadError: "Impossibile scaricare il modello chat dalla memoria", workflowQueued: "workflow aggiunto alla coda",
        invalidNumber: "{widget}: valore numerico non valido", invalidBoolean: "{widget}: valore booleano non valido",
        unavailableOption: "{widget}: opzione non disponibile", invalidText: "{widget}: valore testuale non valido",
        missingNode: "nodo {node} inesistente", missingWidget: "widget {widget} inesistente nel nodo {node}",
        nodeValue: "nodo {node}: {widget} = {value}", nodeMode: "nodo {node}: {mode}",
        unsupportedAction: "azione {action} non consentita", unknown: "sconosciuta",
        generateVideo: "Genera video", generateVideoSend: "Genera il video", assets: "Risorse ComfyUI",
        assetsTitle: "Seleziona un output ComfyUI", assetsLoading: "Caricamento immagini di output…", assetsEmpty: "Nessuna immagine di output trovata.",
        assetsError: "Impossibile caricare le Risorse ComfyUI: {error}", close: "Chiudi", selectTarget: "Seleziona il nodo Carica Immagine da Output",
        assetChatOnly: "Risorsa selezionata per Qwen. Aggiungi un nodo Carica Immagine da Output per sincronizzarla con il workflow.",
        assetSynced: "Risorsa selezionata per Qwen e caricata nel nodo {node}.",
    },
};
const DEFAULT_STATE = {
    language: "en",
    backend: "gguf",
    model: "",
    maxTokens: 1024,
    temperature: 0.2,
    thinking: false,
    messages: [],
};

let state = loadState();
let controller = null;
let elements = {};
let attachedImage = null;

function t(key, values = {}) {
    let text = TRANSLATIONS[state.language]?.[key] ?? TRANSLATIONS.en[key] ?? key;
    for (const [name, value] of Object.entries(values)) text = text.replaceAll(`{${name}}`, String(value));
    return text;
}

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
        elements.messages.append(createElement("div", "qwen-chat-empty", t("empty")));
        return;
    }
    for (const message of state.messages) {
        const row = createElement("div", `qwen-chat-message ${message.role}`);
        row.append(createElement("div", "qwen-chat-role", message.role === "user" ? t("user") : "Qwen"));
        row.append(createElement("div", "qwen-chat-content", message.content));
        if (Array.isArray(message.choices) && message.choices.length) {
            const choiceRow = createElement("div", "qwen-chat-choices");
            for (const choice of message.choices) {
                if (!choice || typeof choice.label !== "string" || typeof choice.send !== "string") continue;
                const button = createElement("button", "qwen-chat-choice", choice.label);
                button.addEventListener("click", () => {
                    if (controller) return;
                    elements.input.value = choice.send;
                    sendMessage();
                });
                choiceRow.append(button);
            }
            if (choiceRow.childElementCount) row.append(choiceRow);
        }
        if (message.thinking) {
            const details = createElement("details", "qwen-chat-thinking");
            details.append(createElement("summary", "", t("thinking")));
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

function collectImageInputs() {
    const inputs = [];
    for (const node of app.graph?._nodes || []) {
        const type = node.type || node.comfyClass || "";
        if (!type.toLowerCase().includes("load") && !type.toLowerCase().includes("image")) continue;
        for (const widget of node.widgets || []) {
            if (!widget?.name || typeof widget.value !== "string" || !widget.value) continue;
            if ((widget.name === "image" || widget.name === "image_url") && !widget.value.startsWith("http")) {
                if (type === "LoadImageOutput" || /\s+\[output\]$/.test(widget.value)) {
                    const asset = parseOutputAsset(widget.value);
                    inputs.push({ filename: asset.filename, node_id: node.id, type: "output", subfolder: asset.subfolder });
                } else {
                    inputs.push({ filename: widget.value, node_id: node.id, type: "input", subfolder: "" });
                }
            } else if (widget.name === "url" && widget.value.startsWith("http")) {
                inputs.push({ url: widget.value, node_id: node.id, type: "url" });
            }
        }
    }
    return inputs.slice(0, 3);
}

async function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(",")[1]);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

async function resizeImage(blob, maxSize = 1024, quality = 0.85) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const scale = Math.min(1, maxSize / Math.max(img.width, img.height));
            const canvas = document.createElement("canvas");
            canvas.width = Math.round(img.width * scale);
            canvas.height = Math.round(img.height * scale);
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            canvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
        };
        img.onerror = reject;
        img.src = URL.createObjectURL(blob);
    });
}

async function fetchWorkflowImage(input) {
    let blob;
    if (input.type === "url") {
        const response = await fetch(input.url);
        if (!response.ok) throw new Error(`Failed to fetch image URL: ${input.url}`);
        blob = await response.blob();
    } else {
        const params = new URLSearchParams({ filename: input.filename, type: input.type, subfolder: input.subfolder || "" });
        const response = await api.fetchApi(`/view?${params.toString()}`);
        if (!response.ok) throw new Error(`Failed to fetch image: ${input.filename}`);
        blob = await response.blob();
    }
    const resized = await resizeImage(blob);
    return blobToBase64(resized);
}

async function collectImagePayload() {
    if (attachedImage) return [attachedImage.base64];
    const inputs = collectImageInputs();
    if (!inputs.length) return [];
    const images = [];
    for (const input of inputs) {
        try {
            images.push(await fetchWorkflowImage(input));
        } catch (error) {
            console.warn("[QwenChat] cannot load workflow image:", error.message);
        }
    }
    return images;
}

function renderAttachment() {
    if (!elements.attachment) return;
    elements.attachment.replaceChildren();
    elements.attachment.classList.toggle("visible", Boolean(attachedImage));
    if (!attachedImage) return;
    const preview = createElement("img", "qwen-chat-attachment-preview");
    preview.src = attachedImage.previewUrl;
    preview.alt = attachedImage.name;
    const details = createElement("div", "qwen-chat-attachment-details");
    details.append(
        createElement("strong", "", attachedImage.name),
        createElement("span", "", t("imagePriority")),
    );
    elements.removeAttachment = createElement("button", "qwen-chat-attachment-remove", t("remove"));
    elements.removeAttachment.type = "button";
    elements.removeAttachment.addEventListener("click", clearAttachment);
    elements.attachment.append(preview, details, elements.removeAttachment);
}

function clearAttachment() {
    if (attachedImage?.previewUrl) URL.revokeObjectURL(attachedImage.previewUrl);
    attachedImage = null;
    if (elements.fileInput) elements.fileInput.value = "";
    renderAttachment();
}

async function attachImage(file) {
    if (!file?.type?.startsWith("image/")) throw new Error(t("invalidImage"));
    const resized = await resizeImage(file);
    if (!resized) throw new Error(t("imageError"));
    const image = {
        base64: await blobToBase64(resized),
        previewUrl: URL.createObjectURL(resized),
        name: file.name || t("attachedImage"),
    };
    if (attachedImage?.previewUrl) URL.revokeObjectURL(attachedImage.previewUrl);
    attachedImage = image;
    renderAttachment();
}

function parseOutputAsset(value) {
    const path = String(value).replace(/\s+\[output\]$/, "");
    const slash = path.lastIndexOf("/");
    const filename = slash >= 0 ? path.slice(slash + 1) : path;
    const subfolder = slash >= 0 ? path.slice(0, slash) : "";
    const params = new URLSearchParams({ filename, subfolder, type: "output" });
    const route = `/view?${params.toString()}`;
    return { value, filename, subfolder, route, url: api.apiURL ? api.apiURL(route) : `/api${route}` };
}

function loadImageOutputNodes() {
    return (app.graph?._nodes || []).filter((node) => {
        const type = node.type || node.comfyClass || "";
        return type === "LoadImageOutput" && (node.widgets || []).some((widget) => widget.name === "image");
    });
}

function closeAssets() {
    elements.assetModal?.classList.remove("visible");
}

async function useOutputAsset(asset, node = null) {
    const response = await api.fetchApi(asset.route);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const resized = await resizeImage(await response.blob());
    if (!resized) throw new Error(t("imageError"));
    if (attachedImage?.previewUrl) URL.revokeObjectURL(attachedImage.previewUrl);
    attachedImage = {
        base64: await blobToBase64(resized),
        previewUrl: URL.createObjectURL(resized),
        name: asset.filename,
    };
    if (node) {
        const widget = (node.widgets || []).find((item) => item.name === "image");
        const previousValue = widget.value;
        widget.value = asset.value;
        widget.callback?.(asset.value, app.canvas, node);
        node.onWidgetChanged?.(widget.name, asset.value, previousValue, widget);
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    }
    renderAttachment();
    closeAssets();
    setStatus(node ? t("assetSynced", { node: node.id }) : t("assetChatOnly"));
}

function renderAssetTargets(asset, nodes) {
    elements.assetGrid.replaceChildren(createElement("div", "qwen-chat-assets-heading", t("selectTarget")));
    for (const node of nodes) {
        const label = node.title && node.title !== node.type ? `${node.title} (${node.id})` : `${node.type} (${node.id})`;
        const button = createElement("button", "qwen-chat-asset-target", label);
        button.addEventListener("click", () => useOutputAsset(asset, node).catch((error) => setStatus(error.message || String(error), true)));
        elements.assetGrid.append(button);
    }
}

async function chooseOutputAsset(asset) {
    const nodes = loadImageOutputNodes();
    if (nodes.length > 1) {
        renderAssetTargets(asset, nodes);
        return;
    }
    await useOutputAsset(asset, nodes[0] || null);
}

async function openAssets() {
    elements.assetModal.classList.add("visible");
    elements.assetGrid.replaceChildren(createElement("div", "qwen-chat-assets-heading", t("assetsLoading")));
    try {
        const response = await fetch("/internal/files/output");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const values = await response.json();
        const assets = (Array.isArray(values) ? values : [])
            .filter((value) => /\.(png|jpe?g|webp)\s+\[output\]$/i.test(String(value)))
            .slice(0, 100)
            .map(parseOutputAsset);
        elements.assetGrid.replaceChildren();
        if (!assets.length) {
            elements.assetGrid.append(createElement("div", "qwen-chat-assets-heading", t("assetsEmpty")));
            return;
        }
        for (const asset of assets) {
            const button = createElement("button", "qwen-chat-asset");
            const image = createElement("img");
            image.src = asset.url;
            image.loading = "lazy";
            image.alt = asset.filename;
            button.append(image, createElement("span", "", asset.filename));
            button.addEventListener("click", () => chooseOutputAsset(asset).catch((error) => setStatus(error.message || String(error), true)));
            elements.assetGrid.append(button);
        }
    } catch (error) {
        elements.assetGrid.replaceChildren(createElement("div", "qwen-chat-assets-heading", t("assetsError", { error: error.message })));
    }
}

function findNode(nodeId) {
    return app.graph?.getNodeById?.(nodeId) || (app.graph?._nodes || []).find((node) => String(node.id) === String(nodeId));
}

function normalizeWidgetValue(widget, value) {
    const current = widget.value;
    if (typeof current === "number") {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) throw new Error(t("invalidNumber", { widget: widget.name }));
        const min = Number(widget.options?.min);
        const max = Number(widget.options?.max);
        let result = numeric;
        if (Number.isFinite(min)) result = Math.max(min, result);
        if (Number.isFinite(max)) result = Math.min(max, result);
        return result;
    }
    if (typeof current === "boolean") {
        if (typeof value !== "boolean") throw new Error(t("invalidBoolean", { widget: widget.name }));
        return value;
    }
    const values = widget.options?.values;
    if (Array.isArray(values) && !values.some((item) => item === value)) throw new Error(t("unavailableOption", { widget: widget.name }));
    if (typeof value !== "string" && value !== null) throw new Error(t("invalidText", { widget: widget.name }));
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
            rejected.push(t("missingNode", { node: action.node_id }));
            continue;
        }
        if (action.type === "set_widget_value") {
            const widget = (node.widgets || []).find((item) => item.name === action.widget);
            if (!widget) {
                rejected.push(t("missingWidget", { widget: action.widget, node: action.node_id }));
                continue;
            }
            try {
                const value = normalizeWidgetValue(widget, action.value);
                const previousValue = widget.value;
                widget.value = value;
                widget.callback?.(value, app.canvas, node);
                node.onWidgetChanged?.(widget.name, value, previousValue, widget);
                node.setDirtyCanvas?.(true, true);
                applied.push(t("nodeValue", { node: action.node_id, widget: widget.name, value: String(value).slice(0, 120) }));
            } catch (error) {
                rejected.push(error.message);
            }
        } else if (action.type === "set_node_mode" && ["bypass", "enable"].includes(action.mode)) {
            node.mode = action.mode === "bypass" ? 4 : 0;
            node.setDirtyCanvas?.(true, true);
            applied.push(t("nodeMode", { node: action.node_id, mode: action.mode }));
        } else {
            rejected.push(t("unsupportedAction", { action: action.type || t("unknown") }));
        }
    }
    app.graph?.setDirtyCanvas?.(true, true);
    if (shouldQueue) {
        const response = await api.fetchApi("/qwenvl/chat/unload", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ backend: state.backend }),
        });
        if (!response.ok) throw new Error(t("unloadError"));
        await app.queuePrompt();
        applied.push(t("workflowQueued"));
    }
    return { applied, rejected };
}

function setBusy(busy) {
    elements.send.disabled = busy;
    elements.repeat.disabled = busy;
    elements.stop.disabled = !busy;
    elements.input.disabled = busy;
    elements.backend.disabled = busy;
    elements.model.disabled = busy;
    elements.maxTokens.disabled = busy;
    elements.temperature.disabled = busy;
    elements.thinking.disabled = busy;
    elements.attach.disabled = busy;
    elements.assetsButton.disabled = busy;
    for (const button of elements.languageButtons || []) button.disabled = busy;
    if (elements.removeAttachment) elements.removeAttachment.disabled = busy;
    elements.status?.classList.toggle("busy", busy);
}

async function sendMessage() {
    const content = elements.input.value.trim();
    if (!content || controller) return;
    if (!state.model) {
        setStatus(t("selectModel"), true);
        return;
    }
    state.messages.push({ role: "user", content });
    state.messages = state.messages.slice(-20);
    elements.input.value = "";
    saveState();
    renderMessages();
    controller = new AbortController();
    setBusy(true);
    setStatus(t("loadingImages"));
    let images = [];
    try {
        images = await collectImagePayload();
    } catch (error) {
        console.warn("[QwenChat] image collection failed:", error);
    }
    if (attachedImage && images.length) {
        setStatus(t("analyzingAttachment"));
    } else if (images.length) {
        setStatus(t("analyzingWorkflowImages", { count: images.length }));
    } else {
        setStatus(t("analyzingWorkflow"));
    }
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
                images,
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
        if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
        const result = await applyActions(data.actions);
        let answer = data.message || t("completed");
        if (result.applied.length) answer += `\n\n${t("applied")}:\n- ${result.applied.join("\n- ")}`;
        if (result.rejected.length) answer += `\n\n${t("rejected")}:\n- ${result.rejected.join("\n- ")}`;
        const responseActions = Array.isArray(data.actions) ? data.actions : [];
        const promptWasSet = responseActions.some((action) => action.type === "set_widget_value" && ["prompt", "custom_prompt", "prompt_text"].includes(action.widget));
        const workflowWasQueued = responseActions.some((action) => action.type === "queue_workflow");
        let choices = Array.isArray(data.choices) ? data.choices : [];
        if (!choices.length && promptWasSet && !workflowWasQueued) {
            choices = [{ label: t("generateVideo"), send: t("generateVideoSend") }];
        }
        state.messages.push({ role: "assistant", content: answer, thinking: data.thinking || "", choices });
        state.messages = state.messages.slice(-20);
        saveState();
        renderMessages();
        setStatus(t("ready"));
    } catch (error) {
        if (error.name === "AbortError") setStatus(t("aborted"), true);
        else setStatus(error.message || String(error), true);
    } finally {
        controller = null;
        setBusy(false);
    }
}

async function loadModels() {
    setStatus(t("loadingModels"));
    try {
        const response = await api.fetchApi("/qwenvl/chat/models");
        const models = await response.json();
        state.availableModels = models;
        populateModels();
        setStatus(t("ready"));
    } catch (error) {
        setStatus(t("modelsUnavailable", { error: error.message }), true);
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
    if (!models.includes(state.model)) {
        state.model = state.backend === "gguf" && models.includes(DEFAULT_GGUF_MODEL) ? DEFAULT_GGUF_MODEL : models[0] || "";
    }
    elements.model.value = state.model;
    saveState();
}

function buildSidebar(container) {
    container.replaceChildren();
    const style = createElement("style");
    style.textContent = `
        .qwen-chat { --qwen-accent:#6d7cff; height:100%; display:flex; flex-direction:column; gap:10px; padding:12px; box-sizing:border-box; color:var(--fg-color); background:linear-gradient(180deg,rgba(109,124,255,.04),transparent 180px); }
        .qwen-chat-language { align-self:flex-end; display:flex; padding:2px; border:1px solid var(--border-color,#444); border-radius:8px; background:rgba(127,127,127,.06); }
        .qwen-chat-language button { min-width:34px; padding:4px 7px; border:0; border-radius:6px; background:transparent; font-size:10px; cursor:pointer; opacity:.55; }
        .qwen-chat-language button.active { color:#fff; background:var(--qwen-accent); opacity:1; }
        .qwen-chat-controls { display:grid; grid-template-columns:auto minmax(0,1fr); gap:7px 10px; align-items:center; padding:10px; border:1px solid var(--border-color,#444); border-radius:12px; background:rgba(127,127,127,.06); }
        .qwen-chat-controls label { font-size:11px; font-weight:600; letter-spacing:.02em; opacity:.72; white-space:nowrap; }
        .qwen-chat-controls input[type="checkbox"] { margin:0; width:auto; justify-self:start; accent-color:var(--qwen-accent); }
        .qwen-chat select,.qwen-chat textarea,.qwen-chat input,.qwen-chat button { box-sizing:border-box; background:var(--comfy-input-bg,#202124); color:inherit; border:1px solid var(--border-color,#4b4d55); border-radius:9px; padding:8px 10px; outline:none; transition:border-color .15s,background .15s,opacity .15s,transform .15s; }
        .qwen-chat select:focus,.qwen-chat textarea:focus,.qwen-chat input:focus { border-color:var(--qwen-accent); box-shadow:0 0 0 2px rgba(109,124,255,.15); }
        .qwen-chat button:not(:disabled):hover { border-color:var(--qwen-accent); background:rgba(109,124,255,.14); }
        .qwen-chat button:not(:disabled):active { transform:translateY(1px); }
        .qwen-chat button:disabled { cursor:default; opacity:.45; }
        .qwen-chat-messages { flex:1; min-height:140px; overflow:auto; display:flex; flex-direction:column; gap:12px; padding:4px 3px 8px; scrollbar-width:thin; }
        .qwen-chat-message { align-self:flex-start; max-width:88%; padding:11px 13px; border:1px solid var(--border-color,#454750); border-radius:14px 14px 14px 4px; white-space:pre-wrap; line-height:1.45; overflow-wrap:anywhere; background:rgba(127,127,127,.09); box-shadow:0 4px 14px rgba(0,0,0,.08); }
        .qwen-chat-message.user { align-self:flex-end; border-color:rgba(109,124,255,.42); border-radius:14px 14px 4px 14px; background:rgba(109,124,255,.16); }
        .qwen-chat-role { margin-bottom:5px; font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; opacity:.58; }
        .qwen-chat-content { user-select:text; }
        .qwen-chat-choices { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
        .qwen-chat-choice { cursor:pointer; font-size:12px; padding:6px 10px; border-color:rgba(109,124,255,.45); background:rgba(109,124,255,.12); }
        .qwen-chat-thinking { margin-top:9px; opacity:.82; }
        .qwen-chat-thinking summary { cursor:pointer; font-size:11px; user-select:none; }
        .qwen-chat-thinking pre { max-height:220px; overflow:auto; margin:8px 0 0; padding:9px; border-radius:8px; white-space:pre-wrap; background:rgba(0,0,0,.16); }
        .qwen-chat-input { width:100%; min-height:92px; resize:vertical; line-height:1.4; }
        .qwen-chat-attachment { display:none; align-items:center; gap:10px; padding:8px; border:1px solid rgba(109,124,255,.32); border-radius:11px; background:rgba(109,124,255,.08); }
        .qwen-chat-attachment.visible { display:flex; }
        .qwen-chat-attachment-preview { width:52px; height:52px; flex:0 0 52px; object-fit:cover; border-radius:8px; border:1px solid rgba(255,255,255,.12); }
        .qwen-chat-attachment-details { min-width:0; flex:1; display:flex; flex-direction:column; gap:3px; }
        .qwen-chat-attachment-details strong { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12px; }
        .qwen-chat-attachment-details span { font-size:10px; opacity:.62; }
        .qwen-chat-attachment-remove { flex:0 0 auto; padding:6px 8px!important; font-size:11px; }
        .qwen-chat-composer-tools { display:flex; justify-content:flex-start; gap:7px; }
        .qwen-chat-attach,.qwen-chat-assets-button { cursor:pointer; font-size:12px; }
        .qwen-chat-file { display:none; }
        .qwen-chat-assets-modal { display:none; position:fixed; inset:0; z-index:100000; align-items:center; justify-content:center; padding:24px; background:rgba(0,0,0,.72); backdrop-filter:blur(4px); }
        .qwen-chat-assets-modal.visible { display:flex; }
        .qwen-chat-assets-panel { width:min(760px,92vw); max-height:82vh; display:flex; flex-direction:column; gap:10px; padding:14px; border:1px solid var(--border-color,#4b4d55); border-radius:14px; background:var(--comfy-menu-bg,#202124); box-shadow:0 20px 60px rgba(0,0,0,.45); }
        .qwen-chat-assets-header { display:flex; align-items:center; justify-content:space-between; gap:12px; font-weight:650; }
        .qwen-chat-assets-grid { min-height:120px; overflow:auto; display:grid; grid-template-columns:repeat(auto-fill,minmax(130px,1fr)); gap:9px; padding:2px; }
        .qwen-chat-assets-heading { grid-column:1/-1; padding:18px; text-align:center; opacity:.7; }
        .qwen-chat-asset { min-width:0; display:flex; flex-direction:column; gap:6px; padding:6px!important; cursor:pointer; text-align:left; }
        .qwen-chat-asset img { width:100%; aspect-ratio:1; object-fit:cover; border-radius:6px; background:#111; }
        .qwen-chat-asset span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:10px; }
        .qwen-chat-asset-target { grid-column:1/-1; cursor:pointer; text-align:left; }
        .qwen-chat-actions { display:grid; grid-template-columns:1.35fr 1fr 1fr 1fr; gap:7px; }
        .qwen-chat-actions button { min-width:0; cursor:pointer; font-size:12px; }
        .qwen-chat-actions button:first-child { border-color:rgba(109,124,255,.58); background:rgba(109,124,255,.2); font-weight:650; }
        .qwen-chat-status { min-height:20px; padding:0 2px; font-size:11px; opacity:.72; display:flex; align-items:center; gap:7px; }
        .qwen-chat-status.busy::before { content:""; width:13px; height:13px; flex:0 0 13px; box-sizing:border-box; border:2px solid rgba(127,127,127,.35); border-top-color:var(--qwen-accent); border-radius:50%; animation:qwen-chat-spin .7s linear infinite; }
        .qwen-chat-status.error { color:#ff7777; opacity:1; }
        @keyframes qwen-chat-spin { to { transform:rotate(360deg); } }
        .qwen-chat-empty { margin:auto; max-width:280px; padding:20px; text-align:center; line-height:1.5; opacity:.55; }
    `;
    container.append(style);
    const root = createElement("div", "qwen-chat");
    const language = createElement("div", "qwen-chat-language");
    elements.languageButtons = [];
    for (const value of ["en", "it"]) {
        const button = createElement("button", value === state.language ? "active" : "", value.toUpperCase());
        button.type = "button";
        button.addEventListener("click", () => {
            if (controller || state.language === value) return;
            state.language = value;
            saveState();
            buildSidebar(container);
        });
        elements.languageButtons.push(button);
        language.append(button);
    }
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
        createElement("label", "", t("backend")), elements.backend,
        createElement("label", "", t("model")), elements.model,
        createElement("label", "", t("maxTokens")), elements.maxTokens,
        createElement("label", "", t("temperature")), elements.temperature,
        createElement("label", "", t("thinking")), elements.thinking,
    );
    elements.messages = createElement("div", "qwen-chat-messages");
    elements.input = createElement("textarea", "qwen-chat-input");
    elements.input.placeholder = t("placeholder");
    elements.attachment = createElement("div", "qwen-chat-attachment");
    const composerTools = createElement("div", "qwen-chat-composer-tools");
    elements.fileInput = createElement("input", "qwen-chat-file");
    elements.fileInput.type = "file";
    elements.fileInput.accept = "image/*";
    elements.attach = createElement("button", "qwen-chat-attach", t("attach"));
    elements.attach.type = "button";
    elements.assetsButton = createElement("button", "qwen-chat-assets-button", t("assets"));
    elements.assetsButton.type = "button";
    composerTools.append(elements.fileInput, elements.attach, elements.assetsButton);
    const actions = createElement("div", "qwen-chat-actions");
    elements.send = createElement("button", "", t("send"));
    elements.repeat = createElement("button", "", t("repeat"));
    elements.repeat.title = t("repeatTitle");
    elements.stop = createElement("button", "", t("stop"));
    elements.stop.disabled = true;
    elements.clear = createElement("button", "", t("newChat"));
    actions.append(elements.send, elements.repeat, elements.stop, elements.clear);
    elements.status = createElement("div", "qwen-chat-status", t("initializing"));
    elements.assetModal = createElement("div", "qwen-chat-assets-modal");
    const assetPanel = createElement("div", "qwen-chat-assets-panel");
    const assetHeader = createElement("div", "qwen-chat-assets-header");
    const assetClose = createElement("button", "", t("close"));
    elements.assetGrid = createElement("div", "qwen-chat-assets-grid");
    assetHeader.append(createElement("span", "", t("assetsTitle")), assetClose);
    assetPanel.append(assetHeader, elements.assetGrid);
    elements.assetModal.append(assetPanel);
    root.append(language, controls, elements.messages, elements.input, elements.attachment, composerTools, actions, elements.status, elements.assetModal);
    container.append(root);
    renderAttachment();
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
    elements.thinking.addEventListener("change", () => {
        state.thinking = elements.thinking.checked;
        saveState();
    });
    elements.attach.addEventListener("click", () => elements.fileInput.click());
    elements.assetsButton.addEventListener("click", openAssets);
    assetClose.addEventListener("click", closeAssets);
    elements.assetModal.addEventListener("click", (event) => {
        if (event.target === elements.assetModal) closeAssets();
    });
    elements.fileInput.addEventListener("change", async () => {
        const file = elements.fileInput.files?.[0];
        if (!file) return;
        elements.attach.disabled = true;
        setStatus(t("preparingImage"));
        try {
            await attachImage(file);
            setStatus(t("imageAttached"));
        } catch (error) {
            clearAttachment();
            setStatus(error.message || String(error), true);
        } finally {
            elements.attach.disabled = false;
        }
    });
    elements.send.addEventListener("click", sendMessage);
    elements.repeat.addEventListener("click", () => {
        if (controller) return;
        const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
        if (!lastUser) {
            setStatus(t("nothingToRepeat"), true);
            return;
        }
        elements.input.value = lastUser.content;
        sendMessage();
    });
    elements.stop.addEventListener("click", () => controller?.abort());
    elements.clear.addEventListener("click", () => {
        state.messages = [];
        clearAttachment();
        saveState();
        renderMessages();
        setStatus(t("newConversation"));
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
