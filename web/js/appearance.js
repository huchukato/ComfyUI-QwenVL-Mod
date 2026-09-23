import { app } from "/scripts/app.js";

// ═══════════════════════════════════════════════════════════════════════════════
// QwenVL-Mod node branding — inspired by ComfyUI-Pixaroma's brand extension.
// Each logical family gets a recognisable title/body colour so big workflows
// are easy to read at a glance. User colour choices and workflow-saved colours
// are always preserved (guard with !this.color / !this.bgcolor).
// ═══════════════════════════════════════════════════════════════════════════════

const COLOR_THEMES = {
    Vision:       { nodeColor: "#6d28d9", nodeBgColor: "#2e1065", width: 360 }, // HF vision
    VisionGGUF:   { nodeColor: "#0d9489", nodeBgColor: "#134e4a", width: 360 }, // GGUF / unified
    Enhancer:     { nodeColor: "#d97706", nodeBgColor: "#451a03", width: 360 }, // prompt enhancer
    Livepeer:     { nodeColor: "#2563eb", nodeBgColor: "#1e3a8a", width: 340 }, // agent render
    Utils:        { nodeColor: "#475569", nodeBgColor: "#1e293b", width: 300 }, // load media, tools
};

const NODE_COLORS = {
    // HF vision
    "AILab_QwenVL": "Vision",
    "AILab_QwenVL_Advanced": "Vision",

    // GGUF / unified
    "AILab_QwenVL_GGUF": "VisionGGUF",
    "AILab_QwenVL_GGUF_Advanced": "VisionGGUF",
    "QwenVL_Unified": "VisionGGUF",
    "QwenVL_Unified_Advanced": "VisionGGUF",

    // Prompt enhancers
    "AILab_QwenVL_PromptEnhancer": "Enhancer",
    "AILab_QwenVL_GGUF_PromptEnhancer": "Enhancer",

    // Livepeer agent render
    "QwenVL_LivepeerRender": "Livepeer",

    // Utils / media
    "QwenVL_LoadMedia": "Utils",
    "StorySplitNode": "Utils",
    "VRAMCleanup": "Utils",
};

function applyTheme(node, theme) {
    if (!theme) return;
    if (theme.nodeColor && !node.color) {
        node.color = theme.nodeColor;
    }
    if (theme.nodeBgColor && !node.bgcolor) {
        node.bgcolor = theme.nodeBgColor;
    }
    if (theme.width) {
        node.size = node.size || [theme.width, 80];
        node.size[0] = Math.max(node.size[0], theme.width);
    }
}

app.registerExtension({
    name: "QwenVL.BrandColors",

    beforeRegisterNodeDef(nodeType, nodeData) {
        const cls = nodeData?.name;
        if (!cls || !NODE_COLORS.hasOwnProperty(cls)) return;
        const themeKey = NODE_COLORS[cls];
        const theme = COLOR_THEMES[themeKey];

        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = orig?.apply(this, arguments);
            applyTheme(this, theme);
            return ret;
        };
    },
});

// ═══════════════════════════════════════════════════════════════════════════════
// Shared stylesheet for QwenVL-Mod DOM pieces (chat sidebar, asset modal, ...).
// Injected once; class names are prefixed with qwen- to avoid collisions.
// ═══════════════════════════════════════════════════════════════════════════════

const STYLE_ID = "qwenvl-shared-styles";
if (!document.getElementById(STYLE_ID)) {
    const link = document.createElement("link");
    link.id = STYLE_ID;
    link.rel = "stylesheet";
    link.href = new URL("css/qwen_nodes.css", import.meta.url).href;
    document.head.appendChild(link);
}
