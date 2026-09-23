import { app } from "/scripts/app.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Legacy node-name aliases
// Old workflows use node type names that have since been renamed. This hook
// rewrites them back to the current canonical name BEFORE ComfyUI checks for
// missing node types, so the alias does NOT need to be exposed in the Add Node
// menu (no duplicate entries).
// ═══════════════════════════════════════════════════════════════════════════════

const ALIASES = {
    "AILab_LivepeerRender": "QwenVL_LivepeerRender",
};

function migrateNodes(nodes) {
    if (!Array.isArray(nodes)) return;
    for (const node of nodes) {
        if (typeof node?.type === "string" && ALIASES[node.type]) {
            node.type = ALIASES[node.type];
        }
        // Subgraph nodes carry their own nested graph.
        if (node?.graph?.nodes) {
            migrateNodes(node.graph.nodes);
        }
    }
}

app.registerExtension({
    name: "QwenVL.LegacyAliases",

    async beforeConfigureGraph(graphData) {
        if (!graphData) return;
        migrateNodes(graphData.nodes);
    },
});
