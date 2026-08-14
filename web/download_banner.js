import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "QwenVL.DownloadLog",
    async setup() {
        const panel = document.createElement("div");
        panel.style.cssText = [
            "position:fixed",
            "bottom:0",
            "left:0",
            "right:0",
            "max-height:180px",
            "overflow-y:auto",
            "background:#1e1e1e",
            "color:#d4d4d4",
            "font-family:monospace",
            "font-size:12px",
            "padding:8px 12px",
            "border-top:2px solid #ff9800",
            "z-index:99999",
            "white-space:pre-wrap",
        ].join(";");

        const header = document.createElement("div");
        header.style.cssText = "font-weight:bold;color:#ff9800;margin-bottom:4px;";
        const headerText = document.createElement("span");
        header.appendChild(headerText);
        const closeBtn = document.createElement("span");
        closeBtn.textContent = " \u2715";
        closeBtn.style.cssText = "float:right;cursor:pointer;color:#888;";
        closeBtn.onclick = () => { panel.style.display = "none"; };
        header.appendChild(closeBtn);
        panel.appendChild(header);

        const logText = document.createElement("div");
        logText.style.cssText = "white-space:pre-wrap;";
        panel.appendChild(logText);

        document.body.appendChild(panel);

        let done = false;

        async function poll() {
            if (done) return;
            try {
                const resp = await fetch("/dlstatus", { cache: "no-store" });
                if (!resp.ok) return;
                const text = await resp.text();
                logText.textContent = text;

                if (text.includes("All models ready")) {
                    headerText.textContent = "\u2705 Download modelli completato";
                    header.style.color = "#4caf50";
                    panel.style.borderTopColor = "#4caf50";
                    done = true;
                    setTimeout(() => { panel.style.display = "none"; }, 20000);
                } else {
                    const m = text.match(/\[(\d+)\/(\d+)\]/);
                    if (m) {
                        headerText.textContent = "\u{1F4E5} Download modelli in corso [" + m[1] + "/" + m[2] + "] \u2014 NON scaricare manualmente";
                    } else if (text.includes("Download log not yet available")) {
                        headerText.textContent = "\u{1F4E5} In attesa del download modelli...";
                    } else {
                        headerText.textContent = "\u{1F4E5} Download modelli in corso \u2014 NON scaricare manualmente";
                    }
                }
            } catch (e) { /* endpoint not up yet */ }
        }

        setInterval(poll, 3000);
        poll();
    }
});
