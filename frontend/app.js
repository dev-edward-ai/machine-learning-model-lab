const API_BASE_URL = "http://localhost:8000";

const form = document.getElementById("predictionForm");
const objectiveSelect = document.getElementById("objectiveSelect");
const fileInput = document.getElementById("fileInput");
const targetColumnInput = document.getElementById("targetColumn");
const trainSplitInput = document.getElementById("trainSplit");
const trainValueEl = document.getElementById("trainValue");
const testValueEl = document.getElementById("testValue");
const usePreviewToggle = document.getElementById("previewToggle");
const returnCsvToggle = document.getElementById("csvToggle");
const statusMessage = document.getElementById("statusMessage");
const metadataPanel = document.getElementById("metadataPanel");
const predictionPanel = document.getElementById("predictionPanel");
const previewSection = document.getElementById("previewSection");
const previewTable = document.getElementById("previewTable");
const downloadBtn = document.getElementById("downloadBtn");
const submitButton = form.querySelector("button[type='submit']");
const fileNameDisplay = document.getElementById("fileNameDisplay");
const uploadZone = document.getElementById("uploadZone");
const uploadLabel = document.getElementById("uploadLabel");

// Default model choices mapped from the high-level business objective.
const objectiveToModel = {
    churn: "logistic_regression",
    revenue: "linear_regression",
    segmentation: "kmeans",
    anomaly: "isolation_forest"
};

let currentDownloadUrl = null;

const toneColors = {
    info: "#8a93b2",
    success: "#5cf4d4",
    error: "#ff8fa3"
};

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!fileInput.files.length) {
        setStatus("Please choose a CSV file first.", "error");
        return;
    }
    submitButton.disabled = true;
    submitButton.textContent = "Running...";
    setStatus("Processing dataset...");

    try {
        const formData = new FormData();
        const objectiveValue = objectiveSelect ? objectiveSelect.value : "";
        if (objectiveValue) {
            formData.append("objective", objectiveValue);
            // Pick a backend model based on the chosen objective; fall back to logistic_regression.
            const mappedModel = objectiveToModel[objectiveValue] || "logistic_regression";
            formData.append("model_name", mappedModel);
        } else {
            // Backend requires model_name; use a sensible default if no objective chosen.
            formData.append("model_name", "logistic_regression");
        }
        formData.append("use_preview", usePreviewToggle.checked ? "true" : "false");
        formData.append("return_csv", returnCsvToggle.checked ? "true" : "false");
        if (targetColumnInput.value.trim()) {
            formData.append("target_column", targetColumnInput.value.trim());
        }
        if (trainSplitInput && trainSplitInput.value) {
            formData.append("train_set_size", trainSplitInput.value); // percentage for training
        }
        formData.append("file", fileInput.files[0]);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorPayload = await response.json().catch(() => ({}));
            let detail = errorPayload.detail || "Prediction failed.";
            const isObjectLike = typeof detail === "object" && detail !== null;
            if (isObjectLike) {
                // Surface full validation or server error payloads instead of [object Object]
                detail = JSON.stringify(detail);
            }
            throw new Error(detail);
        }

        const data = await response.json();
        hydrateUiWithResults(data);
        setStatus("Prediction complete.", "success");
    } catch (error) {
        console.error(error);
        setStatus(error.message || "Unexpected error during prediction.", "error");
        predictionPanel.innerHTML = "";
        metadataPanel.innerHTML = "";
        renderPreview(null);
        setDownloadLink(null, null);
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Run Prediction";
    }
});

if (fileInput) {
    const setUploadNeutral = () => {
        if (!uploadZone) return;
        uploadZone.classList.remove("border-emerald-500", "bg-emerald-500/10", "border-purple-500", "bg-purple-500/5");
        uploadZone.classList.add("border-slate-700", "bg-slate-900/50");
    };

    const setUploadSuccess = (fileName) => {
        if (uploadZone) {
            uploadZone.classList.remove("border-slate-700", "bg-slate-900/50", "border-purple-500", "bg-purple-500/5");
            uploadZone.classList.add("border-emerald-500", "bg-emerald-500/10");
        }
        if (uploadLabel) {
            uploadLabel.textContent = fileName ? `✅ Selected: ${fileName}` : "Upload CSV";
        }
        if (fileNameDisplay) {
            if (fileName) {
                fileNameDisplay.textContent = `Selected: ${fileName}`;
                fileNameDisplay.classList.remove("text-slate-500");
                fileNameDisplay.classList.add("text-purple-200");
            } else {
                fileNameDisplay.textContent = "No file selected.";
                fileNameDisplay.classList.remove("text-purple-200");
                fileNameDisplay.classList.add("text-slate-500");
            }
        }
    };

    const setUploadActive = () => {
        if (!uploadZone) return;
        uploadZone.classList.remove("border-emerald-500", "bg-emerald-500/10", "border-slate-700", "bg-slate-900/50");
        uploadZone.classList.add("border-purple-500", "bg-purple-500/5");
    };

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            const fileName = fileInput.files[0].name;
            setUploadSuccess(fileName);
        } else {
            setUploadNeutral();
            if (uploadLabel) uploadLabel.textContent = "Upload CSV";
        }
    });

    if (uploadZone) {
        uploadZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            setUploadActive();
        });

        uploadZone.addEventListener("dragleave", (e) => {
            e.preventDefault();
            setUploadNeutral();
        });

        uploadZone.addEventListener("drop", (e) => {
            e.preventDefault();
            const dt = e.dataTransfer;
            if (!dt || !dt.files || !dt.files.length) {
                setUploadNeutral();
                return;
            }
            const file = dt.files[0];
            const transfer = new DataTransfer();
            transfer.items.add(file);
            fileInput.files = transfer.files;
            setUploadSuccess(file.name);
        });
    }
}

function hydrateUiWithResults(data) {
    renderResults(data);
    renderPreview(data.preview);
    setDownloadLink(data.csv_base64, data.csv_filename);
}

function renderResults(data) {
    const insight = data.business_insights || data.business_summary || {};
    const metadata = data.metadata || {};
    const modelType = data.model_type || metadata.task_type || "";

    const intent = (objectiveSelect?.value || "").toLowerCase();
    const accent = resolveAccent(intent, modelType);
    const isAlert = ["churn", "fraud", "anomaly"].some((needle) => intent.includes(needle));
    const isGood = intent.includes("revenue") || intent.includes("sales");

    const rows = typeof metadata.rows === "number" ? metadata.rows : "-";
    const accuracy = metadata.accuracy ?? "N/A";

    predictionPanel.innerHTML = `
        <div class="bg-gradient-to-br from-gray-900 to-black border border-purple-500/20 rounded-2xl p-6 shadow-2xl">
            <div class="flex items-start gap-3">
                <div class="shrink-0 h-10 w-10 rounded-full flex items-center justify-center ${isAlert ? "bg-red-500/20 shadow-[0_0_25px_rgba(248,113,113,0.35)]" : "bg-emerald-500/15 shadow-[0_0_25px_rgba(16,185,129,0.25)]"}">
                    <span class="text-lg">${isAlert ? "⚠️" : "✨"}</span>
                </div>
                <div class="space-y-2">
                    <div class="text-xs uppercase tracking-[0.2em] text-slate-500">Executive Summary</div>
                    <div class="text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r ${
                        isAlert
                            ? "from-red-400 to-orange-400"
                            : isGood
                            ? "from-emerald-400 to-cyan-400"
                            : "from-indigo-400 to-purple-400"
                    }">
                        ${insight.headline || "Business insight ready."}
                    </div>
                    <div class="bg-white/5 rounded-xl p-4 mt-2 border-l-4 border-purple-500 text-slate-300">
                        <div class="text-sm font-semibold text-white">Recommended Action</div>
                        <div class="mt-1 text-sm">${
                            insight.recommended_action || "Next step will appear here once predictions are generated."
                        }</div>
                    </div>
                    ${
                        insight.detailed_insight
                            ? `<div class="text-sm text-slate-400">${insight.detailed_insight}</div>`
                            : ""
                    }
                </div>
            </div>

            <details class="mt-5 bg-slate-900/70 border border-white/10 rounded-xl p-4">
                <summary class="cursor-pointer font-semibold text-slate-200">Technical Details</summary>
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-3 text-sm text-slate-300">
                    <div class="space-y-1">
                        <div class="text-xs uppercase text-slate-500">Rows</div>
                        <div class="font-semibold">${rows}</div>
                    </div>
                    <div class="space-y-1">
                        <div class="text-xs uppercase text-slate-500">Model Type</div>
                        <div class="font-semibold">${modelType || "-"}</div>
                    </div>
                    <div class="space-y-1">
                        <div class="text-xs uppercase text-slate-500">Accuracy</div>
                        <div class="font-semibold">${accuracy}</div>
                    </div>
                </div>
            </details>
        </div>
    `;

    // Hide legacy metadata grid
    metadataPanel.innerHTML = "";
}

function resolveAccent(intent, modelType) {
    const danger = {
        gradient: "linear-gradient(135deg,#ff5f6d 0%,#ffc371 100%)",
        text: "#0b0b0b"
    };
    const success = {
        gradient: "linear-gradient(135deg,#32d978 0%,#8ef0b1 100%)",
        text: "#0b0b0b"
    };
    const neutral = {
        gradient: "linear-gradient(135deg,#5a6fff 0%,#9fa8ff 100%)",
        text: "#0b0b0b"
    };

    const isDanger = ["churn", "fraud", "anomaly"].some((needle) =>
        intent.includes(needle)
    );
    const isSales = intent.includes("sales") || intent.includes("revenue");

    if (isDanger || modelType === "classification" || modelType === "anomaly") {
        return danger;
    }
    if (isSales || modelType === "regression") {
        return success;
    }
    return neutral;
}

function renderPreview(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        previewSection.style.display = "none";
        previewTable.querySelector("thead").innerHTML = "";
        previewTable.querySelector("tbody").innerHTML = "";
        return;
    }

    const headers = Object.keys(rows[0] ?? {});
    previewTable.querySelector("thead").innerHTML = `
        <tr>
            ${headers.map((header) => `<th>${header}</th>`).join("")}
        </tr>
    `;
    previewTable.querySelector("tbody").innerHTML = rows
        .map(
            (row) => `
                <tr>
                    ${headers.map((header) => `<td>${row[header] ?? ""}</td>`).join("")}
                </tr>
            `
        )
        .join("");
    previewSection.style.display = "block";
}

function setDownloadLink(csvBase64, filename) {
    if (currentDownloadUrl) {
        URL.revokeObjectURL(currentDownloadUrl);
        currentDownloadUrl = null;
    }
    if (!csvBase64) {
        downloadBtn.style.display = "none";
        downloadBtn.removeAttribute("href");
        return;
    }
    const blob = base64ToBlob(csvBase64, "text/csv;charset=utf-8;");
    currentDownloadUrl = URL.createObjectURL(blob);
    downloadBtn.href = currentDownloadUrl;
    downloadBtn.textContent = "Download Actionable Report";
    downloadBtn.download = filename || "predictions.csv";
    downloadBtn.style.display = "inline-flex";
    downloadBtn.className = "border border-purple-500/30 text-purple-300 hover:bg-purple-500/10 rounded-lg py-2 px-4 transition-all inline-flex items-center justify-center";
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i += 1) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

function setStatus(message, tone = "info") {
    statusMessage.textContent = message;
    statusMessage.style.color = toneColors[tone] || toneColors.info;
}
// No model dropdown needed; just prompt user to upload and choose objective.
setStatus("Upload a CSV, pick your goal, and run the analysis.", "info");

// keep train/test percentages in sync
if (trainSplitInput && trainValueEl && testValueEl) {
    const syncSplit = () => {
        const trainPct = Number(trainSplitInput.value);
        const testPct = 100 - trainPct;
        trainValueEl.textContent = `${trainPct}%`;
        testValueEl.textContent = `${testPct}%`;
    };
    trainSplitInput.addEventListener("input", syncSplit);
    syncSplit();
}
