const API_BASE_URL = "http://localhost:8000";

const form = document.getElementById("predictionForm");
const modelSelect = document.getElementById("modelSelect");
const fileInput = document.getElementById("fileInput");
const targetColumnInput = document.getElementById("targetColumn");
const usePreviewToggle = document.getElementById("previewToggle");
const returnCsvToggle = document.getElementById("csvToggle");
const statusMessage = document.getElementById("statusMessage");
const metadataPanel = document.getElementById("metadataPanel");
const predictionPanel = document.getElementById("predictionPanel");
const previewSection = document.getElementById("previewSection");
const previewTable = document.getElementById("previewTable");
const downloadBtn = document.getElementById("downloadBtn");
const submitButton = form.querySelector("button[type='submit']");

let currentDownloadUrl = null;

const toneColors = {
    info: "#8a93b2",
    success: "#5cf4d4",
    error: "#ff8fa3"
};

async function fetchModels() {
    setStatus("Loading models from the API...");
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (!response.ok) {
            throw new Error(`Failed to load models: ${response.status}`);
        }
        const payload = await response.json();
        populateModelDropdown(payload.models || []);
        setStatus("Models ready. Upload a CSV to continue.", "success");
    } catch (error) {
        console.error(error);
        setStatus("Unable to reach backend. Start FastAPI or check CORS settings.", "error");
        modelSelect.innerHTML = '<option value="" disabled selected>Backend unavailable</option>';
    }
}

function populateModelDropdown(models) {
    if (!Array.isArray(models) || models.length === 0) {
        modelSelect.innerHTML = '<option value="" disabled selected>No models available</option>';
        return;
    }

    modelSelect.innerHTML = models
        .map((model) => {
            const description = model.description ? ` — ${model.description}` : "";
            return `<option value="${model.name}">${model.display_name}${description}</option>`;
        })
        .join("");
}

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
        formData.append("model_name", modelSelect.value);
        formData.append("use_preview", usePreviewToggle.checked ? "true" : "false");
        formData.append("return_csv", returnCsvToggle.checked ? "true" : "false");
        if (targetColumnInput.value.trim()) {
            formData.append("target_column", targetColumnInput.value.trim());
        }
        formData.append("file", fileInput.files[0]);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorPayload = await response.json().catch(() => ({}));
            const detail = errorPayload.detail || "Prediction failed.";
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

function hydrateUiWithResults(data) {
    renderMetadata(data.metadata, data.model_type);
    renderPredictions(data.predictions);
    renderPreview(data.preview);
    setDownloadLink(data.csv_base64, data.csv_filename);
}

function renderMetadata(metadata = {}, modelType = "") {
    const items = [];
    if (typeof metadata.rows === "number") {
        items.push({ label: "Rows", value: metadata.rows });
    }
    if (Array.isArray(metadata.columns)) {
        items.push({ label: "Columns", value: metadata.columns.length });
    }
    if (modelType) {
        items.push({ label: "Model Type", value: modelType });
    }
    if (metadata.target_column) {
        items.push({ label: "Target Column", value: metadata.target_column });
    }

    metadataPanel.innerHTML = items
        .map(
            (item) => `
                <div class="meta-item">
                    <span class="meta-label">${item.label}</span>
                    <span class="meta-value">${item.value}</span>
                </div>
            `
        )
        .join("");
}

function renderPredictions(predictions) {
    if (!Array.isArray(predictions) || predictions.length === 0) {
        predictionPanel.innerHTML = "";
        return;
    }
    const sample = predictions.slice(0, 5);
    const body = JSON.stringify(sample, null, 2);
    predictionPanel.innerHTML = `<pre>${body}</pre>`;
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
    downloadBtn.download = filename || "predictions.csv";
    downloadBtn.style.display = "inline-flex";
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

fetchModels();
