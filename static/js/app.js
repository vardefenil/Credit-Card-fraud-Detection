// ══════════════════════════════════════════════════════════
//  FraudGuard AI — Dashboard JavaScript
// ══════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    if (typeof FEATURE_NAMES !== "undefined" && FEATURE_NAMES.length > 0) {
        buildFeatureInputs();
    }
});

/** Build dynamic feature input fields */
function buildFeatureInputs() {
    const grid = document.getElementById("features-grid");
    if (!grid) return;

    grid.innerHTML = "";
    FEATURE_NAMES.forEach((name, i) => {
        const div = document.createElement("div");
        div.className = "feature-input";
        div.innerHTML = `
            <label for="f_${i}">${name}</label>
            <input type="number" step="any" id="f_${i}" placeholder="0.0" />
        `;
        grid.appendChild(div);
    });
}

/** Load sample transaction data */
async function loadSample(type) {
    try {
        const res = await fetch("/api/sample");
        const data = await res.json();

        const values = data[type];
        if (!values) return;

        FEATURE_NAMES.forEach((_, i) => {
            const input = document.getElementById(`f_${i}`);
            if (input && values[i] !== undefined) {
                input.value = parseFloat(values[i]).toFixed(4);
            }
        });

        // Visual feedback
        const btn = type === "legitimate"
            ? document.getElementById("btn-load-legit")
            : document.getElementById("btn-load-fraud");
        if (btn) {
            btn.style.background = "rgba(108,99,255,0.15)";
            setTimeout(() => { btn.style.background = ""; }, 400);
        }
    } catch (err) {
        console.error("Failed to load sample:", err);
    }
}

/** Send prediction request */
async function predict() {
    const features = [];
    let empty = true;

    FEATURE_NAMES.forEach((_, i) => {
        const input = document.getElementById(`f_${i}`);
        const val = parseFloat(input ? input.value : 0) || 0;
        features.push(val);
        if (val !== 0) empty = false;
    });

    if (empty) {
        alert("Please load a sample or enter feature values first.");
        return;
    }

    // Show loading state
    const btnText = document.querySelector(".btn-text");
    const btnLoader = document.querySelector(".btn-loader");
    if (btnText) btnText.style.display = "none";
    if (btnLoader) btnLoader.style.display = "inline";

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features }),
        });

        const data = await res.json();

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        showResult(data);
    } catch (err) {
        alert("Request failed: " + err.message);
    } finally {
        if (btnText) btnText.style.display = "inline";
        if (btnLoader) btnLoader.style.display = "none";
    }
}

/** Display prediction result */
function showResult(data) {
    const placeholder = document.getElementById("result-placeholder");
    const content = document.getElementById("result-content");
    const badge = document.getElementById("result-badge");
    const probBar = document.getElementById("prob-bar");
    const probValue = document.getElementById("prob-value");
    const details = document.getElementById("result-details");

    if (placeholder) placeholder.style.display = "none";
    if (content) content.style.display = "block";

    // Badge
    if (badge) {
        badge.textContent = data.label;
        badge.className = "result-badge " + (data.is_fraud ? "fraud" : "legit");
    }

    // Probability bar
    const pct = (data.probability * 100).toFixed(2);
    if (probBar) {
        // Small delay for animation
        setTimeout(() => { probBar.style.width = pct + "%"; }, 100);
    }
    if (probValue) {
        probValue.textContent = pct + "% fraud probability";
    }

    // Details
    if (details) {
        details.innerHTML = `
            <p><strong>Prediction:</strong> ${data.is_fraud ? "Fraudulent" : "Legitimate"}</p>
            <p><strong>Fraud Probability:</strong> ${(data.probability * 100).toFixed(4)}%</p>
            <p><strong>Confidence:</strong> ${data.confidence}%</p>
        `;
    }

    // Scroll to result on mobile
    if (window.innerWidth < 900 && content) {
        content.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}
