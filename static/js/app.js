/**
 * FloraPredict Frontend Logic
 * Connects UI to Flask API endpoints: /api/info, /api/predict, and /api/train
 */

const PRESETS = {
  setosa: {
    sepal_length: 5.1,
    sepal_width: 3.5,
    petal_length: 1.4,
    petal_width: 0.2
  },
  versicolor: {
    sepal_length: 6.0,
    sepal_width: 2.9,
    petal_length: 4.5,
    petal_width: 1.5
  },
  virginica: {
    sepal_length: 6.7,
    sepal_width: 3.1,
    petal_length: 5.6,
    petal_width: 2.4
  }
};

const SPECIES_ICONS = {
  setosa: "🌸",
  versicolor: "🌿",
  virginica: "🌺"
};

// Sync input range value to label display
function syncVal(inputId, displayId, suffix = "") {
  const input = document.getElementById(inputId);
  const display = document.getElementById(displayId);
  if (input && display) {
    display.textContent = input.value + suffix;
  }
  updatePythonSnippet();
}

// Apply quick sample preset
function applyPreset(speciesKey) {
  const p = PRESETS[speciesKey];
  if (!p) return;

  document.getElementById("sepal_length").value = p.sepal_length;
  document.getElementById("sepal_width").value = p.sepal_width;
  document.getElementById("petal_length").value = p.petal_length;
  document.getElementById("petal_width").value = p.petal_width;

  syncVal("sepal_length", "val-sepal-length", " cm");
  syncVal("sepal_width", "val-sepal-width", " cm");
  syncVal("petal_length", "val-petal-length", " cm");
  syncVal("petal_width", "val-petal-width", " cm");

  // Auto-predict on preset selection
  handlePredict();
}

// Update the requests code block to reflect current slider values
function updatePythonSnippet() {
  const sl = parseFloat(document.getElementById("sepal_length").value);
  const sw = parseFloat(document.getElementById("sepal_width").value);
  const pl = parseFloat(document.getElementById("petal_length").value);
  const pw = parseFloat(document.getElementById("petal_width").value);

  const snippet = `import requests

# Predict with Flask API
url = "http://127.0.0.1:5000/api/predict"
payload = {
    "features": [${sl}, ${sw}, ${pl}, ${pw}]
}
response = requests.post(url, json=payload)
data = response.json()

print(f"Prediction: {data['prediction_label']}")
print(f"Probabilities: {data['probabilities']}")`;

  const codeEl = document.getElementById("python-code-display");
  if (codeEl) {
    codeEl.textContent = snippet;
  }
}

// Copy Python snippet
function copyRequestsCode() {
  const codeEl = document.getElementById("python-code-display");
  if (codeEl) {
    navigator.clipboard.writeText(codeEl.textContent);
    const btn = document.querySelector(".btn-copy");
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = orig; }, 1800);
    }
  }
}

// Fetch model metadata on load
async function loadModelInfo() {
  const statusDot = document.querySelector(".status-dot");
  const statusText = document.getElementById("model-status-text");

  try {
    const res = await fetch("/api/info");
    if (!res.ok) throw new Error("API returned non-200");
    const data = await res.json();

    document.getElementById("stat-model-type").textContent = data.model_type || "RandomForest";
    document.getElementById("stat-accuracy").textContent = `${data.accuracy}%`;
    document.getElementById("stat-estimators").textContent = `${data.n_estimators} Trees`;

    if (statusText) statusText.textContent = "API & Model Online";
  } catch (err) {
    console.error("Failed to load model info:", err);
    if (statusText) statusText.textContent = "Server Offline";
    if (statusDot) statusDot.style.background = "#ef4444";
  }
}

// Handle Predict button submit
async function handlePredict() {
  const btn = document.getElementById("btn-predict");
  const latencyBadge = document.getElementById("latency-badge");
  const origBtnContent = btn.innerHTML;

  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon">⏳</span> Computing...`;

  const sl = parseFloat(document.getElementById("sepal_length").value);
  const sw = parseFloat(document.getElementById("sepal_width").value);
  const pl = parseFloat(document.getElementById("petal_length").value);
  const pw = parseFloat(document.getElementById("petal_width").value);

  const payload = {
    features: [sl, sw, pl, pw]
  };

  const startTime = performance.now();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const elapsed = Math.round(performance.now() - startTime);
    latencyBadge.textContent = `${elapsed} ms`;

    const data = await res.json();

    if (data.status === "success") {
      document.getElementById("result-empty").style.display = "none";
      document.getElementById("result-active").style.display = "block";

      const label = data.prediction_label;
      const cleanKey = label.toLowerCase().replace("iris-", "").trim();
      const icon = SPECIES_ICONS[cleanKey] || "🌸";

      document.getElementById("pred-icon").textContent = icon;
      document.getElementById("pred-class-name").textContent = `Iris ${cleanKey.charAt(0).toUpperCase() + cleanKey.slice(1)}`;

      const heroCard = document.getElementById("pred-hero-card");
      if (cleanKey === "setosa") {
        heroCard.style.borderColor = "rgba(236, 72, 153, 0.6)";
        heroCard.style.background = "linear-gradient(135deg, rgba(236, 72, 153, 0.18) 0%, rgba(244, 63, 94, 0.1) 100%)";
      } else if (cleanKey === "versicolor") {
        heroCard.style.borderColor = "rgba(59, 130, 246, 0.6)";
        heroCard.style.background = "linear-gradient(135deg, rgba(59, 130, 246, 0.18) 0%, rgba(6, 182, 212, 0.1) 100%)";
      } else {
        heroCard.style.borderColor = "rgba(139, 92, 246, 0.6)";
        heroCard.style.background = "linear-gradient(135deg, rgba(139, 92, 246, 0.18) 0%, rgba(168, 85, 247, 0.1) 100%)";
      }

      // Set confidence and bars
      const probs = data.probabilities || {};
      const setosaP = probs["setosa"] ?? 0;
      const versicolorP = probs["versicolor"] ?? 0;
      const virginicaP = probs["virginica"] ?? 0;

      const highestProb = Math.max(setosaP, versicolorP, virginicaP);
      document.getElementById("pred-confidence").textContent = `${highestProb}% Confidence`;

      document.getElementById("prob-val-setosa").textContent = `${setosaP}%`;
      document.getElementById("bar-setosa").style.width = `${setosaP}%`;

      document.getElementById("prob-val-versicolor").textContent = `${versicolorP}%`;
      document.getElementById("bar-versicolor").style.width = `${versicolorP}%`;

      document.getElementById("prob-val-virginica").textContent = `${virginicaP}%`;
      document.getElementById("bar-virginica").style.width = `${virginicaP}%`;

      // JSON details view
      document.getElementById("raw-json-output").textContent = JSON.stringify(data, null, 2);
    } else {
      alert("Prediction error: " + (data.error || "Unknown"));
    }
  } catch (err) {
    console.error("Predict error:", err);
    alert("Failed to communicate with Flask /api/predict");
  } finally {
    btn.disabled = false;
    btn.innerHTML = origBtnContent;
  }
}

// Handle On-Demand Retraining
async function handleRetrain() {
  const btn = document.getElementById("btn-train");
  const feedback = document.getElementById("train-feedback");
  const origBtnContent = btn.innerHTML;

  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon">⏳</span> Training Model...`;
  feedback.className = "train-feedback";
  feedback.textContent = "";

  const payload = {
    n_estimators: parseInt(document.getElementById("train_n_estimators").value, 10),
    test_size: parseFloat(document.getElementById("train_test_size").value),
    random_state: parseInt(document.getElementById("train_random_state").value, 10)
  };

  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (data.status === "success") {
      feedback.className = "train-feedback success";
      feedback.textContent = `✓ ${data.message} | New Accuracy: ${data.accuracy}% (Trees: ${data.n_estimators})`;

      // Refresh top stats
      document.getElementById("stat-accuracy").textContent = `${data.accuracy}%`;
      document.getElementById("stat-estimators").textContent = `${data.n_estimators} Trees`;

      // Re-run prediction if active
      if (document.getElementById("result-active").style.display !== "none") {
        handlePredict();
      }
    } else {
      feedback.className = "train-feedback error";
      feedback.textContent = `✗ Retraining failed: ${data.message || data.error}`;
    }
  } catch (err) {
    feedback.className = "train-feedback error";
    feedback.textContent = `✗ Request failed: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = origBtnContent;
  }
}

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  loadModelInfo();
  updatePythonSnippet();
  // Auto-run first sample prediction for instant WOW factor
  applyPreset("versicolor");
});
