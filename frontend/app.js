// ==========================================================================
// Speech Emotion Recognition App - Frontend Logic
// ==========================================================================

const API_BASE = window.location.origin;

let classBalanceChart = null;
let mfccDistChart = null;
let currentEdaData = null;
let selectedAudioFile = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupPredictHandlers();
  checkSystemStatus();
  loadAllData();
});

// Navigation Controller
function setupNavigation() {
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const pageId = btn.getAttribute("data-page");
      switchPage(pageId);
    });
  });
}

function switchPage(pageId) {
  // Update nav buttons
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-page") === pageId);
  });

  // Update page sections
  document.querySelectorAll(".page-section").forEach(sec => {
    sec.classList.remove("active");
  });

  const activeSection = document.getElementById(`page-${pageId}`);
  if (activeSection) {
    activeSection.classList.add("active");
  }

  // Refresh charts if opening eda or results
  if (pageId === "eda" && currentEdaData) {
    renderClassBalanceChart(currentEdaData.class_balance);
    const select = document.getElementById("mfccSelect");
    renderMfccChart(select ? select.value : "MFCC_1");
  }
}

// System Status Check
async function checkSystemStatus() {
  const statusEl = document.getElementById("headerStatus");
  const alertEl = document.getElementById("trainingAlert");
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error("Status endpoint unavailable");
    const data = await res.json();
    
    if (data.status === "ready") {
      statusEl.className = "header-status ready";
      statusEl.querySelector(".status-text").textContent = "Model & Metrics Ready";
      if (alertEl) alertEl.classList.add("hidden");
    } else {
      statusEl.className = "header-status waiting";
      statusEl.querySelector(".status-text").textContent = "Needs Pipeline Run";
      if (alertEl) alertEl.classList.remove("hidden");
    }
  } catch (err) {
    statusEl.className = "header-status";
    statusEl.querySelector(".status-text").textContent = "Backend Offline";
  }
}

// Load data for all sections
async function loadAllData() {
  await Promise.allSettled([
    loadEDA(),
    loadResults()
  ]);
}

// ==========================================================================
// EDA Data Loading & Rendering
// ==========================================================================
async function loadEDA() {
  try {
    const res = await fetch(`${API_BASE}/api/eda`);
    if (!res.ok) return;
    const eda = await res.json();
    currentEdaData = eda;

    // 1. Populate Summary Stats Table
    renderSummaryStatsTable(eda.summary_stats);

    // 2. Render Class Balance Chart
    renderClassBalanceChart(eda.class_balance);

    // 3. Setup MFCC Dropdown & Chart
    setupMfccDropdown(eda.mfcc_distributions);

    // 4. Render Correlation Heatmap
    renderCorrelationHeatmap(eda.correlation);

    // 5. Populate Observations
    renderObservations(eda.observations);

  } catch (err) {
    console.warn("EDA data not ready yet:", err);
  }
}

function renderSummaryStatsTable(stats) {
  const tbody = document.querySelector("#edaSummaryTable tbody");
  if (!tbody || !stats || !stats.groups) return;

  tbody.innerHTML = stats.groups.map(g => `
    <tr>
      <td><strong>${g.feature_group}</strong></td>
      <td><code>${g.dimension}</code></td>
      <td>${g.mean}</td>
      <td>${g.median}</td>
      <td>${g.std}</td>
      <td>${g.min}</td>
      <td>${g.max}</td>
      <td><span class="text-muted">${g.description}</span></td>
    </tr>
  `).join("");
}

function renderClassBalanceChart(classBalance) {
  const canvas = document.getElementById("classBalanceChart");
  if (!canvas || !classBalance) return;

  if (classBalanceChart) {
    classBalanceChart.destroy();
  }

  const emotionColors = {
    calm: "#0284c7",
    happy: "#16a34a",
    fearful: "#ea580c",
    disgust: "#7c3aed"
  };

  const bgColors = classBalance.labels.map(l => emotionColors[l.toLowerCase()] || "#2563eb");

  classBalanceChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: classBalance.labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets: [{
        label: "Sample Count",
        data: classBalance.counts,
        backgroundColor: bgColors,
        borderRadius: 6,
        barThickness: 48
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw} samples (1:1:1:1 balanced)`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          suggestedMax: 220,
          grid: { color: "#f1f5f9" },
          ticks: { font: { family: "Inter" } }
        },
        x: {
          grid: { display: false },
          ticks: { font: { family: "Inter", weight: 600 } }
        }
      }
    }
  });
}

function setupMfccDropdown(mfccDistributions) {
  const select = document.getElementById("mfccSelect");
  if (!select) return;

  select.addEventListener("change", (e) => {
    renderMfccChart(e.target.value);
  });

  renderMfccChart("MFCC_1");
}

function renderMfccChart(mfccKey) {
  const canvas = document.getElementById("mfccDistChart");
  if (!canvas || !currentEdaData || !currentEdaData.mfcc_distributions) return;

  const distData = currentEdaData.mfcc_distributions[mfccKey];
  if (!distData) return;

  if (mfccDistChart) {
    mfccDistChart.destroy();
  }

  const emotionColors = {
    calm: { border: "#0284c7", bg: "rgba(2, 132, 199, 0.15)" },
    happy: { border: "#16a34a", bg: "rgba(22, 163, 74, 0.15)" },
    fearful: { border: "#ea580c", bg: "rgba(234, 88, 12, 0.15)" },
    disgust: { border: "#7c3aed", bg: "rgba(124, 58, 237, 0.15)" }
  };

  const emotions = Object.keys(distData);
  const firstEmotion = emotions[0];
  const labels = distData[firstEmotion].bin_centers;

  const datasets = emotions.map(emo => {
    const styling = emotionColors[emo.toLowerCase()] || { border: "#2563eb", bg: "rgba(37, 99, 235, 0.1)" };
    return {
      label: emo.charAt(0).toUpperCase() + emo.slice(1),
      data: distData[emo].hist_counts,
      borderColor: styling.border,
      backgroundColor: styling.bg,
      fill: true,
      tension: 0.35,
      borderWidth: 2,
      pointRadius: 2
    };
  });

  mfccDistChart = new Chart(canvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: { font: { family: "Inter", size: 11 } }
        },
        tooltip: {
          callbacks: {
            title: (items) => `Value: ~${items[0].label}`
          }
        }
      },
      scales: {
        y: {
          title: { display: true, text: "Sample Count", font: { family: "Inter", size: 11 } },
          grid: { color: "#f1f5f9" }
        },
        x: {
          title: { display: true, text: "Acoustic Magnitude", font: { family: "Inter", size: 11 } },
          grid: { display: false }
        }
      }
    }
  });
}

function renderCorrelationHeatmap(corr) {
  const container = document.getElementById("correlationHeatmapContainer");
  if (!container || !corr) return;

  const { labels, matrix } = corr;

  let tableHtml = `<table class="heatmap-table"><thead><tr><th></th>`;
  labels.forEach(l => {
    tableHtml += `<th>${l}</th>`;
  });
  tableHtml += `</tr></thead><tbody>`;

  matrix.forEach((row, i) => {
    tableHtml += `<tr><th>${labels[i]}</th>`;
    row.forEach((val, j) => {
      // Linear color scale between -1 (cool red/white) to 1 (deep blue)
      const color = getCorrelationColor(val);
      const textColor = Math.abs(val) > 0.45 ? "#ffffff" : "#0f172a";
      tableHtml += `<td style="background-color: ${color}; color: ${textColor};" title="${labels[i]} vs ${labels[j]}: r = ${val}">${val.toFixed(2)}</td>`;
    });
    tableHtml += `</tr>`;
  });

  tableHtml += `</tbody></table>`;
  container.innerHTML = tableHtml;
}

function getCorrelationColor(val) {
  // Normalize -1 to 1 into blue-slate gradient
  if (val >= 0) {
    const intensity = Math.min(val, 1);
    const r = Math.round(239 - (239 - 37) * intensity);
    const g = Math.round(246 - (246 - 99) * intensity);
    const b = Math.round(255 - (255 - 235) * intensity);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const intensity = Math.min(Math.abs(val), 1);
    const r = Math.round(254 - (254 - 220) * intensity);
    const g = Math.round(242 - (242 - 38) * intensity);
    const b = Math.round(242 - (242 - 38) * intensity);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

function renderObservations(observations) {
  const list = document.getElementById("edaObservationsList");
  if (!list || !observations) return;

  list.innerHTML = observations.map(obs => `<li>${obs}</li>`).join("");
}

// ==========================================================================
// Results & Model Data Loading & Rendering
// ==========================================================================
async function loadResults() {
  try {
    const res = await fetch(`${API_BASE}/api/results`);
    if (!res.ok) return;
    const results = await res.json();

    // 1. Model Config
    renderModelConfig(results.model_config);

    // 2. Metrics Table
    renderMetricsTable(results.metrics_table);

    // 3. Confusion Matrix
    renderConfusionMatrix(results.confusion_matrix);

    // 4. Per-Class Metrics & Fearful Recall Highlight
    renderPerClassMetrics(results.per_class_metrics, results.fearful_recall_highlight);

    // 5. Interpretation
    renderInterpretation(results.interpretation);

  } catch (err) {
    console.warn("Results data not ready yet:", err);
  }
}

function renderModelConfig(cfg) {
  if (!cfg) return;
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setVal("cfg-type", cfg.model_type || "MLPClassifier");
  setVal("cfg-arch", `(${cfg.hidden_layer_sizes.join(", ")},)`);
  setVal("cfg-lr", cfg.learning_rate);
  setVal("cfg-act", cfg.activation);
  setVal("cfg-solv", cfg.solver);
  setVal("cfg-iter", `${cfg.iterations_completed} iterations`);
}

function renderMetricsTable(metrics) {
  const tbody = document.querySelector("#metricsTable tbody");
  if (!tbody || !metrics) return;

  tbody.innerHTML = Object.entries(metrics).map(([metricName, vals]) => {
    const diff = (vals.train - vals.test);
    const diffPct = (diff * 100).toFixed(2);
    const diffLabel = diff > 0.15 ? `<span class="badge gray-badge" style="color:#d97706">+${diffPct}% gap</span>` : `<span class="badge blue-badge">+${diffPct}%</span>`;

    return `
      <tr>
        <td><strong>${metricName}</strong></td>
        <td>${(vals.train * 100).toFixed(2)}% (${vals.train})</td>
        <td><strong>${(vals.test * 100).toFixed(2)}%</strong> (${vals.test})</td>
        <td>${diffLabel}</td>
      </tr>
    `;
  }).join("");
}

function renderConfusionMatrix(cm) {
  const container = document.getElementById("confusionMatrixContainer");
  if (!container || !cm) return;

  const { labels, matrix_counts, matrix_normalized } = cm;

  let tableHtml = `<table class="cm-table"><thead><tr><th style="font-size:0.75rem; color:#64748b;">True \\ Pred</th>`;
  labels.forEach(l => {
    tableHtml += `<th>${l.charAt(0).toUpperCase() + l.slice(1)}</th>`;
  });
  tableHtml += `</tr></thead><tbody>`;

  matrix_counts.forEach((row, i) => {
    tableHtml += `<tr><th>${labels[i].charAt(0).toUpperCase() + labels[i].slice(1)}</th>`;
    row.forEach((count, j) => {
      const normVal = matrix_normalized[i][j];
      const isDiag = i === j;
      const color = isDiag
        ? `rgba(37, 99, 235, ${0.15 + normVal * 0.7})`
        : `rgba(234, 88, 12, ${normVal * 0.85})`;
      const textColor = (isDiag && normVal > 0.4) ? "#ffffff" : "#0f172a";

      tableHtml += `
        <td>
          <div class="cm-cell" style="background-color: ${color}; color: ${textColor};" title="True: ${labels[i]}, Pred: ${labels[j]}">
            <div class="cm-cell-count">${count}</div>
            <div class="cm-cell-pct">${(normVal * 100).toFixed(1)}%</div>
          </div>
        </td>
      `;
    });
    tableHtml += `</tr>`;
  });

  tableHtml += `</tbody></table>`;
  container.innerHTML = tableHtml;
}

function renderPerClassMetrics(perClass, fearfulHighlight) {
  const tbody = document.querySelector("#perClassTable tbody");
  if (tbody && perClass) {
    tbody.innerHTML = Object.entries(perClass).map(([emo, vals]) => {
      const isFearful = emo.toLowerCase() === "fearful";
      const rowStyle = isFearful ? `style="background-color: #fff7ed; font-weight:600;"` : "";
      return `
        <tr ${rowStyle}>
          <td>
            <span class="pred-badge ${emo.toLowerCase()}" style="font-size:0.75rem; padding:0.2rem 0.6rem;">
              ${emo.toUpperCase()}
            </span>
            ${isFearful ? '<span style="font-size:0.75rem; color:#ea580c; margin-left:4px;">(SDG 3 Focus)</span>' : ''}
          </td>
          <td>${(vals.precision * 100).toFixed(2)}%</td>
          <td><strong>${(vals.recall * 100).toFixed(2)}%</strong></td>
          <td>${(vals.f1_score * 100).toFixed(2)}%</td>
          <td>${vals.support}</td>
        </tr>
      `;
    }).join("");
  }

  // Update highlight card
  if (fearfulHighlight) {
    const valEl = document.getElementById("fearfulRecallVal");
    if (valEl) valEl.textContent = `${fearfulHighlight.recall_pct}%`;
  }
}

function renderInterpretation(interp) {
  const container = document.getElementById("interpretationContent");
  if (!container || !interp) return;

  container.innerHTML = `
    <p style="font-size:1.02rem; line-height: 1.65; color: #1e293b;">
      ${interp.narrative}
    </p>
    <div style="margin-top: 1rem; display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;">
      <div class="sub-card">
        <strong>Overfitting Gap:</strong><br>${interp.overfitting_gap_pct}% (${interp.gap_status})
      </div>
      <div class="sub-card">
        <strong>Top Confusion Pair:</strong><br>${interp.top_confusion_pair}
      </div>
      <div class="sub-card">
        <strong>Hypothesis Status:</strong><br>
        ${interp.hypothesis_test_acc_above_random ? '✅ Accuracy > 25%' : '❌ Under baseline'} |
        ${interp.hypothesis_confusion_supported ? '✅ Acoustic confusion observed' : '⚠️ Errors broadly distributed'}
      </div>
    </div>
  `;
}

// ==========================================================================
// Predict Page Handlers (Upload, Waveform, Inference)
// ==========================================================================
function setupPredictHandlers() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("audioFileInput");
  const predictBtn = document.getElementById("predictBtn");
  const audioPreview = document.getElementById("audioPreview");
  const audioPlayerWrap = document.getElementById("audioPlayerWrap");

  if (!dropZone || !fileInput) return;

  // Click on dropzone triggers input
  dropZone.addEventListener("click", () => fileInput.click());

  // Drag & Drop
  ["dragenter", "dragover"].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  predictBtn.addEventListener("click", () => {
    if (selectedAudioFile) {
      runPrediction(selectedAudioFile);
    }
  });
}

function handleFileSelected(file) {
  if (!file.name.toLowerCase().endsWith(".wav")) {
    showPredictError("Please select a valid .wav speech audio file.");
    return;
  }

  selectedAudioFile = file;

  // Update file info display
  const fileInfo = document.getElementById("fileInfo");
  const fileName = document.getElementById("fileName");
  const fileSize = document.getElementById("fileSize");
  fileName.textContent = file.name;
  fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
  fileInfo.classList.remove("hidden");

  // Setup preview
  const audioPreview = document.getElementById("audioPreview");
  const audioPlayerWrap = document.getElementById("audioPlayerWrap");
  audioPreview.src = URL.createObjectURL(file);
  audioPlayerWrap.classList.remove("hidden");

  // Enable button
  const predictBtn = document.getElementById("predictBtn");
  predictBtn.disabled = false;

  // Clear previous errors
  hidePredictError();
}

async function runPrediction(file) {
  const predictBtn = document.getElementById("predictBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const statusBadge = document.getElementById("inferenceStatusBadge");
  const emptyState = document.getElementById("predictEmptyState");
  const resultContent = document.getElementById("predictResultContent");

  predictBtn.disabled = true;
  btnSpinner.classList.remove("hidden");
  statusBadge.textContent = "Extracting Features...";
  hidePredictError();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: "Prediction request failed." }));
      throw new Error(errData.detail || "Prediction request failed.");
    }

    const data = await res.json();

    // Render result
    emptyState.classList.add("hidden");
    resultContent.classList.remove("hidden");
    statusBadge.textContent = "Prediction Complete";

    const badge = document.getElementById("predEmotionBadge");
    badge.textContent = data.predicted_emotion;
    badge.className = `pred-badge ${data.predicted_emotion.toLowerCase()}`;

    document.getElementById("predConfidence").textContent = `${data.confidence}%`;
    document.getElementById("predDescription").textContent = data.description;

    // Render probability bars
    renderProbabilityBars(data.probabilities);

    // Render audio descriptors summary
    renderAudioDescriptors(data.audio_summary);

  } catch (err) {
    showPredictError(err.message);
    statusBadge.textContent = "Error";
  } finally {
    predictBtn.disabled = false;
    btnSpinner.classList.add("hidden");
  }
}

function renderProbabilityBars(probs) {
  const wrap = document.getElementById("probBarsWrap");
  if (!wrap || !probs) return;

  wrap.innerHTML = Object.entries(probs).map(([emo, val]) => {
    const pct = (val * 100).toFixed(1);
    const emoClass = emo.toLowerCase();
    return `
      <div class="prob-row">
        <div class="prob-info">
          <span style="text-transform: capitalize;">${emo}</span>
          <span><strong>${pct}%</strong></span>
        </div>
        <div class="bar-bg">
          <div class="bar-fill ${emoClass}" style="width: ${pct}%;"></div>
        </div>
      </div>
    `;
  }).join("");
}

function renderAudioDescriptors(summary) {
  const grid = document.getElementById("featSummaryGrid");
  if (!grid || !summary) return;

  grid.innerHTML = `
    <div class="summary-pill">
      <span class="pill-label">Duration</span>
      <span class="pill-val">${summary.duration_seconds}s</span>
    </div>
    <div class="summary-pill">
      <span class="pill-label">Sampling Rate</span>
      <span class="pill-val">${summary.sample_rate_hz} Hz</span>
    </div>
    <div class="summary-pill">
      <span class="pill-label">Total Samples</span>
      <span class="pill-val">${summary.n_samples}</span>
    </div>
    <div class="summary-pill">
      <span class="pill-label">Mean MFCC Energy</span>
      <span class="pill-val">${summary.mean_mfcc_energy}</span>
    </div>
    <div class="summary-pill">
      <span class="pill-label">Mean Chroma Energy</span>
      <span class="pill-val">${summary.mean_chroma_energy}</span>
    </div>
    <div class="summary-pill">
      <span class="pill-label">Mean Mel Energy</span>
      <span class="pill-val">${summary.mean_mel_energy}</span>
    </div>
  `;
}

function showPredictError(msg) {
  const banner = document.getElementById("predictErrorBanner");
  const text = document.getElementById("predictErrorText");
  if (banner && text) {
    text.textContent = msg;
    banner.classList.remove("hidden");
  }
}

function hidePredictError() {
  const banner = document.getElementById("predictErrorBanner");
  if (banner) banner.classList.add("hidden");
}
