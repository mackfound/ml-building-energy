/* ── State ─────────────────────────────────────────────────────────────────── */
let chart = null;
let scatterHeat = null;
let scatterCool = null;
let pollTimer = null;
let appReady = false;

/* ── Polling for model readiness ───────────────────────────────────────────── */
function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(checkStatus, 3000);
  checkStatus();
}

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    if (data.ready) {
      clearInterval(pollTimer);
      appReady = true;
      hideOverlay('loading-overlay');
      updateBadge('ready', `Ready — ${data.state}`);
      await loadChoices();
      await loadScatterPlots();
    } else if (data.error) {
      clearInterval(pollTimer);
      showError(data.error);
      updateBadge('error', 'Error');
    } else {
      updateBadge('loading', 'Loading data…');
    }
  } catch {
    updateBadge('error', 'Server unreachable');
  }
}

/* ── Load dropdown choices from the server ─────────────────────────────────── */
async function loadChoices() {
  try {
    const res = await fetch('/api/choices');
    if (!res.ok) return;
    const choices = await res.json();

    populateSelect('sel-fuel',       choices.fuel       || []);
    populateSelect('sel-foundation', choices.foundation || []);
  } catch (e) {
    console.error('Failed to load choices', e);
  }
}

function populateSelect(id, options) {
  const sel = document.getElementById(id);
  sel.innerHTML = options.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
}

/* ── Reload when state changes ─────────────────────────────────────────────── */
document.getElementById('sel-state').addEventListener('change', async function () {
  if (!appReady) return;
  const state = this.value;
  appReady = false;
  updateBadge('loading', `Loading ${state}…`);
  showOverlay('loading-overlay');
  document.getElementById('loading-msg').textContent =
    `Fetching ResStock data for ${state} and retraining models… (~60–90 s)`;

  await fetch('/api/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state }),
  });
  startPolling();
});

/* ── Run prediction ────────────────────────────────────────────────────────── */
async function runPrediction() {
  if (!appReady) return;

  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  btn.textContent = 'Predicting…';

  try {
    const payload = {
      state:      document.getElementById('sel-state').value,
      fuel:       document.getElementById('sel-fuel').value,
      foundation: document.getElementById('sel-foundation').value,
      floor_area: document.getElementById('sl-area').value,
      setpoint:   document.getElementById('sl-setpoint').value,
      stories:    document.querySelector('input[name="stories"]:checked').value,
    };

    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      alert('Prediction failed: ' + (err.error || res.statusText));
      return;
    }

    const d = await res.json();
    showResults(d);
  } catch (e) {
    alert('Request failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Prediction';
  }
}

/* ── Render results ────────────────────────────────────────────────────────── */
function showResults(d) {
  document.getElementById('results-placeholder').classList.add('hidden');
  document.getElementById('results-content').classList.remove('hidden');

  document.getElementById('res-heat').textContent  = fmt(d.heating);
  document.getElementById('res-cool').textContent  = fmt(d.cooling);
  document.getElementById('res-total').textContent = fmt(d.total);
  document.getElementById('res-cost').textContent  = fmtMoney(d.cost);
  document.getElementById('avg-state').textContent = d.state;

  renderChart(d);
}

function renderChart(d) {
  const ctx = document.getElementById('bar-chart').getContext('2d');

  const heatColor = '#f97316';
  const coolColor = '#3b82f6';
  const avgAlpha  = 'rgba(156,163,175,0.7)';

  if (chart) {
    chart.data.datasets[0].data = [d.heating, d.cooling];
    chart.data.datasets[1].data = [d.avg_heating, d.avg_cooling];
    chart.update();
    return;
  }

  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Heating', 'Cooling'],
      datasets: [
        {
          label: 'Your Home',
          data: [d.heating, d.cooling],
          backgroundColor: [heatColor, coolColor],
          borderRadius: 6,
          borderSkipped: false,
        },
        {
          label: `${d.state} Average`,
          data: [d.avg_heating, d.avg_cooling],
          backgroundColor: avgAlpha,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { font: { size: 12 }, boxWidth: 12, padding: 16 },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.parsed.y)} kWh`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(0,0,0,.05)' },
          ticks: {
            callback: v => fmt(v),
            font: { size: 11 },
          },
          title: { display: true, text: 'Annual kWh', font: { size: 11 } },
        },
        x: {
          grid: { display: false },
          ticks: { font: { size: 12, weight: '600' } },
        },
      },
    },
  });
}

/* ── Overlay helpers ───────────────────────────────────────────────────────── */
function showOverlay(id) {
  document.getElementById(id).classList.remove('hidden');
}
function hideOverlay(id) {
  document.getElementById(id).classList.add('hidden');
}
function showError(msg) {
  document.getElementById('error-msg').textContent = msg;
  hideOverlay('loading-overlay');
  showOverlay('error-overlay');
}

async function retryLoad() {
  hideOverlay('error-overlay');
  showOverlay('loading-overlay');
  document.getElementById('loading-msg').textContent =
    'Connecting to NREL S3 and training models…';
  const state = document.getElementById('sel-state').value;
  await fetch('/api/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state }),
  });
  startPolling();
}

/* ── Badge helper ──────────────────────────────────────────────────────────── */
function updateBadge(type, text) {
  const badge = document.getElementById('status-badge');
  badge.className = `badge badge-${type}`;
  document.getElementById('status-text').textContent = text;
}

/* ── Utility ───────────────────────────────────────────────────────────────── */
function fmt(n) {
  return Number(n).toLocaleString();
}
function fmtMoney(n) {
  return Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ── Scatter plots ─────────────────────────────────────────────────────────── */
async function loadScatterPlots() {
  try {
    const res = await fetch('/api/rf_predictions');
    if (!res.ok) return;
    const data = await res.json();
    renderScatter('scatter-heat', data.heating, 'RF Heating Load', '#f97316');
    renderScatter('scatter-cool', data.cooling, 'RF Cooling Load', '#3b82f6');
  } catch (e) {
    console.error('Failed to load RF scatter data', e);
  }
}

function renderScatter(canvasId, d, label, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const points = d.actual.map((a, i) => ({ x: a, y: d.predicted[i] }));
  const maxVal = Math.max(...d.actual, ...d.predicted) * 1.05;

  const existing = canvasId === 'scatter-heat' ? scatterHeat : scatterCool;
  if (existing) existing.destroy();

  const c = new Chart(ctx.getContext('2d'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Test set buildings',
          data: points,
          backgroundColor: color + '55',
          borderColor:     color + 'cc',
          borderWidth: 0.5,
          pointRadius: 3,
          pointHoverRadius: 5,
        },
        {
          label: 'Perfect prediction',
          data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
          type: 'line',
          borderColor: '#64748b',
          borderDash: [5, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: label,
          font: { size: 13, weight: '700' },
          color: '#1f2937',
          padding: { bottom: 2 },
        },
        subtitle: {
          display: true,
          text: `R² = ${d.r2.toFixed(4)}`,
          font: { size: 11 },
          color: '#6b7280',
          padding: { bottom: 8 },
        },
        legend: {
          position: 'bottom',
          labels: { font: { size: 10 }, boxWidth: 10, padding: 10 },
        },
        tooltip: {
          callbacks: {
            label: ctx => ctx.datasetIndex === 0
              ? ` Actual: ${fmt(Math.round(ctx.parsed.x))} | Predicted: ${fmt(Math.round(ctx.parsed.y))}`
              : null,
          },
          filter: item => item.datasetIndex === 0,
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Actual (kWh)', font: { size: 10 } },
          min: 0, max: maxVal,
          ticks: { callback: v => fmt(v), font: { size: 9 }, maxTicksLimit: 6 },
          grid: { color: 'rgba(0,0,0,.04)' },
        },
        y: {
          title: { display: true, text: 'Predicted (kWh)', font: { size: 10 } },
          min: 0, max: maxVal,
          ticks: { callback: v => fmt(v), font: { size: 9 }, maxTicksLimit: 6 },
          grid: { color: 'rgba(0,0,0,.04)' },
        },
      },
    },
  });

  if (canvasId === 'scatter-heat') scatterHeat = c;
  else scatterCool = c;
}

/* ── Boot ──────────────────────────────────────────────────────────────────── */
startPolling();
