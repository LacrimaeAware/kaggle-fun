const state = {
  rows: [],
  filtered: [],
  notes: {},
  index: 0,
  zoom: 1,
  pack: "remaining",
  saveTimer: null,
};

const els = {
  title: document.getElementById("title"),
  packSelect: document.getElementById("packSelect"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  groupFilter: document.getElementById("groupFilter"),
  tierFilter: document.getElementById("tierFilter"),
  searchInput: document.getElementById("searchInput"),
  imageList: document.getElementById("imageList"),
  depthGuess: document.getElementById("depthGuess"),
  scaleState: document.getElementById("scaleState"),
  depthSource: document.getElementById("depthSource"),
  imageMeta: document.getElementById("imageMeta"),
  viewport: document.getElementById("viewport"),
  reviewImage: document.getElementById("reviewImage"),
  depthOverlay: document.getElementById("depthOverlay"),
  zoomOutBtn: document.getElementById("zoomOutBtn"),
  zoomResetBtn: document.getElementById("zoomResetBtn"),
  zoomInBtn: document.getElementById("zoomInBtn"),
  pxCm: document.getElementById("pxCm"),
  pxMm: document.getElementById("pxMm"),
  tickPx: document.getElementById("tickPx"),
  rulerPx: document.getElementById("rulerPx"),
  depthText: document.getElementById("depthText"),
  ocrDepthText: document.getElementById("ocrDepthText"),
  imageFamily: document.getElementById("imageFamily"),
  submittedState: document.getElementById("submittedState"),
  scaleZ: document.getElementById("scaleZ"),
  reason: document.getElementById("reason"),
  note: document.getElementById("note"),
  oracleScale: document.getElementById("oracleScale"),
  oracleDepth: document.getElementById("oracleDepth"),
  oracleTicks: document.getElementById("oracleTicks"),
  oracleComment: document.getElementById("oracleComment"),
  saveState: document.getElementById("saveState"),
};

function fmt(value, suffix = "") {
  if (value === undefined || value === null || value === "" || value === "nan") return "--";
  const n = Number(value);
  if (Number.isFinite(n)) return `${n.toFixed(n >= 100 ? 1 : 2)}${suffix}`;
  return `${value}${suffix}`;
}

function depthLabel(row) {
  return row && row.depth_guess_mm ? fmt(row.depth_guess_mm, " mm") : "Unknown";
}

function rowKey(row) {
  return row ? row.image_id : "";
}

async function load(pack = state.pack) {
  state.pack = pack;
  const res = await fetch(`/api/manifest?pack=${encodeURIComponent(pack)}`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  state.rows = data.rows || [];
  state.notes = data.notes || {};
  state.index = 0;
  populateFilters();
  applyFilters();
}

function populateFilters() {
  const groups = ["all", ...new Set(state.rows.map((r) => r.review_group))];
  const tiers = ["all", ...new Set(state.rows.map((r) => r.tier))];
  els.groupFilter.innerHTML = groups.map((g) => `<option value="${g}">${g}</option>`).join("");
  els.tierFilter.innerHTML = tiers.map((t) => `<option value="${t}">${t}</option>`).join("");
}

function applyFilters() {
  const group = els.groupFilter.value || "all";
  const tier = els.tierFilter.value || "all";
  const search = els.searchInput.value.trim().toLowerCase();
  state.filtered = state.rows.filter((row) => {
    if (group !== "all" && row.review_group !== group) return false;
    if (tier !== "all" && row.tier !== tier) return false;
    if (search && !row.image_id.toLowerCase().includes(search)) return false;
    return true;
  });
  if (state.index >= state.filtered.length) state.index = 0;
  renderList();
  renderCurrent();
}

function renderList() {
  els.imageList.innerHTML = state.filtered
    .map((row, i) => {
      const note = state.notes[rowKey(row)] || {};
      const status = note.status ? `status: ${note.status}` : row.depth_guess_source || row.reason;
      const selected = i === state.index ? " selected" : "";
      const family = row.image_family ? ` - ${row.image_family}` : "";
      return `<button class="list-item${selected}" data-index="${i}">
        <strong>${row.image_id}</strong>
        <span class="tier-${row.tier}">depth ${depthLabel(row)} - ${row.submitted_scale_state || row.tier}</span>
        <span>${status || ""}${family}</span>
      </button>`;
    })
    .join("");
}

function currentRow() {
  return state.filtered[state.index] || null;
}

function renderCurrent() {
  const row = currentRow();
  if (!row) {
    els.title.textContent = "No rows";
    return;
  }
  const note = state.notes[rowKey(row)] || {};
  els.title.textContent = row.image_id;
  els.reviewImage.src = `/image/${encodeURIComponent(row.image_id)}`;
  els.depthGuess.textContent = depthLabel(row);
  els.depthOverlay.textContent = depthLabel(row);
  els.scaleState.textContent = row.submitted_scale_state || row.tier || "--";
  els.depthSource.textContent = row.depth_guess_source || "--";
  els.imageMeta.textContent = `${state.index + 1} / ${state.filtered.length}`;
  els.depthText.textContent = depthLabel(row);
  els.submittedState.textContent = row.submitted_scale_state || row.tier || "--";
  els.pxCm.textContent = fmt(row.scale_px_per_cm);
  els.pxMm.textContent = fmt(row.scale_px_per_mm);
  els.tickPx.textContent = fmt(row.tick_px_cm);
  els.rulerPx.textContent = fmt(row.ruler_px_cm);
  els.ocrDepthText.textContent = fmt(row.text_depth_mm, " mm");
  els.imageFamily.textContent = row.image_family || "--";
  els.scaleZ.textContent = fmt(row.scale_robust_z);
  els.reason.textContent = row.depth_guess_note || row.reason || "";
  els.note.textContent = row.note || "";
  els.oracleScale.value = note.oracle_scale_px_per_cm || "";
  els.oracleDepth.value = note.oracle_depth_mm || "";
  els.oracleTicks.value = note.oracle_ticks || "";
  els.oracleComment.value = note.comment || "";
  document.querySelectorAll(".status-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.status === note.status);
  });
  state.zoom = 1;
  applyZoom();
  renderList();
}

function applyZoom() {
  els.reviewImage.style.transform = `scale(${state.zoom})`;
  els.zoomResetBtn.textContent = `${Math.round(state.zoom * 100)}%`;
}

function move(delta) {
  if (!state.filtered.length) return;
  state.index = (state.index + delta + state.filtered.length) % state.filtered.length;
  renderCurrent();
}

function buildPayload(row) {
  return { image_id: row.image_id, ...(state.notes[rowKey(row)] || {}) };
}

function updateNote(status = null, options = {}) {
  const row = currentRow();
  if (!row) return null;
  const key = rowKey(row);
  const existing = state.notes[key] || {};
  state.notes[key] = {
    ...existing,
    status: status ?? existing.status ?? "",
    oracle_scale_px_per_cm: els.oracleScale.value,
    oracle_depth_mm: els.oracleDepth.value,
    oracle_ticks: els.oracleTicks.value,
    comment: els.oracleComment.value,
  };
  document.querySelectorAll(".status-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.status === state.notes[key].status);
  });
  const payload = buildPayload(row);
  if (!options.deferSave) scheduleSave(payload);
  return payload;
}

async function markStatus(status) {
  const row = currentRow();
  if (status === "correct" && row?.depth_guess_mm && !els.oracleDepth.value.trim()) {
    els.oracleDepth.value = String(Number(row.depth_guess_mm));
  }
  const payload = updateNote(status, { deferSave: status === "correct" });
  if (status === "correct" && payload) {
    clearTimeout(state.saveTimer);
    const saved = await savePayload(payload);
    if (saved) move(1);
  }
}

function scheduleSave(payload) {
  clearTimeout(state.saveTimer);
  els.saveState.textContent = "Autosaving...";
  state.saveTimer = setTimeout(() => savePayload(payload), 350);
}

async function savePayload(payload) {
  if (!payload) return false;
  els.saveState.textContent = "Saving...";
  const res = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  els.saveState.textContent = res.ok ? "Saved" : "Save failed";
  renderList();
  return res.ok;
}

els.packSelect.addEventListener("change", () => load(els.packSelect.value));
els.prevBtn.addEventListener("click", () => move(-1));
els.nextBtn.addEventListener("click", () => move(1));
els.groupFilter.addEventListener("change", applyFilters);
els.tierFilter.addEventListener("change", applyFilters);
els.searchInput.addEventListener("input", applyFilters);
els.imageList.addEventListener("click", (event) => {
  const item = event.target.closest(".list-item");
  if (!item) return;
  state.index = Number(item.dataset.index);
  renderCurrent();
});
els.zoomOutBtn.addEventListener("click", () => {
  state.zoom = Math.max(0.3, state.zoom - 0.2);
  applyZoom();
});
els.zoomInBtn.addEventListener("click", () => {
  state.zoom = Math.min(4, state.zoom + 0.2);
  applyZoom();
});
els.zoomResetBtn.addEventListener("click", () => {
  state.zoom = 1;
  applyZoom();
});
document.querySelectorAll(".status-btn").forEach((btn) => {
  btn.addEventListener("click", () => markStatus(btn.dataset.status));
});
[els.oracleScale, els.oracleDepth, els.oracleTicks, els.oracleComment].forEach((el) => {
  el.addEventListener("input", () => updateNote());
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) return;
  const key = event.key.toLowerCase();
  if (key === "arrowright" || key === "d") move(1);
  if (key === "arrowleft" || key === "a") move(-1);
  if (key === "q") markStatus("correct");
  if (key === "w") markStatus("wrong");
  if (key === "e") markStatus("unclear");
});

load().catch((err) => {
  els.title.textContent = "Load failed";
  els.reason.textContent = String(err);
});
