(function () {
  const TERMS = [
    ["pa_deg", "PA", "deg"],
    ["fl_mm", "FL", "mm"],
    ["mt_mm", "MT", "mm"],
  ];
  const LAYER_KEYS = [
    ["apo", "Apo mask"],
    ["fasc", "Fragment mask"],
    ["ignore", "Ignore mask"],
    ["diag", "Old diag"],
    ["scan", "Scan region"],
    ["imagefield", "Image field"],
    ["usfield", "US field"],
    ["boundary", "Model boundary"],
    ["spans", "Model spans"],
    ["orientation", "PA orientation"],
    ["scratch", "Scratch"],
  ];
  const TOOLS = [
    ["inspect", "Inspect"],
    ["upper", "Upper boundary"],
    ["lower", "Lower boundary"],
    ["ruler", "Straight ruler"],
    ["trial", "Trial FL"],
  ];

  const dom = {};
  const state = {
    rows: [],
    summary: [],
    metadata: {},
    session: { items: {}, updated_at: null },
    filtered: [],
    index: 0,
    activeModelId: "",
    activeTab: "models",
    activeTool: "inspect",
    zoom: 1,
    search: "",
    sort: "overall_desc",
    classFilters: new Set(),
    layerVisible: {
      apo: true,
      fasc: true,
      ignore: false,
      diag: false,
      scan: true,
      imagefield: true,
      usfield: true,
      boundary: true,
      spans: true,
      orientation: true,
      scratch: true,
    },
    pendingPoint: null,
    inspectedModelItem: null,
    saveTimer: null,
    reviewSaveTimer: null,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function initDom() {
    [
      "prevBtn", "nextBtn", "counter", "zoomOutBtn", "zoomResetBtn", "zoomInBtn", "saveStatus",
      "fitBtn", "searchInput", "sortSelect", "classFilters", "imageList", "deltaStrip", "stageScroll",
      "imageStack", "baseImage", "apoLayer", "fascLayer", "ignoreLayer", "diagLayer", "overlayCanvas",
      "modelCards", "layerToggles", "storyGate", "classTags", "classMatrix", "toolButtons",
      "undoScratchBtn", "clearScratchBtn", "scratchReadout", "labelQuality", "failureKind",
      "reviewNotes", "viewerNotes",
    ].forEach((id) => {
      dom[id] = $(id);
    });
  }

  function fmt(value, digits = 2) {
    return value === null || value === undefined || Number.isNaN(Number(value))
      ? "n/a"
      : Number(value).toFixed(digits);
  }

  function signed(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
    const n = Number(value);
    return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;
  }

  function termClass(value) {
    if (value === null || value === undefined || Math.abs(Number(value)) < 1e-9) return "neutral";
    return Number(value) > 0 ? "over" : "under";
  }

  function safeText(value) {
    return String(value ?? "");
  }

  function currentRow() {
    return state.filtered[state.index] || state.rows[0] || null;
  }

  function currentModel(row = currentRow()) {
    if (!row) return null;
    return row.models.find((m) => m.id === state.activeModelId) || row.models[0] || null;
  }

  function sessionItem(row = currentRow()) {
    if (!row) return {};
    const key = row.label_id || row.image_id;
    state.session.items ||= {};
    state.session.items[key] ||= {
      viewerNotes: "",
      scratch: { upper: [], lower: [], rulers: [], trials: [] },
    };
    state.session.items[key].scratch ||= { upper: [], lower: [], rulers: [], trials: [] };
    return state.session.items[key];
  }

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(`${url} failed: ${res.status}`);
    return res.json();
  }

  async function loadData() {
    const [review, session] = await Promise.all([
      fetchJson("/api/review"),
      fetchJson("/api/session").catch(() => ({ items: {}, updated_at: null })),
    ]);
    state.rows = (review.rows || []).map((row, index) => ({ ...row, __index: index }));
    state.summary = review.summary || [];
    state.metadata = review.metadata || {};
    state.session = session || { items: {}, updated_at: null };
    if (state.session.layerVisible && Object.keys(state.session.layerVisible).length) {
      state.layerVisible = { ...state.layerVisible, ...state.session.layerVisible };
    }
    state.filtered = state.rows.slice();
    const sessionModel = state.session.selectedModel || "";
    state.activeModelId = state.rows[0]?.models?.some((m) => m.id === sessionModel)
      ? sessionModel
      : (state.rows[0]?.models?.[0]?.id || "");
  }

  function applyFilters() {
    const needle = state.search.trim().toLowerCase();
    const activeClasses = [...state.classFilters];
    const selectedId = state.activeModelId;
    state.filtered = state.rows.filter((row) => {
      const hay = [
        row.image_id,
        row.label_id,
        row.source,
        ...(row.classes || []),
      ].join(" ").toLowerCase();
      if (needle && !hay.includes(needle)) return false;
      return activeClasses.every((name) => row.class_flags?.[name]);
    });
    state.filtered.sort((a, b) => {
      const am = a.models.find((m) => m.id === selectedId) || a.models[0] || {};
      const bm = b.models.find((m) => m.id === selectedId) || b.models[0] || {};
      if (state.sort === "image_id") return safeText(a.image_id).localeCompare(safeText(b.image_id));
      const key = state.sort === "pa_desc" ? "pa_deg" : state.sort === "fl_desc" ? "fl_mm" : state.sort === "mt_desc" ? "mt_mm" : "";
      const av = key ? Math.abs(Number(am.deltas?.[key] ?? -1)) : Number(am.overall_norm ?? -1);
      const bv = key ? Math.abs(Number(bm.deltas?.[key] ?? -1)) : Number(bm.overall_norm ?? -1);
      return bv - av;
    });
    if (state.index >= state.filtered.length) state.index = Math.max(0, state.filtered.length - 1);
  }

  function renderClassFilters() {
    dom.classFilters.replaceChildren();
    const names = state.metadata.class_names || [];
    if (!names.length) {
      dom.classFilters.append(empty("No class data loaded."));
      return;
    }
    names.forEach((name) => {
      const label = document.createElement("label");
      label.className = "toggle";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = state.classFilters.has(name);
      input.addEventListener("change", () => {
        if (input.checked) state.classFilters.add(name);
        else state.classFilters.delete(name);
        applyFilters();
        renderAll();
      });
      label.append(input, document.createTextNode(name.replaceAll("_", " ")));
      dom.classFilters.append(label);
    });
  }

  function renderImageList() {
    dom.imageList.replaceChildren();
    state.filtered.forEach((row, idx) => {
      const model = currentModel(row);
      const btn = document.createElement("button");
      btn.className = `image-item ${idx === state.index ? "active" : ""}`;
      btn.type = "button";
      btn.addEventListener("click", () => {
        state.index = idx;
        loadCurrentImage();
        renderAll();
      });
      const title = document.createElement("div");
      title.className = "image-title";
      title.innerHTML = `<span>${row.image_id}</span><span>${fmt(model?.overall_norm, 3)}</span>`;
      const sub = document.createElement("div");
      sub.className = "image-sub";
      sub.textContent = `image score | ${(row.classes || []).slice(0, 3).join(", ") || row.source || "unclassified"}`;
      btn.append(title, sub);
      dom.imageList.append(btn);
    });
  }

  function renderDeltaStrip() {
    const row = currentRow();
    const model = currentModel(row);
    dom.deltaStrip.replaceChildren();
    if (!row || !model) {
      dom.deltaStrip.append(empty("No row selected."));
      return;
    }
    const title = document.createElement("div");
    title.className = "delta-title";
    const bench = model.summary?.overall_norm;
    title.innerHTML = `<strong>${model.label}</strong><span>${row.image_id} | image score ${fmt(model.overall_norm, 3)} | benchmark avg ${fmt(bench, 3)}</span>`;
    dom.deltaStrip.append(title);
    TERMS.forEach(([key, label, unit]) => {
      const delta = model.deltas?.[key];
      const card = document.createElement("div");
      card.className = `delta-card ${termClass(delta)}`;
      card.innerHTML = `<span>${label} delta</span><b>${signed(delta)}</b><span>${unit} ${Number(delta || 0) >= 0 ? "over" : "under"}</span>`;
      dom.deltaStrip.append(card);
    });
    const matrix = document.createElement("div");
    matrix.className = "score-matrix";
    matrix.innerHTML = `
      <div></div><b>PA</b><b>FL</b><b>MT</b>
      <span>image pred</span>
      <strong>${fmt(model.predictions?.pa_deg, 2)}</strong>
      <strong>${fmt(model.predictions?.fl_mm, 2)}</strong>
      <strong>${fmt(model.predictions?.mt_mm, 2)}</strong>
      <span>image delta</span>
      <strong>${signed(model.deltas?.pa_deg, 2)}</strong>
      <strong>${signed(model.deltas?.fl_mm, 2)}</strong>
      <strong>${signed(model.deltas?.mt_mm, 2)}</strong>
      <span>benchmark avg</span>
      <strong>${fmt(model.summary?.pa_norm, 3)}</strong>
      <strong>${fmt(model.summary?.fl_norm, 3)}</strong>
      <strong>${fmt(model.summary?.mt_norm, 3)}</strong>
    `;
    dom.deltaStrip.append(matrix);
  }

  function renderModelCards() {
    const row = currentRow();
    dom.modelCards.replaceChildren();
    if (!row) return;
    row.models.forEach((model) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = `model-card ${model.id === state.activeModelId ? "active" : ""}`;
      card.style.setProperty("--model-color", model.color || "#38bdf8");
      card.addEventListener("click", () => {
        state.activeModelId = model.id;
        state.inspectedModelItem = null;
        state.session.selectedModel = model.id;
        scheduleSessionSave();
        renderAll();
        drawOverlay();
      });
      const stats = `
        <div></div><span>PA</span><span>FL</span><span>MT</span>
        <em>pred</em>
        <b>${fmt(model.predictions?.pa_deg, 1)}</b>
        <b>${fmt(model.predictions?.fl_mm, 1)}</b>
        <b>${fmt(model.predictions?.mt_mm, 1)}</b>
        <em>delta</em>
        <b>${signed(model.deltas?.pa_deg, 1)}</b>
        <b>${signed(model.deltas?.fl_mm, 1)}</b>
        <b>${signed(model.deltas?.mt_mm, 1)}</b>
      `;
      card.innerHTML = `
        <div class="model-head"><strong>${model.label}</strong><span class="model-kind">${model.kind || "model"}</span></div>
        <div class="model-score-line">image ${fmt(model.overall_norm, 3)} · mean ${fmt(model.summary?.overall_norm, 3)}</div>
        <div class="model-matrix">${stats}</div>
      `;
      card.querySelector(".model-score-line").textContent = `image score ${fmt(model.overall_norm, 3)} | benchmark avg ${fmt(model.summary?.overall_norm, 3)}`;
      dom.modelCards.append(card);
    });
  }

  function renderLayerToggles() {
    dom.layerToggles.replaceChildren();
    LAYER_KEYS.forEach(([key, label]) => {
      const item = document.createElement("label");
      item.className = "toggle";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = state.layerVisible[key];
      input.addEventListener("change", () => {
        state.layerVisible[key] = input.checked;
        state.session.layerVisible = { ...state.layerVisible };
        scheduleSessionSave();
        updateLayerVisibility();
        drawOverlay();
      });
      item.append(input, document.createTextNode(label));
      dom.layerToggles.append(item);
    });
  }

  function renderClasses() {
    const row = currentRow();
    const model = currentModel(row);
    dom.storyGate.replaceChildren();
    dom.classTags.replaceChildren();
    dom.classMatrix.replaceChildren();
    if (!row || !model) return;
    const gate = document.createElement("div");
    gate.innerHTML = `<strong>${model.label}</strong><br>${model.gate_reason || model.description || "No story gate for this model."}<br><br><strong>Overlay:</strong> ${model.geometry_note || "No geometry note."}`;
    dom.storyGate.append(gate);
    const tags = row.classes || [];
    if (!tags.length) dom.classTags.append(empty("No active class tags."));
    tags.forEach((name) => {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = name.replaceAll("_", " ");
      dom.classTags.append(tag);
    });
    renderClassMatrix(tags[0] || "");
  }

  function renderClassMatrix(className) {
    const rows = (state.metadata.class_feature_scores || []).filter((row) => row.class === className).slice(0, 12);
    if (!className || !rows.length) {
      dom.classMatrix.append(empty("Select an image with class data to see class-level feature scores."));
      return;
    }
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Variant</th><th>All</th><th>PA</th><th>FL</th><th>MT</th></tr></thead>";
    const body = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.variant}</td><td>${fmt(row.overall, 3)}</td><td>${fmt(row.pa, 3)}</td><td>${fmt(row.fl, 3)}</td><td>${fmt(row.mt, 3)}</td>`;
      body.append(tr);
    });
    table.append(body);
    dom.classMatrix.append(table);
  }

  function renderTools() {
    dom.toolButtons.replaceChildren();
    TOOLS.forEach(([key, label]) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = label;
      btn.className = key === state.activeTool ? "active" : "";
      btn.addEventListener("click", () => {
        state.activeTool = key;
        state.pendingPoint = null;
        renderTools();
      });
      dom.toolButtons.append(btn);
    });
    renderScratchReadout();
  }

  function renderScratchReadout() {
    const row = currentRow();
    const item = sessionItem(row);
    const scratch = item.scratch || {};
    const trialStats = computeTrialStats();
    const ruler = (scratch.rulers || []).at(-1);
    const model = currentModel(row);
    const wave = model?.diagnostics?.wave_non_crossing;
    const parts = [
      `<strong>Scale:</strong> ${fmt(row?.scale_px_per_mm, 3)} px/mm (${row?.calibration_method || "unknown"})`,
      `<strong>Upper points:</strong> ${(scratch.upper || []).length}`,
      `<strong>Lower points:</strong> ${(scratch.lower || []).length}`,
      `<strong>Trial lines:</strong> ${(scratch.trials || []).length}`,
    ];
    if (wave) {
      parts.push(`<strong>Wave consensus:</strong> ${fmt(wave.consensus_angle_deg, 2)} deg; <strong>order:</strong> ${wave.order}; changed ${wave.n_changed}; seed inverse ${wave.n_seed_opposite_sign || 0}; raw crossing lines ${wave.n_raw_crossing}; crossing pairs ${wave.raw_crossing_pairs} -> ${wave.remaining_crossing_pairs}; cascade steps ${wave.cascade_steps}`);
    }
    if (state.inspectedModelItem) {
      const inspected = state.inspectedModelItem;
      const moveDir = inspected.rawConsensusDev !== null && inspected.correctedConsensusDev !== null
        ? (inspected.correctedConsensusDev <= inspected.rawConsensusDev + 0.05 ? "toward/neutral" : "away")
        : "n/a";
      parts.push(
        `<strong>Inspected model line:</strong> ${inspected.fragId || `#${inspected.index + 1}`} ${inspected.modelLabel}`,
        `<strong>Raw angle:</strong> ${fmt(inspected.rawAngle, 2)} deg; <strong>corrected:</strong> ${fmt(inspected.correctedAngle, 2)} deg; <strong>delta:</strong> ${signed(inspected.deltaAngle, 2)} deg`,
        `<strong>Raw theta:</strong> ${fmt(inspected.rawTheta, 2)} deg; <strong>corrected theta:</strong> ${fmt(inspected.correctedTheta, 2)} deg`,
        `<strong>Consensus:</strong> ${fmt(inspected.consensusAngle, 2)} deg; <strong>raw dev:</strong> ${fmt(inspected.rawConsensusDev, 2)} deg; <strong>corrected dev:</strong> ${fmt(inspected.correctedConsensusDev, 2)} deg; <strong>move:</strong> ${moveDir}`,
        `<strong>Raw FL:</strong> ${fmt(inspected.rawFl, 2)} mm; <strong>corrected FL:</strong> ${fmt(inspected.correctedFl, 2)} mm; <strong>raw crosses:</strong> ${inspected.rawCrosses ?? 0}`,
        `<strong>Corrected support:</strong> visible ${fmt(inspected.visibleFrac, 2)}; US field ${fmt(inspected.usFrac, 2)}; area x US ${fmt(inspected.areaUsWeight, 1)}; area x US x visible ${fmt(inspected.areaUsVisibleWeight, 1)}`,
        `<strong>Raw support:</strong> visible ${fmt(inspected.rawVisibleFrac, 2)}; US field ${fmt(inspected.rawUsFrac, 2)}; area x US ${fmt(inspected.rawAreaUsWeight, 1)}`,
        `<strong>Other:</strong> scan ${fmt(inspected.scanFrac, 2)}; cascade steps ${inspected.cascadeSteps ?? 0}; reason ${inspected.reason || "cascade/raw"}; id ${inspected.fragMatch || "native"}`
      );
    }
    if (ruler) {
      const lenPx = dist(ruler[0], ruler[1]);
      const lenMm = row?.scale_px_per_mm ? lenPx / row.scale_px_per_mm : null;
      parts.push(`<strong>Last ruler:</strong> ${fmt(lenPx, 1)} px / ${fmt(lenMm, 2)} mm`);
    }
    if (trialStats) {
      parts.push(`<strong>Scratch PA median:</strong> ${fmt(trialStats.pa, 2)} deg`);
      parts.push(`<strong>Scratch FL median:</strong> ${fmt(trialStats.flMm, 2)} mm (${fmt(trialStats.flPx, 1)} px)`);
    }
    dom.scratchReadout.innerHTML = parts.join("<br>");
  }

  function renderNotes() {
    const row = currentRow();
    if (!row) return;
    const item = sessionItem(row);
    const review = row.review || {};
    dom.labelQuality.value = review.label_quality || "";
    dom.failureKind.value = review.failure_kind || "";
    dom.reviewNotes.value = review.notes || "";
    dom.viewerNotes.value = item.viewerNotes || "";
  }

  function renderAll() {
    dom.counter.textContent = `${state.filtered.length ? state.index + 1 : 0} / ${state.filtered.length}`;
    renderImageList();
    renderDeltaStrip();
    renderModelCards();
    renderLayerToggles();
    renderClasses();
    renderTools();
    renderNotes();
    updateLayerVisibility();
  }

  function empty(text) {
    const div = document.createElement("div");
    div.className = "empty";
    div.textContent = text;
    return div;
  }

  function updateLayerVisibility() {
    dom.apoLayer.style.display = state.layerVisible.apo && dom.apoLayer.dataset.available === "1" ? "block" : "none";
    dom.fascLayer.style.display = state.layerVisible.fasc && dom.fascLayer.dataset.available === "1" ? "block" : "none";
    dom.ignoreLayer.style.display = state.layerVisible.ignore && dom.ignoreLayer.dataset.available === "1" ? "block" : "none";
    dom.diagLayer.style.display = state.layerVisible.diag && dom.diagLayer.dataset.available === "1" ? "block" : "none";
  }

  function loadCurrentImage() {
    const row = currentRow();
    if (!row) return;
    state.inspectedModelItem = null;
    if (!row.models.some((m) => m.id === state.activeModelId)) {
      state.activeModelId = row.models[0]?.id || "";
      state.session.selectedModel = state.activeModelId;
    }
    const idx = row.__index;
    dom.baseImage.onload = () => {
      resizeCanvas();
      fitToView();
      drawOverlay();
    };
    dom.baseImage.src = `/image/${idx}`;
    setLayer(dom.apoLayer, `/label/${idx}/apo`);
    setLayer(dom.fascLayer, `/label/${idx}/fasc`);
    setLayer(dom.ignoreLayer, `/label/${idx}/ignore`);
    setLayer(dom.diagLayer, `/label/${idx}/diag`);
  }

  function setLayer(img, src) {
    img.dataset.available = "0";
    img.style.display = "none";
    img.onload = () => {
      img.dataset.available = "1";
      updateLayerVisibility();
    };
    img.onerror = () => {
      img.dataset.available = "0";
      img.removeAttribute("src");
      img.style.display = "none";
    };
    img.src = src;
  }

  function resizeCanvas() {
    const w = dom.baseImage.naturalWidth || dom.baseImage.width;
    const h = dom.baseImage.naturalHeight || dom.baseImage.height;
    dom.overlayCanvas.width = w;
    dom.overlayCanvas.height = h;
    dom.overlayCanvas.style.width = `${w}px`;
    dom.overlayCanvas.style.height = `${h}px`;
    dom.imageStack.style.transform = `scale(${state.zoom})`;
  }

  function drawOverlay() {
    const canvas = dom.overlayCanvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const model = currentModel();
    if (state.layerVisible.scan) drawRegionBox(ctx, currentRow()?.scan_region, "#ffffff", "scan region", true);
    if (state.layerVisible.imagefield) drawRegionBox(ctx, currentRow()?.image_field, "#38bdf8", "image field", false);
    if (state.layerVisible.usfield) drawRegionBox(ctx, currentRow()?.us_field, "#22c55e", "US field", false);
    if (model && state.layerVisible.boundary) drawBoundary(ctx, model);
    if (model && state.layerVisible.spans) drawSpans(ctx, model);
    if (model) drawModelDiagnostics(ctx, model);
    if (state.layerVisible.scratch) drawScratch(ctx);
    drawInspectedMarker(ctx);
    renderScratchReadout();
  }

  function pointSegmentDistance(p, a, b) {
    const vx = b.x - a.x;
    const vy = b.y - a.y;
    const wx = p.x - a.x;
    const wy = p.y - a.y;
    const c1 = vx * wx + vy * wy;
    if (c1 <= 0) return Math.hypot(p.x - a.x, p.y - a.y);
    const c2 = vx * vx + vy * vy;
    if (c2 <= c1) return Math.hypot(p.x - b.x, p.y - b.y);
    const t = c1 / c2;
    return Math.hypot(p.x - (a.x + t * vx), p.y - (a.y + t * vy));
  }

  function nearestModelItem(point) {
    const model = currentModel();
    if (!model) return null;
    const candidates = [];
    const wave = model.diagnostics?.wave_non_crossing || null;
    (model.projection_spans || []).forEach((span, index) => {
      candidates.push({
        index,
        fragId: span.frag_id || null,
        distance: pointSegmentDistance(point, { x: span.x1, y: span.y1 }, { x: span.x2, y: span.y2 }),
        modelLabel: model.label,
        rawAngle: span.raw_angle_deg ?? span.angle_deg ?? null,
        correctedAngle: span.corrected_angle_deg ?? span.angle_deg ?? null,
        deltaAngle: span.delta_angle_deg ?? 0,
        rawTheta: span.raw_theta_deg ?? null,
        correctedTheta: span.corrected_theta_deg ?? null,
        consensusAngle: wave?.consensus_angle_deg ?? null,
        rawConsensusDev: span.raw_consensus_dev_deg ?? null,
        correctedConsensusDev: span.corrected_consensus_dev_deg ?? null,
        rawFl: span.raw_fl_mm ?? null,
        correctedFl: span.fl_mm ?? null,
        rawCrosses: span.raw_crosses ?? 0,
        visibleFrac: span.visible_frac ?? null,
        rawVisibleFrac: span.raw_visible_frac ?? null,
        usFrac: span.us_field_frac ?? null,
        rawUsFrac: span.raw_us_field_frac ?? null,
        areaUsWeight: span.area_us_weight ?? null,
        rawAreaUsWeight: span.raw_area_us_weight ?? null,
        areaUsVisibleWeight: span.area_us_visible_weight ?? null,
        scanFrac: span.strict_scan_region_frac ?? null,
        cascadeSteps: span.cascade_steps ?? 0,
        reason: span.correction_reason ?? null,
        fragMatch: span.frag_match ?? null,
        x: (span.x1 + span.x2) / 2,
        y: (span.y1 + span.y2) / 2,
      });
      if (span.raw_x1 !== undefined) {
        candidates.push({
          index,
          fragId: span.frag_id || null,
          distance: pointSegmentDistance(point, { x: span.raw_x1, y: span.raw_y1 }, { x: span.raw_x2, y: span.raw_y2 }),
          modelLabel: `${model.label} raw`,
          rawAngle: span.raw_angle_deg ?? span.angle_deg ?? null,
          correctedAngle: span.corrected_angle_deg ?? span.angle_deg ?? null,
          deltaAngle: span.delta_angle_deg ?? 0,
          rawTheta: span.raw_theta_deg ?? null,
          correctedTheta: span.corrected_theta_deg ?? null,
          consensusAngle: wave?.consensus_angle_deg ?? null,
          rawConsensusDev: span.raw_consensus_dev_deg ?? null,
          correctedConsensusDev: span.corrected_consensus_dev_deg ?? null,
          rawFl: span.raw_fl_mm ?? null,
          correctedFl: span.fl_mm ?? null,
          rawCrosses: span.raw_crosses ?? 0,
          visibleFrac: span.visible_frac ?? null,
          rawVisibleFrac: span.raw_visible_frac ?? null,
          usFrac: span.us_field_frac ?? null,
          rawUsFrac: span.raw_us_field_frac ?? null,
          areaUsWeight: span.area_us_weight ?? null,
          rawAreaUsWeight: span.raw_area_us_weight ?? null,
          areaUsVisibleWeight: span.area_us_visible_weight ?? null,
          scanFrac: span.strict_scan_region_frac ?? null,
          cascadeSteps: span.cascade_steps ?? 0,
          reason: span.correction_reason ?? null,
          fragMatch: span.frag_match ?? null,
          x: (span.raw_x1 + span.raw_x2) / 2,
          y: (span.raw_y1 + span.raw_y2) / 2,
        });
      }
    });
    (wave?.items || []).forEach((item, index) => {
      const seg = item.corrected_span;
      if (!seg) return;
      candidates.push({
        index,
        fragId: item.frag_id || null,
        distance: pointSegmentDistance(point, { x: seg.x1, y: seg.y1 }, { x: seg.x2, y: seg.y2 }),
        modelLabel: model.label,
        rawAngle: item.raw_angle,
        correctedAngle: item.corrected_angle,
        deltaAngle: item.delta_angle_deg,
        rawTheta: item.raw_theta_deg ?? null,
        correctedTheta: item.corrected_theta_deg ?? null,
        consensusAngle: wave?.consensus_angle_deg ?? null,
        rawConsensusDev: item.raw_consensus_dev_deg ?? null,
        correctedConsensusDev: item.corrected_consensus_dev_deg ?? null,
        rawFl: item.raw_span?.fl_mm,
        correctedFl: item.corrected_span?.fl_mm,
        rawCrosses: item.raw_crosses,
        visibleFrac: item.visible_frac ?? null,
        rawVisibleFrac: item.raw_visible_frac ?? null,
        usFrac: item.us_field_frac ?? null,
        rawUsFrac: item.raw_us_field_frac ?? null,
        areaUsWeight: item.area_us_weight ?? null,
        rawAreaUsWeight: item.raw_area_us_weight ?? null,
        areaUsVisibleWeight: item.area_us_visible_weight ?? null,
        scanFrac: item.strict_scan_region_frac ?? null,
        cascadeSteps: item.cascade_steps ?? 0,
        reason: item.correction_reason ?? null,
        x: item.cx,
        y: item.cy,
      });
    });
    const best = candidates.sort((a, b) => a.distance - b.distance)[0];
    return best && best.distance <= 28 ? best : null;
  }

  function drawInspectedMarker(ctx) {
    const item = state.inspectedModelItem;
    if (!item) return;
    ctx.save();
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(item.x, item.y, 8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.font = "14px system-ui";
    ctx.fillText(item.fragId || `#${item.index + 1}`, item.x + 10, item.y - 10);
    ctx.restore();
  }

  function drawBoundary(ctx, model) {
    const boundary = model.boundary;
    if (!boundary) return;
    ctx.save();
    ctx.lineWidth = 3;
    drawBoundaryPart(ctx, boundary.top, model.color || "#d665d6");
    drawBoundaryPart(ctx, boundary.deep, "#facc15");
    ctx.restore();
  }

  function drawBoundaryPart(ctx, part, color) {
    if (!part) return;
    ctx.strokeStyle = color;
    ctx.beginPath();
    if (part.type === "piecewise" && Array.isArray(part.points)) {
      part.points.forEach((p, idx) => {
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
    } else if (typeof part.slope === "number") {
      ctx.moveTo(0, part.intercept);
      ctx.lineTo(dom.overlayCanvas.width, part.slope * dom.overlayCanvas.width + part.intercept);
    }
    ctx.stroke();
  }

  function drawSpans(ctx, model) {
    const spans = model.projection_spans || [];
    ctx.save();
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.82;
    spans.forEach((span) => {
      if (span.raw_x1 !== undefined) {
        ctx.save();
        ctx.strokeStyle = span.raw_crosses > 0 ? "#fb7185" : "#94a3b8";
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.62;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(span.raw_x1, span.raw_y1);
        ctx.lineTo(span.raw_x2, span.raw_y2);
        ctx.stroke();
        ctx.restore();
      }
      ctx.strokeStyle = span.changed ? "#22d3ee" : (span.gap_index === 1 ? "#f97316" : span.gap_index === 2 ? "#22c55e" : (model.color || "#38bdf8"));
      const support = span.us_field_frac ?? span.strict_scan_region_frac ?? span.visible_frac;
      ctx.globalAlpha = support === undefined ? 0.82 : Math.max(0.25, Math.min(0.95, 0.25 + Number(support) * 0.7));
      ctx.lineWidth = span.changed ? 3.5 : (support === undefined ? 2 : 1 + Number(support) * 3.2);
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(span.x1, span.y1);
      ctx.lineTo(span.x2, span.y2);
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawRegionBox(ctx, region, color, label, dashed) {
    if (!region) return;
    ctx.save();
    ctx.strokeStyle = color;
    if (dashed) ctx.setLineDash([9, 6]);
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.9;
    ctx.strokeRect(region.x, region.y, region.w, region.h);
    ctx.fillStyle = dashed ? "rgba(255, 255, 255, 0.045)" : "rgba(56, 189, 248, 0.055)";
    ctx.fillRect(region.x, region.y, region.w, region.h);
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = "13px system-ui";
    ctx.fillText(label, region.x + 8, Math.max(16, region.y + 18));
    ctx.restore();
  }

  function drawPolyline(ctx, points, color, width = 2) {
    if (!Array.isArray(points) || points.length < 2) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    points.forEach((p, idx) => {
      if (idx === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.restore();
  }

  function boundaryY(part, x) {
    if (!part) return null;
    if (part.type === "piecewise" && Array.isArray(part.points) && part.points.length) {
      const pts = part.points;
      if (x <= pts[0].x) return lineEval(lineFromPoints(pts[0], pts[1]), x);
      if (x >= pts[pts.length - 1].x) return lineEval(lineFromPoints(pts[pts.length - 2], pts[pts.length - 1]), x);
      for (let i = 0; i < pts.length - 1; i += 1) {
        const lo = Math.min(pts[i].x, pts[i + 1].x);
        const hi = Math.max(pts[i].x, pts[i + 1].x);
        if (x >= lo && x <= hi) return lineEval(lineFromPoints(pts[i], pts[i + 1]), x);
      }
    }
    if (typeof part.slope === "number") return part.slope * x + part.intercept;
    return null;
  }

  function lineEval(line, x) {
    return line ? line.slope * x + line.intercept : null;
  }

  function drawModelDiagnostics(ctx, model) {
    const d = model.diagnostics || {};
    if (state.layerVisible.orientation) drawOrientationDiagnostics(ctx, d.pa_orientation || []);
    if (state.layerVisible.orientation) drawWaveLabels(ctx, d.wave_non_crossing);
    if (state.layerVisible.boundary) {
      drawPolyline(ctx, d.lower_smooth, "#fb7185", 2);
      drawPolyline(ctx, d.lower_quartile, "#f97316", 3);
    }
    if (!state.layerVisible.boundary || !Array.isArray(d.mt_positions) || !model.boundary) return;
    ctx.save();
    ctx.strokeStyle = "#fef08a";
    ctx.lineWidth = 3;
    ctx.setLineDash([4, 4]);
    d.mt_positions.forEach((x) => {
      const top = boundaryY(model.boundary.top, Number(x));
      const deep = boundaryY(model.boundary.deep, Number(x));
      if (top === null || deep === null) return;
      ctx.beginPath();
      ctx.moveTo(Number(x), top);
      ctx.lineTo(Number(x), deep);
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawOrientationDiagnostics(ctx, fragments) {
    if (!Array.isArray(fragments) || !fragments.length) return;
    ctx.save();
    fragments.forEach((frag) => {
      const raw = frag.raw_segment;
      const corrected = frag.corrected_segment;
      if (raw) {
        ctx.globalAlpha = frag.changed ? 0.8 : 0.35;
        ctx.strokeStyle = frag.changed ? "#fb7185" : "#cbd5e1";
        ctx.lineWidth = frag.changed ? 2 : 1.5;
        ctx.beginPath();
        ctx.moveTo(raw.x1, raw.y1);
        ctx.lineTo(raw.x2, raw.y2);
        ctx.stroke();
      }
      if (corrected && frag.changed) {
        ctx.globalAlpha = 0.95;
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(corrected.x1, corrected.y1);
        ctx.lineTo(corrected.x2, corrected.y2);
        ctx.stroke();
      }
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = frag.changed ? "#22d3ee" : "#e2e8f0";
      ctx.beginPath();
      ctx.arc(frag.cx, frag.cy, frag.changed ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  function drawWaveLabels(ctx, wave) {
    if (!wave || !Array.isArray(wave.items) || !wave.items.length) return;
    ctx.save();
    ctx.font = "12px system-ui";
    ctx.fillStyle = "#22d3ee";
    ctx.globalAlpha = 0.95;
    wave.items.forEach((item, idx) => {
      if (!item.changed) return;
      ctx.fillText(item.frag_id || `${idx + 1}`, item.cx + 5, item.cy - 5);
    });
    ctx.restore();
  }

  function drawScratch(ctx) {
    const scratch = sessionItem().scratch || {};
    ctx.save();
    drawPointList(ctx, scratch.upper || [], "#f472b6", true);
    drawPointList(ctx, scratch.lower || [], "#fde047", true);
    drawSegments(ctx, scratch.rulers || [], "#f8fafc");
    drawSegments(ctx, scratch.trials || [], "#22c55e");
    if (state.pendingPoint) {
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(state.pendingPoint.x, state.pendingPoint.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawPointList(ctx, points, color, connect) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    points.forEach((p, idx) => {
      if (connect && idx > 0) {
        const prev = points[idx - 1];
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function drawSegments(ctx, segments, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    segments.forEach((seg) => {
      if (!seg[0] || !seg[1]) return;
      ctx.beginPath();
      ctx.moveTo(seg[0].x, seg[0].y);
      ctx.lineTo(seg[1].x, seg[1].y);
      ctx.stroke();
    });
  }

  function canvasPoint(evt) {
    const rect = dom.overlayCanvas.getBoundingClientRect();
    return {
      x: (evt.clientX - rect.left) / state.zoom,
      y: (evt.clientY - rect.top) / state.zoom,
    };
  }

  function handleCanvasClick(evt) {
    const item = sessionItem();
    const scratch = item.scratch;
    const point = canvasPoint(evt);
    if (state.activeTool === "inspect") {
      state.inspectedModelItem = nearestModelItem(point);
      drawOverlay();
      renderScratchReadout();
      return;
    }
    if (state.activeTool === "upper" || state.activeTool === "lower") {
      scratch[state.activeTool].push(point);
      afterScratchChange();
      return;
    }
    if (state.activeTool === "ruler" || state.activeTool === "trial") {
      if (!state.pendingPoint) {
        state.pendingPoint = point;
      } else {
        const key = state.activeTool === "ruler" ? "rulers" : "trials";
        scratch[key].push([state.pendingPoint, point]);
        state.pendingPoint = null;
      }
      afterScratchChange();
    }
  }

  function afterScratchChange() {
    scheduleSessionSave();
    drawOverlay();
  }

  function undoScratch() {
    const scratch = sessionItem().scratch;
    if (state.pendingPoint) {
      state.pendingPoint = null;
    } else if (state.activeTool === "upper" || state.activeTool === "lower") {
      scratch[state.activeTool].pop();
    } else if (state.activeTool === "ruler") {
      scratch.rulers.pop();
    } else if (state.activeTool === "trial") {
      scratch.trials.pop();
    }
    afterScratchChange();
  }

  function clearScratch() {
    sessionItem().scratch = { upper: [], lower: [], rulers: [], trials: [] };
    state.pendingPoint = null;
    afterScratchChange();
  }

  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function lineFromPoints(a, b) {
    const dx = b.x - a.x;
    if (Math.abs(dx) < 1e-9) return null;
    const slope = (b.y - a.y) / dx;
    return { slope, intercept: a.y - slope * a.x };
  }

  function lineHit(a, b) {
    if (!a || !b) return null;
    const denom = a.slope - b.slope;
    if (Math.abs(denom) < 1e-9) return null;
    const x = (b.intercept - a.intercept) / denom;
    return { x, y: a.slope * x + a.intercept };
  }

  function boundaryFromScratchOrModel() {
    const scratch = sessionItem().scratch || {};
    const model = currentModel();
    const upperLine = scratch.upper?.length >= 2 ? lineFromPoints(scratch.upper[0], scratch.upper.at(-1)) : null;
    const lowerLine = scratch.lower?.length >= 2 ? lineFromPoints(scratch.lower[0], scratch.lower.at(-1)) : null;
    return {
      top: upperLine || model?.boundary?.top || null,
      deep: lowerLine || model?.boundary?.deep || null,
    };
  }

  function hitTop(line, top) {
    if (!top) return null;
    if (top.type === "piecewise" && Array.isArray(top.points)) {
      const hits = [];
      for (let i = 0; i < top.points.length - 1; i += 1) {
        const seg = lineFromPoints(top.points[i], top.points[i + 1]);
        const hit = lineHit(line, seg);
        if (!hit) continue;
        const lo = Math.min(top.points[i].x, top.points[i + 1].x) - 10;
        const hi = Math.max(top.points[i].x, top.points[i + 1].x) + 10;
        hits.push({ penalty: hit.x >= lo && hit.x <= hi ? 0 : 1, hit });
      }
      return hits.sort((a, b) => a.penalty - b.penalty)[0]?.hit || null;
    }
    return lineHit(line, top);
  }

  function angleToDeep(line, deep) {
    if (!line || !deep) return null;
    let angle = (Math.atan(line.slope) - Math.atan(deep.slope)) * 180 / Math.PI;
    while (angle <= -90) angle += 180;
    while (angle > 90) angle -= 180;
    return Math.abs(angle);
  }

  function median(xs) {
    if (!xs.length) return null;
    const sorted = xs.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function computeTrialStats() {
    const row = currentRow();
    const scratch = sessionItem(row).scratch || {};
    const boundary = boundaryFromScratchOrModel();
    if (!boundary.top || !boundary.deep || !(scratch.trials || []).length) return null;
    const flPx = [];
    const angles = [];
    scratch.trials.forEach((seg) => {
      const line = lineFromPoints(seg[0], seg[1]);
      if (!line) return;
      const top = hitTop(line, boundary.top);
      const deep = lineHit(line, boundary.deep);
      if (!top || !deep) return;
      flPx.push(dist(top, deep));
      const angle = angleToDeep(line, boundary.deep);
      if (angle !== null) angles.push(angle);
    });
    const px = median(flPx);
    const pa = median(angles);
    if (px === null && pa === null) return null;
    return {
      flPx: px,
      flMm: row?.scale_px_per_mm && px !== null ? px / row.scale_px_per_mm : null,
      pa,
    };
  }

  function scheduleSessionSave() {
    dom.saveStatus.textContent = "autosaving";
    dom.saveStatus.className = "save-status dirty";
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(saveSession, 650);
  }

  async function saveSession() {
    try {
      const res = await fetchJson("/api/save_session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.session),
      });
      dom.saveStatus.textContent = `saved ${new Date(res.updated_at).toLocaleTimeString()}`;
      dom.saveStatus.className = "save-status saved";
    } catch (err) {
      dom.saveStatus.textContent = "session save failed";
      dom.saveStatus.className = "save-status error";
    }
  }

  function scheduleReviewSave() {
    const row = currentRow();
    if (!row) return;
    row.review ||= {};
    row.review.label_quality = dom.labelQuality.value;
    row.review.failure_kind = dom.failureKind.value;
    row.review.notes = dom.reviewNotes.value;
    clearTimeout(state.reviewSaveTimer);
    state.reviewSaveTimer = setTimeout(saveReview, 700);
  }

  async function saveReview() {
    const row = currentRow();
    if (!row) return;
    try {
      await fetchJson("/api/save_review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          index: row.__index,
          label_quality: dom.labelQuality.value,
          failure_kind: dom.failureKind.value,
          notes: dom.reviewNotes.value,
        }),
      });
      dom.saveStatus.textContent = "review saved";
      dom.saveStatus.className = "save-status saved";
    } catch (err) {
      dom.saveStatus.textContent = "review save failed";
      dom.saveStatus.className = "save-status error";
    }
  }

  function bindEvents() {
    dom.prevBtn.addEventListener("click", () => {
      state.index = state.filtered.length ? (state.index - 1 + state.filtered.length) % state.filtered.length : 0;
      loadCurrentImage();
      renderAll();
    });
    dom.nextBtn.addEventListener("click", () => {
      state.index = state.filtered.length ? (state.index + 1) % state.filtered.length : 0;
      loadCurrentImage();
      renderAll();
    });
    dom.zoomOutBtn.addEventListener("click", () => setZoom(state.zoom / 1.15));
    dom.zoomInBtn.addEventListener("click", () => setZoom(state.zoom * 1.15));
    dom.zoomResetBtn.addEventListener("click", () => setZoom(1));
    dom.fitBtn.addEventListener("click", fitToView);
    dom.searchInput.addEventListener("input", () => {
      state.search = dom.searchInput.value;
      applyFilters();
      state.index = 0;
      loadCurrentImage();
      renderAll();
    });
    dom.sortSelect.addEventListener("change", () => {
      state.sort = dom.sortSelect.value;
      applyFilters();
      state.index = 0;
      loadCurrentImage();
      renderAll();
    });
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => setTab(tab.dataset.tab));
    });
    dom.overlayCanvas.addEventListener("click", handleCanvasClick);
    dom.undoScratchBtn.addEventListener("click", undoScratch);
    dom.clearScratchBtn.addEventListener("click", clearScratch);
    [dom.labelQuality, dom.failureKind, dom.reviewNotes].forEach((node) => {
      node.addEventListener("input", scheduleReviewSave);
      node.addEventListener("change", scheduleReviewSave);
    });
    dom.viewerNotes.addEventListener("input", () => {
      sessionItem().viewerNotes = dom.viewerNotes.value;
      scheduleSessionSave();
    });
    window.addEventListener("keydown", (evt) => {
      if (evt.target && ["INPUT", "TEXTAREA", "SELECT"].includes(evt.target.tagName)) return;
      if (evt.key === "ArrowLeft") dom.prevBtn.click();
      if (evt.key === "ArrowRight") dom.nextBtn.click();
      if (evt.key.toLowerCase() === "z") undoScratch();
    });
  }

  function setZoom(value) {
    state.zoom = Math.min(3, Math.max(0.25, value));
    dom.zoomResetBtn.textContent = `${Math.round(state.zoom * 100)}%`;
    resizeCanvas();
    drawOverlay();
  }

  function fitToView() {
    const w = dom.baseImage.naturalWidth || dom.baseImage.width;
    const h = dom.baseImage.naturalHeight || dom.baseImage.height;
    if (!w || !h) return;
    const availW = Math.max(220, dom.stageScroll.clientWidth - 36);
    const availH = Math.max(220, dom.stageScroll.clientHeight - 36);
    setZoom(Math.min(1.5, availW / w, availH / h));
  }

  function setTab(name) {
    state.activeTab = name;
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
    document.querySelectorAll(".tab-page").forEach((page) => page.classList.toggle("active", page.id === `tab-${name}`));
  }

  async function main() {
    initDom();
    bindEvents();
    try {
      await loadData();
      renderClassFilters();
      applyFilters();
      loadCurrentImage();
      renderAll();
      dom.saveStatus.textContent = "ready";
      dom.saveStatus.className = "save-status saved";
    } catch (err) {
      dom.saveStatus.textContent = "load failed";
      dom.saveStatus.className = "save-status error";
      dom.deltaStrip.textContent = err.message;
      console.error(err);
    }
  }

  main();
})();
