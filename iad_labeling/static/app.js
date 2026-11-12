(() => {
  "use strict";

  const SIZE = 64;
  const PIXELS = SIZE * SIZE;
  const byId = (id) => document.getElementById(id);
  const elements = {
    annotator: byId("annotator"),
    loadAnnotator: byId("load-annotator"),
    statusFilter: byId("status-filter"),
    sampleList: byId("sample-list"),
    progress: byId("progress"),
    previous: byId("previous"),
    next: byId("next"),
    sampleName: byId("sample-name"),
    sampleMeta: byId("sample-meta"),
    emptyState: byId("empty-state"),
    spectrogramCanvas: byId("spectrogram-canvas"),
    proposalCanvas: byId("proposal-canvas"),
    annotationCanvas: byId("annotation-canvas"),
    proposalToggle: byId("proposal-toggle"),
    proposalNote: byId("proposal-note"),
    brushSize: byId("brush-size"),
    brushSizeOutput: byId("brush-size-output"),
    undo: byId("undo"),
    clear: byId("clear"),
    status: byId("status"),
    confidence: byId("confidence"),
    confidenceOutput: byId("confidence-output"),
    foregroundCount: byId("foreground-count"),
    bboxValue: byId("bbox-value"),
    lastSave: byId("last-save"),
    save: byId("save"),
    saveMessage: byId("save-message"),
  };

  const contexts = {
    base: elements.spectrogramCanvas.getContext("2d", { alpha: false }),
    proposal: elements.proposalCanvas.getContext("2d"),
    annotation: elements.annotationCanvas.getContext("2d"),
  };
  for (const context of Object.values(contexts)) {
    context.imageSmoothingEnabled = false;
  }

  let samples = [];
  let visibleSamples = [];
  let current = null;
  let mask = new Uint8Array(PIXELS);
  let proposalMask = null;
  let history = [];
  let activeTool = "brush";
  let pointerActive = false;
  let lastPoint = null;
  let rectangleStart = null;
  let rectangleBase = null;
  let dirty = false;
  let loadGeneration = 0;

  function annotatorId() {
    return elements.annotator.value.trim();
  }

  function validAnnotator(value) {
    return /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value);
  }

  async function requestJSON(url, options = {}) {
    const response = await fetch(url, {
      cache: "no-store",
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });
    let document = null;
    try {
      document = await response.json();
    } catch (_error) {
      document = null;
    }
    if (!response.ok) {
      throw new Error(document?.error || `Request failed (${response.status})`);
    }
    return document;
  }

  function setMessage(message, error = false) {
    elements.saveMessage.textContent = message;
    elements.saveMessage.classList.toggle("error", error);
  }

  function setDirty(value) {
    dirty = value;
    elements.sampleName.classList.toggle("unsaved", dirty);
    elements.save.disabled = !current || !validAnnotator(annotatorId());
  }

  function confirmDiscard() {
    return !dirty || window.confirm("Discard unsaved changes for this sample?");
  }

  function annotationSummary(sample) {
    if (!sample.annotation) return "not annotated";
    return `${sample.annotation.status} · ${Number(sample.annotation.confidence).toFixed(2)}`;
  }

  function applyFilter() {
    const filter = elements.statusFilter.value;
    visibleSamples = samples.filter((sample) => {
      if (filter === "all") return true;
      if (filter === "remaining") return !sample.annotation;
      if (filter === "annotated") return Boolean(sample.annotation);
      return sample.annotation?.status === filter;
    });
    renderSampleList();
    updateNavigation();
  }

  function renderSampleList() {
    elements.sampleList.replaceChildren();
    for (const sample of visibleSamples) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample-item";
      button.classList.toggle("active", current?.sample_id === sample.sample_id);
      button.dataset.sampleId = sample.sample_id;

      const dot = document.createElement("span");
      dot.className = `status-dot ${sample.annotation?.status || ""}`;
      dot.title = annotationSummary(sample);

      const primary = document.createElement("span");
      primary.className = "sample-primary";
      const title = document.createElement("strong");
      title.textContent = `${sample.modulation} · source ${sample.source_index}`;
      const secondary = document.createElement("small");
      secondary.textContent = sample.annotation
        ? annotationSummary(sample)
        : sample.sample_id.slice(-12);
      primary.append(title, secondary);

      const proposal = document.createElement("span");
      proposal.className = "proposal-mark";
      proposal.textContent = sample.proposal_available ? "P" : "";
      proposal.title = sample.proposal_available ? "Binary proposal available" : "";

      button.append(dot, primary, proposal);
      button.addEventListener("click", () => {
        if (current?.sample_id !== sample.sample_id && confirmDiscard()) {
          loadSample(sample.sample_id);
        }
      });
      item.append(button);
      elements.sampleList.append(item);
    }

    const annotated = samples.filter((sample) => sample.annotation).length;
    elements.progress.textContent = `${annotated}/${samples.length} annotated · ${visibleSamples.length} shown`;
  }

  function updateNavigation() {
    const index = visibleSamples.findIndex(
      (sample) => sample.sample_id === current?.sample_id,
    );
    elements.previous.disabled = index <= 0;
    elements.next.disabled = index < 0 || index >= visibleSamples.length - 1;
  }

  async function loadWork() {
    const annotator = annotatorId();
    if (!validAnnotator(annotator)) {
      setMessage(
        "Annotator ID must use 1–64 letters, digits, dots, underscores, or hyphens.",
        true,
      );
      return;
    }
    if (!confirmDiscard()) return;
    elements.loadAnnotator.disabled = true;
    setMessage("Loading pilot manifest…");
    try {
      const document = await requestJSON(
        `/api/samples?annotator=${encodeURIComponent(annotator)}`,
      );
      samples = document.samples;
      current = null;
      mask = new Uint8Array(PIXELS);
      proposalMask = null;
      history = [];
      setDirty(false);
      applyFilter();
      if (visibleSamples.length) {
        await loadSample(visibleSamples[0].sample_id, { skipDiscardCheck: true });
      } else {
        showEmpty("No samples match the current filter.");
      }
      setMessage("");
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      elements.loadAnnotator.disabled = false;
    }
  }

  function showEmpty(message) {
    elements.emptyState.textContent = message;
    elements.emptyState.hidden = false;
    elements.save.disabled = true;
  }

  function currentContrast() {
    return document.querySelector('input[name="contrast"]:checked').value;
  }

  async function loadBaseImage(sampleId, generation) {
    const image = new Image();
    const loaded = new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("Could not render the spectrogram image."));
    });
    image.src =
      `/api/samples/${encodeURIComponent(sampleId)}/image` +
      `?contrast=${encodeURIComponent(currentContrast())}&v=${generation}`;
    await loaded;
    if (generation !== loadGeneration) return;
    contexts.base.clearRect(0, 0, SIZE, SIZE);
    contexts.base.drawImage(image, 0, 0, SIZE, SIZE);
  }

  async function loadSample(sampleId, options = {}) {
    if (!options.skipDiscardCheck && !confirmDiscard()) return;
    const sample = samples.find((item) => item.sample_id === sampleId);
    if (!sample) return;
    const annotator = annotatorId();
    if (!validAnnotator(annotator)) return;

    const generation = ++loadGeneration;
    current = sample;
    setDirty(false);
    showEmpty("Loading original 64×64 spectrogram…");
    elements.sampleName.textContent = `${sample.modulation} · source ${sample.source_index}`;
    elements.sampleMeta.textContent =
      `${sample.split} · label ${sample.label} · ${sample.shape.join("×")} · ${sample.sample_id.slice(-12)}`;
    renderSampleList();
    updateNavigation();
    setMessage("");

    try {
      const detailPromise = requestJSON(
        `/api/samples/${encodeURIComponent(sampleId)}?annotator=${encodeURIComponent(annotator)}`,
      );
      const proposalPromise = sample.proposal_available
        ? requestJSON(`/api/samples/${encodeURIComponent(sampleId)}/proposal`)
        : Promise.resolve(null);
      const imagePromise = loadBaseImage(sampleId, generation);
      const [detail, proposalDocument] = await Promise.all([
        detailPromise,
        proposalPromise,
        imagePromise,
      ]).then(([detailValue, proposalValue]) => [detailValue, proposalValue]);
      if (generation !== loadGeneration) return;

      const annotation = detail.sample.annotation;
      mask = annotation
        ? Uint8Array.from(annotation.mask_64.flat())
        : new Uint8Array(PIXELS);
      proposalMask = proposalDocument
        ? Uint8Array.from(proposalDocument.proposal.mask_64.flat())
        : null;
      elements.proposalToggle.disabled = !proposalMask;
      elements.proposalToggle.checked = false;
      elements.proposalNote.textContent = proposalMask
        ? "Binary proposal available (off by default)"
        : "No proposal available";
      applyAnnotationForm(annotation);
      history = [];
      elements.undo.disabled = true;
      elements.emptyState.hidden = true;
      renderProposal();
      renderAnnotation();
      setDirty(false);
    } catch (error) {
      if (generation !== loadGeneration) return;
      showEmpty(error.message);
      setMessage(error.message, true);
    }
  }

  function applyAnnotationForm(annotation) {
    elements.status.value = annotation?.status || "draft";
    elements.confidence.value = annotation?.confidence ?? 0.5;
    elements.confidenceOutput.value = Number(elements.confidence.value).toFixed(2);
    const reasons = new Set(annotation?.ambiguity_reasons || []);
    for (const checkbox of document.querySelectorAll('input[name="ambiguity"]')) {
      checkbox.checked = reasons.has(checkbox.value);
    }
    elements.lastSave.textContent = annotation?.updated_at || "Not saved";
  }

  function renderProposal() {
    contexts.proposal.clearRect(0, 0, SIZE, SIZE);
    if (!proposalMask || !elements.proposalToggle.checked) return;
    const image = contexts.proposal.createImageData(SIZE, SIZE);
    for (let index = 0; index < PIXELS; index += 1) {
      if (!proposalMask[index]) continue;
      const offset = index * 4;
      image.data[offset] = 65;
      image.data[offset + 1] = 235;
      image.data[offset + 2] = 205;
      image.data[offset + 3] = 105;
    }
    contexts.proposal.putImageData(image, 0, 0);
  }

  function tightBBox() {
    let x0 = SIZE;
    let y0 = SIZE;
    let x1 = 0;
    let y1 = 0;
    let count = 0;
    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        if (!mask[y * SIZE + x]) continue;
        count += 1;
        x0 = Math.min(x0, x);
        y0 = Math.min(y0, y);
        x1 = Math.max(x1, x + 1);
        y1 = Math.max(y1, y + 1);
      }
    }
    return { count, bbox: count ? [x0, y0, x1, y1] : null };
  }

  function renderAnnotation() {
    contexts.annotation.clearRect(0, 0, SIZE, SIZE);
    const image = contexts.annotation.createImageData(SIZE, SIZE);
    for (let index = 0; index < PIXELS; index += 1) {
      if (!mask[index]) continue;
      const offset = index * 4;
      image.data[offset] = 255;
      image.data[offset + 1] = 112;
      image.data[offset + 2] = 87;
      image.data[offset + 3] = 150;
    }
    contexts.annotation.putImageData(image, 0, 0);
    const derived = tightBBox();
    elements.foregroundCount.textContent = `${derived.count} px`;
    elements.bboxValue.textContent = derived.bbox
      ? `[${derived.bbox.join(", ")}]`
      : "—";
  }

  function canvasPoint(event) {
    const bounds = elements.annotationCanvas.getBoundingClientRect();
    return {
      x: Math.max(
        0,
        Math.min(SIZE - 1, Math.floor(((event.clientX - bounds.left) / bounds.width) * SIZE)),
      ),
      y: Math.max(
        0,
        Math.min(SIZE - 1, Math.floor(((event.clientY - bounds.top) / bounds.height) * SIZE)),
      ),
    };
  }

  function pushHistory() {
    history.push(mask.slice());
    if (history.length > 50) history.shift();
    elements.undo.disabled = history.length === 0;
  }

  function paintPoint(point, value) {
    const diameter = Number(elements.brushSize.value);
    const radius = Math.max(0, (diameter - 1) / 2);
    const extent = Math.ceil(radius);
    for (let dy = -extent; dy <= extent; dy += 1) {
      for (let dx = -extent; dx <= extent; dx += 1) {
        if (dx * dx + dy * dy > (radius + 0.35) ** 2) continue;
        const x = point.x + dx;
        const y = point.y + dy;
        if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
          mask[y * SIZE + x] = value;
        }
      }
    }
  }

  function paintLine(from, to, value) {
    const steps = Math.max(Math.abs(to.x - from.x), Math.abs(to.y - from.y), 1);
    for (let step = 0; step <= steps; step += 1) {
      paintPoint(
        {
          x: Math.round(from.x + ((to.x - from.x) * step) / steps),
          y: Math.round(from.y + ((to.y - from.y) * step) / steps),
        },
        value,
      );
    }
  }

  function fillRectangle(from, to) {
    const x0 = Math.min(from.x, to.x);
    const y0 = Math.min(from.y, to.y);
    const x1 = Math.max(from.x, to.x);
    const y1 = Math.max(from.y, to.y);
    for (let y = y0; y <= y1; y += 1) {
      mask.fill(1, y * SIZE + x0, y * SIZE + x1 + 1);
    }
  }

  function beginDrawing(event) {
    if (!current || event.button !== 0) return;
    event.preventDefault();
    elements.annotationCanvas.setPointerCapture(event.pointerId);
    pointerActive = true;
    const point = canvasPoint(event);
    pushHistory();
    if (activeTool === "rectangle") {
      rectangleStart = point;
      rectangleBase = mask.slice();
      fillRectangle(point, point);
    } else {
      paintPoint(point, activeTool === "eraser" ? 0 : 1);
      lastPoint = point;
    }
    renderAnnotation();
    setDirty(true);
  }

  function continueDrawing(event) {
    if (!pointerActive) return;
    event.preventDefault();
    const point = canvasPoint(event);
    if (activeTool === "rectangle") {
      mask = rectangleBase.slice();
      fillRectangle(rectangleStart, point);
    } else {
      paintLine(lastPoint, point, activeTool === "eraser" ? 0 : 1);
      lastPoint = point;
    }
    renderAnnotation();
  }

  function endDrawing(event) {
    if (!pointerActive) return;
    event.preventDefault();
    pointerActive = false;
    lastPoint = null;
    rectangleStart = null;
    rectangleBase = null;
    if (elements.annotationCanvas.hasPointerCapture(event.pointerId)) {
      elements.annotationCanvas.releasePointerCapture(event.pointerId);
    }
  }

  function chooseTool(tool) {
    activeTool = tool;
    for (const button of document.querySelectorAll(".tool")) {
      button.classList.toggle("active", button.dataset.tool === tool);
    }
  }

  function maskRows() {
    const rows = [];
    for (let y = 0; y < SIZE; y += 1) {
      rows.push(Array.from(mask.slice(y * SIZE, (y + 1) * SIZE)));
    }
    return rows;
  }

  function checkedReasons() {
    return Array.from(
      document.querySelectorAll('input[name="ambiguity"]:checked'),
      (checkbox) => checkbox.value,
    );
  }

  async function saveAnnotation() {
    if (!current) return;
    const annotator = annotatorId();
    if (!validAnnotator(annotator)) {
      setMessage("Set a valid annotator ID before saving.", true);
      return;
    }
    const derived = tightBBox();
    if (elements.status.value === "final" && current.label === 1 && derived.count === 0) {
      setMessage(
        "A final anomaly mask cannot be empty. Mark a region or use unlocalizable.",
        true,
      );
      return;
    }
    if (elements.status.value === "final" && current.label === 0 && derived.count !== 0) {
      setMessage("A final normal sample must have an empty mask.", true);
      return;
    }
    if (elements.status.value === "unlocalizable" && derived.count !== 0) {
      setMessage("Unlocalizable requires an empty mask. Clear the mask first.", true);
      return;
    }

    elements.save.disabled = true;
    setMessage("Saving atomically…");
    const payload = {
      annotator,
      mask_64: maskRows(),
      status: elements.status.value,
      confidence: Number(elements.confidence.value),
      ambiguity_reasons: checkedReasons(),
      proposal_visible_at_save:
        Boolean(proposalMask) && elements.proposalToggle.checked,
    };
    try {
      const document = await requestJSON(
        `/api/samples/${encodeURIComponent(current.sample_id)}/annotation`,
        { method: "PUT", body: JSON.stringify(payload) },
      );
      current.annotation = {
        status: document.annotation.status,
        confidence: document.annotation.confidence,
        updated_at: document.annotation.updated_at,
      };
      elements.lastSave.textContent = document.annotation.updated_at;
      setDirty(false);
      renderSampleList();
      setMessage("Saved. Tight bbox and 16×16 max-pooled mask were derived by the server.");
    } catch (error) {
      setMessage(error.message, true);
      elements.save.disabled = false;
    }
  }

  elements.loadAnnotator.addEventListener("click", loadWork);
  elements.statusFilter.addEventListener("change", applyFilter);
  elements.previous.addEventListener("click", () => {
    const index = visibleSamples.findIndex(
      (sample) => sample.sample_id === current?.sample_id,
    );
    if (index > 0 && confirmDiscard()) loadSample(visibleSamples[index - 1].sample_id);
  });
  elements.next.addEventListener("click", () => {
    const index = visibleSamples.findIndex(
      (sample) => sample.sample_id === current?.sample_id,
    );
    if (index >= 0 && index < visibleSamples.length - 1 && confirmDiscard()) {
      loadSample(visibleSamples[index + 1].sample_id);
    }
  });
  for (const radio of document.querySelectorAll('input[name="contrast"]')) {
    radio.addEventListener("change", () => {
      if (current) {
        const generation = ++loadGeneration;
        loadBaseImage(current.sample_id, generation).catch((error) =>
          setMessage(error.message, true),
        );
      }
    });
  }
  elements.proposalToggle.addEventListener("change", renderProposal);
  elements.brushSize.addEventListener("input", () => {
    elements.brushSizeOutput.value = `${elements.brushSize.value} px`;
  });
  for (const button of document.querySelectorAll(".tool")) {
    button.addEventListener("click", () => chooseTool(button.dataset.tool));
  }
  elements.annotationCanvas.addEventListener("pointerdown", beginDrawing);
  elements.annotationCanvas.addEventListener("pointermove", continueDrawing);
  elements.annotationCanvas.addEventListener("pointerup", endDrawing);
  elements.annotationCanvas.addEventListener("pointercancel", endDrawing);
  elements.undo.addEventListener("click", () => {
    const previous = history.pop();
    if (!previous) return;
    mask = previous;
    renderAnnotation();
    elements.undo.disabled = history.length === 0;
    setDirty(true);
  });
  elements.clear.addEventListener("click", () => {
    if (!current || !mask.some((value) => value)) return;
    pushHistory();
    mask.fill(0);
    renderAnnotation();
    setDirty(true);
  });
  elements.status.addEventListener("change", () => setDirty(true));
  elements.confidence.addEventListener("input", () => {
    elements.confidenceOutput.value = Number(elements.confidence.value).toFixed(2);
    setDirty(true);
  });
  for (const checkbox of document.querySelectorAll('input[name="ambiguity"]')) {
    checkbox.addEventListener("change", () => setDirty(true));
  }
  elements.save.addEventListener("click", saveAnnotation);
  window.addEventListener("beforeunload", (event) => {
    if (!dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      elements.undo.click();
      return;
    }
    if (event.key.toLowerCase() === "b") chooseTool("brush");
    if (event.key.toLowerCase() === "e") chooseTool("eraser");
    if (event.key.toLowerCase() === "r") chooseTool("rectangle");
    if (event.key === "ArrowLeft") elements.previous.click();
    if (event.key === "ArrowRight") elements.next.click();
  });

  elements.annotator.value = window.IAD_UI_CONFIG.annotator;
  showEmpty("Loading the server-locked reviewer workspace…");
  loadWork();
})();
