import * as ort from "onnxruntime-web/webgpu";
import ortBootstrapUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import "./style.css";

const BASE = import.meta.env.BASE_URL;
const MODELS = {
  yolov8n: { name: "YOLOv8n", url: `${BASE}models/cursor-yolov8n.onnx` },
  yolo26n: { name: "YOLO26n · macOS augmented", url: `${BASE}models/cursor-yolo26n.onnx` },
  temporal: { name: "MobileNetV4 · temporal heatmap", url: `${BASE}models/mobilenetv4-temporal.onnx`, temporal: true }
};
const SIZE = 640;
const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");
const video = document.querySelector("#video");
const modelPicker = document.querySelector("#modelPicker");
const videoControls = document.querySelector("#videoControls");
const playPause = document.querySelector("#playPause");
const stepBack = document.querySelector("#stepBack");
const stepForward = document.querySelector("#stepForward");
const timeline = document.querySelector("#timeline");
const currentTimeLabel = document.querySelector("#currentTime");
const durationLabel = document.querySelector("#duration");
const playbackRate = document.querySelector("#playbackRate");
const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const emptyState = document.querySelector("#emptyState");
const resultLabel = document.querySelector("#result");
const fpsLabel = document.querySelector("#fps");
const timingLabel = document.querySelector("#timing");
const backendLabel = document.querySelector("#backend");
const statusLabel = document.querySelector("#modelStatus");
const confidence = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidenceValue");
const traceToggle = document.querySelector("#traceToggle");
const inpaintToggle = document.querySelector("#inpaintToggle");
const demoVideoButton = document.querySelector("#demoVideoButton");
const shareScreenButton = document.querySelector("#shareScreenButton");

let session;
let activeModelId;
const sessionCache = new Map();
let modelLoadId = 0;
let modelLoadQueue = Promise.resolve();
let currentImage;
let videoFrameRequest;
let busy = false;
let lastBoxes = [];
let objectUrl;
let smoothedInferenceMs;
let pendingDetectionSource;
let skimTimer;
let temporalHistory = [];
let tracePoints = [];
let displayStream;
let isLiveScreen = false;
let backgroundCanvas;
let validBackgroundCanvas;
let repairCanvas;
let inpaintMaskCanvas;
let lastInpaintTime;
let fakeCursorFilters = [];
let inferenceFrameCanvas;
let previousFrameCanvas;
let previousCursorRegions = [];
let comparisonCanvas;
const FRAME_SECONDS = 1 / 30;
const TRACE_GAP_SECONDS = 0.3;

// ORT's WebGPU backend still uses a WASM bootstrap. Vite emits both assets
// locally; the explicit overrides avoid any runtime CDN dependency.
ort.env.wasm.wasmPaths = { mjs: ortBootstrapUrl, wasm: ortWasmUrl };
ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 1, 4) : 1;

function loadModel(modelId = modelPicker.value) {
  const loadId = ++modelLoadId;
  const model = MODELS[modelId];
  statusLabel.className = "status";
  statusLabel.innerHTML = `<span></span> Loading ${model.name}…`;
  modelLoadQueue = modelLoadQueue.catch(() => {}).then(() => loadModelNow(modelId, loadId));
  return modelLoadQueue;
}

async function loadModelNow(modelId, loadId) {
  const model = MODELS[modelId];
  const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  try {
    const loaded = sessionCache.get(modelId) || await ort.InferenceSession.create(model.url, { executionProviders: providers });
    if (loadId !== modelLoadId) return;
    sessionCache.set(modelId, loaded);
    session = loaded;
    activeModelId = modelId;
    temporalHistory = [];
    statusLabel.classList.add("ready");
    statusLabel.innerHTML = `<span></span> ${model.name} ready`;
    backendLabel.textContent = navigator.gpu ? "WebGPU" : "WASM fallback";
    smoothedInferenceMs = undefined;
    fpsLabel.textContent = "";
    const source = currentImage || (hasVideoSource() ? video : null);
    if (source) detect(source);
  } catch (error) {
    statusLabel.classList.add("error");
    statusLabel.textContent = "Model failed to load";
    resultLabel.textContent = error.message;
    throw error;
  }
}

function preprocess(source) {
  const width = source.videoWidth || source.naturalWidth || source.width;
  const height = source.videoHeight || source.naturalHeight || source.height;
  const scale = Math.min(SIZE / width, SIZE / height);
  const drawWidth = Math.round(width * scale);
  const drawHeight = Math.round(height * scale);
  const padX = Math.floor((SIZE - drawWidth) / 2);
  const padY = Math.floor((SIZE - drawHeight) / 2);

  const work = new OffscreenCanvas(SIZE, SIZE);
  const workCtx = work.getContext("2d", { willReadFrequently: true });
  workCtx.fillStyle = "rgb(114,114,114)";
  workCtx.fillRect(0, 0, SIZE, SIZE);
  workCtx.drawImage(source, padX, padY, drawWidth, drawHeight);
  const pixels = workCtx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  const data = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i++) {
    data[i] = pixels[i * 4] / 255;
    data[plane + i] = pixels[i * 4 + 1] / 255;
    data[plane * 2 + i] = pixels[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]), scale, padX, padY, width, height };
}

function preprocessTemporal(source) {
  const width = source.videoWidth || source.naturalWidth || source.width;
  const height = source.videoHeight || source.naturalHeight || source.height;
  const scale = Math.min(SIZE / width, SIZE / height);
  const drawWidth = Math.round(width * scale);
  const drawHeight = Math.round(height * scale);
  const padX = Math.floor((SIZE - drawWidth) / 2);
  const padY = Math.floor((SIZE - drawHeight) / 2);
  const work = new OffscreenCanvas(SIZE, SIZE);
  const workCtx = work.getContext("2d", { willReadFrequently: true });
  workCtx.fillStyle = "rgb(114,114,114)";
  workCtx.fillRect(0, 0, SIZE, SIZE);
  workCtx.drawImage(source, padX, padY, drawWidth, drawHeight);
  const pixels = workCtx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  const current = new Float32Array(plane);
  for (let i = 0; i < plane; i++) {
    current[i] = (pixels[i * 4] * 0.299 + pixels[i * 4 + 1] * 0.587 + pixels[i * 4 + 2] * 0.114) / 255;
  }
  const previous = temporalHistory.at(-1) || current;
  const oldest = temporalHistory.at(-2) || previous;
  const data = new Float32Array(plane * 3);
  data.set(current);
  const threshold = 10 / 255;
  const widthRamp = 30 / 255;
  for (let i = 0; i < plane; i++) {
    data[plane + i] = Math.min(1, Math.max(0, (Math.abs(current[i] - previous[i]) - threshold) / widthRamp));
    data[plane * 2 + i] = Math.min(1, Math.max(0, (Math.abs(current[i] - oldest[i]) - threshold) / widthRamp));
  }
  temporalHistory.push(current);
  if (temporalHistory.length > 2) temporalHistory.shift();
  return { tensor: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]), scale, padX, padY, width, height };
}

function decodeTemporal(outputs, meta, threshold) {
  const heatmap = outputs.heatmap || outputs[session.outputNames[0]];
  const offset = outputs.offset || outputs[session.outputNames[1]];
  let bestIndex = 0;
  for (let i = 1; i < heatmap.data.length; i++) {
    if (heatmap.data[i] > heatmap.data[bestIndex]) bestIndex = i;
  }
  const score = heatmap.data[bestIndex];
  if (score < threshold) return [];
  const heatmapWidth = heatmap.dims[3];
  const xCell = bestIndex % heatmapWidth;
  const yCell = Math.floor(bestIndex / heatmapWidth);
  const plane = heatmap.dims[2] * heatmapWidth;
  const inputX = (xCell + offset.data[bestIndex]) * 4;
  const inputY = (yCell + offset.data[plane + bestIndex]) * 4;
  const x = (inputX - meta.padX) / meta.scale;
  const y = (inputY - meta.padY) / meta.scale;
  if (x < 0 || y < 0 || x >= meta.width || y >= meta.height) return [];
  return [{ x, y, w: 0, h: 0, score, point: true }];
}

function decode(output, meta, threshold) {
  const data = output.data;
  // YOLO26 exports end-to-end detections as [1, 300, 6], where each row is
  // x1, y1, x2, y2, score, class. NMS is already part of the model behavior.
  if (output.dims[1] === 300 && output.dims[2] === 6) {
    const boxes = [];
    for (let i = 0; i < 300; i++) {
      const offset = i * 6;
      const score = data[offset + 4];
      if (score < threshold) continue;
      const x1 = (data[offset] - meta.padX) / meta.scale;
      const y1 = (data[offset + 1] - meta.padY) / meta.scale;
      const x2 = (data[offset + 2] - meta.padX) / meta.scale;
      const y2 = (data[offset + 3] - meta.padY) / meta.scale;
      boxes.push({
        x: Math.max(0, x1), y: Math.max(0, y1),
        w: Math.min(meta.width, x2) - Math.max(0, x1),
        h: Math.min(meta.height, y2) - Math.max(0, y1), score
      });
    }
    return boxes.slice(0, 10);
  }

  // YOLOv8 exports raw predictions as [1, 5, 8400].
  const count = output.dims[2]; // [1, 5, 8400]: cx, cy, w, h, one class score
  const boxes = [];
  for (let i = 0; i < count; i++) {
    const score = data[count * 4 + i];
    if (score < threshold) continue;
    const cx = data[i];
    const cy = data[count + i];
    const w = data[count * 2 + i];
    const h = data[count * 3 + i];
    const x = (cx - w / 2 - meta.padX) / meta.scale;
    const y = (cy - h / 2 - meta.padY) / meta.scale;
    boxes.push({
      x: Math.max(0, x), y: Math.max(0, y),
      w: Math.min(meta.width - x, w / meta.scale),
      h: Math.min(meta.height - y, h / meta.scale), score
    });
  }
  return nms(boxes, 0.45).slice(0, 10);
}

function nms(boxes, iouThreshold) {
  const sorted = boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  while (sorted.length) {
    const best = sorted.shift();
    kept.push(best);
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(best, sorted[i]) > iouThreshold) sorted.splice(i, 1);
    }
  }
  return kept;
}

function iou(a, b) {
  const intersection = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) *
    Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return intersection / (a.w * a.h + b.w * b.h - intersection);
}

function resetInpainting() {
  backgroundCanvas = undefined;
  validBackgroundCanvas = undefined;
  repairCanvas = undefined;
  inpaintMaskCanvas = undefined;
  lastInpaintTime = undefined;
  fakeCursorFilters = [];
  previousFrameCanvas = undefined;
  previousCursorRegions = [];
  comparisonCanvas = undefined;
}

function snapshotVideoFrame(source) {
  const width = source.videoWidth;
  const height = source.videoHeight;
  if (!inferenceFrameCanvas || inferenceFrameCanvas.width !== width || inferenceFrameCanvas.height !== height) {
    inferenceFrameCanvas = new OffscreenCanvas(width, height);
  }
  inferenceFrameCanvas.getContext("2d").drawImage(source, 0, 0, width, height);
  return inferenceFrameCanvas;
}

function smoothingAlpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

function lowPass(value, previous, alpha) {
  return alpha * value + (1 - alpha) * previous;
}

function smoothFakeCursor(point, index, time, update) {
  const previous = fakeCursorFilters[index];
  if (!update && previous) return { x: previous.x, y: previous.y };
  const dt = previous ? time - previous.time : 0;
  if (!previous || dt <= 0 || dt > 0.25) {
    fakeCursorFilters[index] = { ...point, rawX: point.x, rawY: point.y, dx: 0, dy: 0, time };
    return point;
  }

  // One Euro filter: stable while nearly stationary, but raises its cutoff as
  // the cursor accelerates so the marker stays close to the actual hotspot.
  const derivativeAlpha = smoothingAlpha(1, dt);
  const dx = lowPass((point.x - previous.rawX) / dt, previous.dx, derivativeAlpha);
  const dy = lowPass((point.y - previous.rawY) / dt, previous.dy, derivativeAlpha);
  const speed = Math.hypot(dx, dy);
  const positionAlpha = smoothingAlpha(2.5 + 0.015 * speed, dt);
  const filtered = {
    x: lowPass(point.x, previous.x, positionAlpha),
    y: lowPass(point.y, previous.y, positionAlpha),
    rawX: point.x, rawY: point.y, dx, dy, time
  };
  fakeCursorFilters[index] = filtered;
  return { x: filtered.x, y: filtered.y };
}

function ensureInpaintCanvases(width, height) {
  if (backgroundCanvas?.width === width && backgroundCanvas.height === height) return;
  backgroundCanvas = new OffscreenCanvas(width, height);
  validBackgroundCanvas = new OffscreenCanvas(width, height);
  repairCanvas = new OffscreenCanvas(width, height);
  inpaintMaskCanvas = new OffscreenCanvas(width, height);
}

function cursorRegions(boxes, width, height) {
  return boxes.map(box => {
    if (!box.point) {
      const padding = Math.max(5, Math.min(width, height) * 0.008);
      return {
        x: box.x + box.w / 2,
        y: box.y + box.h / 2,
        radius: Math.max(box.w, box.h) / 2 + padding
      };
    }
    // The hotspot of an arrow is near its top-left, while most of its pixels
    // extend down and right. A slight offset also remains safe for symmetric
    // cursors such as the I-beam and pointing hand.
    const radius = Math.max(22, Math.min(64, Math.min(width, height) * 0.032));
    return { x: box.x + radius * 0.22, y: box.y + radius * 0.22, radius };
  });
}

function paintCursorMask(maskCtx, regions, feather = true) {
  maskCtx.clearRect(0, 0, maskCtx.canvas.width, maskCtx.canvas.height);
  for (const region of regions) {
    if (feather) {
      const inner = Math.max(0, region.radius - Math.max(3, region.radius * 0.18));
      const gradient = maskCtx.createRadialGradient(region.x, region.y, inner, region.x, region.y, region.radius);
      gradient.addColorStop(0, "rgba(255,255,255,1)");
      gradient.addColorStop(1, "rgba(255,255,255,0)");
      maskCtx.fillStyle = gradient;
    } else {
      maskCtx.fillStyle = "white";
    }
    maskCtx.beginPath();
    maskCtx.arc(region.x, region.y, region.radius, 0, Math.PI * 2);
    maskCtx.fill();
  }
}

function isSceneStableNearCursor(source, regions, width, height) {
  if (!previousFrameCanvas || !regions.length || !source.getContext) return true;
  const region = regions[0];
  const radius = region.radius * 2;
  const x = Math.max(0, Math.floor(region.x - radius));
  const y = Math.max(0, Math.floor(region.y - radius));
  const w = Math.min(width - x, Math.ceil(radius * 2));
  const h = Math.min(height - y, Math.ceil(radius * 2));
  if (w <= 0 || h <= 0) return true;
  const current = source.getContext("2d").getImageData(x, y, w, h).data;
  const previous = previousFrameCanvas.getContext("2d").getImageData(x, y, w, h).data;
  let difference = 0;
  let samples = 0;
  for (let py = 0; py < h; py += 3) {
    for (let px = 0; px < w; px += 3) {
      const screenX = x + px;
      const screenY = y + py;
      const distance = Math.hypot(screenX - region.x, screenY - region.y);
      if (distance < region.radius * 1.15 || distance > region.radius * 1.9) continue;
      if (previousCursorRegions.some(old => Math.hypot(screenX - old.x, screenY - old.y) < old.radius)) continue;
      const offset = (py * w + px) * 4;
      difference += Math.abs(current[offset] - previous[offset]);
      difference += Math.abs(current[offset + 1] - previous[offset + 1]);
      difference += Math.abs(current[offset + 2] - previous[offset + 2]);
      samples += 3;
    }
  }
  return !samples || difference / samples < 10;
}

function rememberInpaintFrame(source, regions, width, height) {
  if (!previousFrameCanvas || previousFrameCanvas.width !== width || previousFrameCanvas.height !== height) {
    previousFrameCanvas = new OffscreenCanvas(width, height);
  }
  previousFrameCanvas.getContext("2d").drawImage(source, 0, 0, width, height);
  previousCursorRegions = regions.map(region => ({ ...region }));
}

function drawInpainted(source, boxes, width, height, updateBackground) {
  ensureInpaintCanvases(width, height);
  const regions = cursorRegions(boxes, width, height);
  const backgroundCtx = backgroundCanvas.getContext("2d");
  const validCtx = validBackgroundCanvas.getContext("2d");
  const maskCtx = inpaintMaskCanvas.getContext("2d");
  const sceneStable = isSceneStableNearCursor(source, regions, width, height);

  if (updateBackground) {
    // Update every currently visible pixel except the cursor region. The
    // validity canvas prevents an initially obscured pixel from being used as
    // background until it has actually been seen.
    const cleanArea = new Path2D();
    cleanArea.rect(0, 0, width, height);
    for (const region of regions) cleanArea.arc(region.x, region.y, region.radius, 0, Math.PI * 2);
    backgroundCtx.save();
    backgroundCtx.clip(cleanArea, "evenodd");
    backgroundCtx.drawImage(source, 0, 0, width, height);
    backgroundCtx.restore();
    validCtx.save();
    validCtx.clip(cleanArea, "evenodd");
    validCtx.fillStyle = "white";
    validCtx.fillRect(0, 0, width, height);
    validCtx.restore();
  }

  ctx.drawImage(source, 0, 0, width, height);
  if (!regions.length) {
    if (updateBackground) rememberInpaintFrame(source, regions, width, height);
    return;
  }
  paintCursorMask(maskCtx, regions);
  const repairCtx = repairCanvas.getContext("2d");
  repairCtx.clearRect(0, 0, width, height);
  repairCtx.globalCompositeOperation = "source-over";
  if (sceneStable) {
    repairCtx.drawImage(backgroundCanvas, 0, 0);
    repairCtx.globalCompositeOperation = "destination-in";
    repairCtx.drawImage(inpaintMaskCanvas, 0, 0);
    repairCtx.drawImage(validBackgroundCanvas, 0, 0);
  }
  if (previousFrameCanvas) {
    // Prefer the immediately preceding frame wherever it did not contain the
    // cursor. This keeps replacements fresh during fast motion and scrolling.
    comparisonCanvas ||= new OffscreenCanvas(width, height);
    if (comparisonCanvas.width !== width || comparisonCanvas.height !== height) {
      comparisonCanvas = new OffscreenCanvas(width, height);
    }
    const comparisonCtx = comparisonCanvas.getContext("2d");
    comparisonCtx.clearRect(0, 0, width, height);
    comparisonCtx.globalCompositeOperation = "source-over";
    comparisonCtx.drawImage(previousFrameCanvas, 0, 0);
    paintCursorMask(maskCtx, previousCursorRegions, false);
    comparisonCtx.globalCompositeOperation = "destination-out";
    comparisonCtx.drawImage(inpaintMaskCanvas, 0, 0);
    paintCursorMask(maskCtx, regions);
    comparisonCtx.globalCompositeOperation = "destination-in";
    comparisonCtx.drawImage(inpaintMaskCanvas, 0, 0);
    comparisonCtx.globalCompositeOperation = "source-over";
    repairCtx.globalCompositeOperation = "source-over";
    repairCtx.drawImage(comparisonCanvas, 0, 0);
  }
  repairCtx.globalCompositeOperation = "source-over";
  ctx.drawImage(repairCanvas, 0, 0);
  if (updateBackground) rememberInpaintFrame(source, regions, width, height);
}

function draw(source, boxes, updateBackground = false, videoFrame = source === video,
    frameTime = videoFrame ? video.currentTime : undefined) {
  const width = source.videoWidth || source.naturalWidth || source.width;
  const height = source.videoHeight || source.naturalHeight || source.height;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  if (inpaintToggle.checked) drawInpainted(source, boxes, width, height, updateBackground);
  else ctx.drawImage(source, 0, 0, width, height);
  // Once the original cursor is gone, retain a small visual indication of the
  // detected hotspot without bringing back the full detector annotation UI.
  if (inpaintToggle.checked) {
    if (videoFrame && traceToggle.checked) drawTrace(frameTime, width);
    ctx.save();
    ctx.strokeStyle = "#0a84ff";
    ctx.lineWidth = 3;
    const overlayTime = videoFrame && Number.isFinite(frameTime)
      ? frameTime
      : performance.now() / 1000;
    for (const [index, box] of boxes.entries()) {
      const point = smoothFakeCursor(tracePointFor(box), index, overlayTime, updateBackground);
      ctx.beginPath();
      ctx.arc(point.x, point.y, 24, 0, Math.PI * 2);
      ctx.stroke();
    }
    fakeCursorFilters.length = boxes.length;
    ctx.restore();
    return;
  }
  if (videoFrame && traceToggle.checked) drawTrace(video.currentTime, width);
  const line = Math.max(2, width / 600);
  ctx.font = `${Math.max(14, width / 65)}px ui-monospace, monospace`;
  ctx.lineWidth = line;
  for (const box of boxes) {
    ctx.strokeStyle = "#d7ff45";
    if (box.point) {
      const radius = Math.max(7, width / 100);
      ctx.beginPath(); ctx.arc(box.x, box.y, radius, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(box.x - radius * 1.4, box.y); ctx.lineTo(box.x + radius * 1.4, box.y);
      ctx.moveTo(box.x, box.y - radius * 1.4); ctx.lineTo(box.x, box.y + radius * 1.4); ctx.stroke();
    } else {
      ctx.strokeRect(box.x, box.y, box.w, box.h);
    }
    const text = `cursor ${Math.round(box.score * 100)}%`;
    const metrics = ctx.measureText(text);
    const labelHeight = parseInt(ctx.font, 10) + 8;
    ctx.fillStyle = "#d7ff45";
    ctx.fillRect(box.x, Math.max(0, box.y - labelHeight), metrics.width + 10, labelHeight);
    ctx.fillStyle = "#11130e";
    ctx.fillText(text, box.x + 5, Math.max(labelHeight - 5, box.y - 6));
  }
}

function tracePointFor(box) {
  return box.point
    ? { x: box.x, y: box.y }
    : { x: box.x + box.w / 2, y: box.y + box.h / 2 };
}

function recordTrace(boxes, time) {
  if (!Number.isFinite(time) || !boxes.length) return;
  const best = boxes.reduce((a, b) => a.score >= b.score ? a : b);
  const point = { ...tracePointFor(best), time };
  const nearby = tracePoints.findIndex(item => Math.abs(item.time - time) < FRAME_SECONDS / 2);
  if (nearby >= 0) tracePoints[nearby] = point;
  else tracePoints.push(point);
  tracePoints.sort((a, b) => a.time - b.time);
}

function strokeSmooth(points) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length - 1; i++) {
    const next = points[i + 1];
    ctx.quadraticCurveTo(points[i].x, points[i].y, (points[i].x + next.x) / 2, (points[i].y + next.y) / 2);
  }
  if (points.length > 1) ctx.lineTo(points.at(-1).x, points.at(-1).y);
  ctx.stroke();
}

function drawTrace(currentTime, width) {
  const visible = tracePoints.filter(point => point.time <= currentTime + FRAME_SECONDS / 2);
  if (!visible.length) return;
  const segments = [];
  let segment = [];
  for (const point of visible) {
    if (segment.length && point.time - segment.at(-1).time > TRACE_GAP_SECONDS) {
      segments.push(segment);
      segment = [];
    }
    segment.push(point);
  }
  if (segment.length) segments.push(segment);

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const points of segments) {
    ctx.strokeStyle = "rgba(255,255,255,.78)";
    ctx.lineWidth = Math.max(5, width / 150);
    strokeSmooth(points);
    ctx.strokeStyle = "rgba(10,132,255,.9)";
    ctx.lineWidth = Math.max(2.5, width / 300);
    strokeSmooth(points);
  }
  ctx.restore();
}

async function detect(source) {
  if (!session) return;
  if (busy) {
    pendingDetectionSource = source;
    return;
  }
  busy = true;
  const started = performance.now();
  const sourceTime = source === video ? video.currentTime : undefined;
  // A video element continues advancing while inference is in flight. Keep an
  // immutable copy so the returned hotspot, background update, and rendered
  // pixels all refer to exactly the same frame.
  const inferenceSource = source === video ? snapshotVideoFrame(source) : source;
  try {
    const temporal = MODELS[activeModelId]?.temporal;
    const meta = temporal ? preprocessTemporal(inferenceSource) : preprocess(inferenceSource);
    const outputs = await session.run({ [session.inputNames[0]]: meta.tensor });
    lastBoxes = temporal
      ? decodeTemporal(outputs, meta, Number(confidence.value) / 100)
      : decode(outputs[session.outputNames[0]], meta, Number(confidence.value) / 100);
    recordTrace(lastBoxes, sourceTime);
    if (source === video && Number.isFinite(sourceTime) && Number.isFinite(lastInpaintTime) &&
        sourceTime < lastInpaintTime) {
      resetInpainting();
    }
    draw(inferenceSource, lastBoxes, true, source === video, sourceTime);
    if (source === video) lastInpaintTime = sourceTime;
    const elapsed = performance.now() - started;
    smoothedInferenceMs = smoothedInferenceMs == null ? elapsed : smoothedInferenceMs * 0.8 + elapsed * 0.2;
    resultLabel.textContent = lastBoxes.length ? `${lastBoxes.length} cursor${lastBoxes.length > 1 ? "s" : ""} detected` : "No cursor detected";
    timingLabel.textContent = `${Math.round(elapsed)} ms`;
    fpsLabel.textContent = `${(1000 / smoothedInferenceMs).toFixed(1)} inference FPS`;
  } finally {
    busy = false;
    if (pendingDetectionSource) {
      const latestSource = pendingDetectionSource;
      pendingDetectionSource = undefined;
      detect(latestSource);
    }
  }
}

function stopVideo() {
  video.pause();
  if (videoFrameRequest) {
    video.cancelVideoFrameCallback(videoFrameRequest);
    videoFrameRequest = undefined;
  }
}

function hasVideoSource() {
  return Boolean(video.src || video.srcObject);
}

function stopScreenShare({ updateResult = true } = {}) {
  if (!displayStream) return;
  const stream = displayStream;
  displayStream = undefined;
  isLiveScreen = false;
  stream.getTracks().forEach(track => track.stop());
  stopVideo();
  video.srcObject = null;
  shareScreenButton.textContent = "Share screen";
  shareScreenButton.classList.remove("sharing");
  shareScreenButton.setAttribute("aria-pressed", "false");
  if (updateResult) resultLabel.textContent = "Screen sharing stopped";
}

function videoLoop() {
  detect(video);
  updateTransport();
  if (!video.paused && !video.ended) videoFrameRequest = video.requestVideoFrameCallback(videoLoop);
  else videoFrameRequest = undefined;
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00.000";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function updateTransport() {
  if (isLiveScreen) return;
  const duration = video.duration || 0;
  timeline.value = duration ? Math.round(video.currentTime / duration * 1000) : 0;
  currentTimeLabel.textContent = formatTime(video.currentTime);
  durationLabel.textContent = formatTime(duration);
  playPause.textContent = video.paused ? "▶" : "❚❚";
  playPause.setAttribute("aria-label", video.paused ? "Play" : "Pause");
}

async function seekTo(time) {
  const target = Math.max(0, Math.min(video.duration || 0, time));
  if (Math.abs(video.currentTime - target) < 0.0001) {
    updateTransport();
    await detect(video);
    return;
  }
  video.currentTime = target;
  await new Promise(resolve => video.addEventListener("seeked", resolve, { once: true }));
  updateTransport();
  await detect(video);
}

function openFile(file) {
  if (!file) return;
  stopScreenShare({ updateResult: false });
  stopVideo();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  smoothedInferenceMs = undefined;
  temporalHistory = [];
  tracePoints = [];
  resetInpainting();
  fpsLabel.textContent = "";
  emptyState.hidden = true;
  if (file.type.startsWith("video/")) {
    currentImage = null;
    videoControls.hidden = false;
    video.muted = true;
    video.autoplay = true;
    video.src = objectUrl;
    video.onloadeddata = async () => {
      // Show the first decoded frame immediately, without waiting for inference.
      lastBoxes = [];
      draw(video, lastBoxes);
      updateTransport();
      try {
        await video.play();
        if (!videoFrameRequest) videoLoop();
      } catch {
        // Muted autoplay is normally allowed, but leave the video ready for the
        // manual play button if a browser policy still blocks it.
        await detect(video);
        updateTransport();
      }
    };
  } else {
    videoControls.hidden = true;
    const image = new Image();
    image.onload = () => { currentImage = image; detect(image); };
    image.src = objectUrl;
  }
}

fileInput.addEventListener("change", () => openFile(fileInput.files[0]));
confidence.addEventListener("input", () => {
  confidenceValue.textContent = `${confidence.value}%`;
  if (currentImage) detect(currentImage);
});
confidence.addEventListener("change", () => {
  tracePoints = [];
  if (hasVideoSource() && !currentImage) detect(video);
});
traceToggle.addEventListener("change", () => {
  if (hasVideoSource() && !currentImage) draw(video, lastBoxes);
});
inpaintToggle.addEventListener("change", () => {
  resetInpainting();
  if (currentImage) detect(currentImage);
  else if (hasVideoSource()) detect(video);
});
modelPicker.addEventListener("change", () => {
  tracePoints = [];
  resetInpainting();
  loadModel(modelPicker.value);
});
["dragenter", "dragover"].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault(); dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault(); dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", event => openFile(event.dataTransfer.files[0]));
dropZone.addEventListener("click", () => {
  if (!isLiveScreen) fileInput.click();
});
dropZone.addEventListener("keydown", event => {
  if (!isLiveScreen && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    fileInput.click();
  }
});

document.querySelector("#sampleButton").addEventListener("click", () => {
  stopScreenShare({ updateResult: false });
  stopVideo();
  temporalHistory = [];
  tracePoints = [];
  resetInpainting();
  videoControls.hidden = true;
  const sample = new OffscreenCanvas(1280, 720);
  const c = sample.getContext("2d");
  c.fillStyle = "#f5f2ea"; c.fillRect(0, 0, 1280, 720);
  c.fillStyle = "#23251f"; c.font = "bold 72px system-ui"; c.fillText("Cursor test page", 90, 145);
  c.fillStyle = "#d9ddd0"; c.fillRect(90, 220, 1100, 340);
  c.fillStyle = "#65705b"; c.fillRect(465, 430, 350, 76);
  c.fillStyle = "white"; c.font = "30px system-ui"; c.fillText("A cursor is near this button", 445, 620);
  // Familiar black arrow cursor with white outline.
  c.beginPath(); c.moveTo(795, 455); c.lineTo(795, 505); c.lineTo(808, 493); c.lineTo(820, 520);
  c.lineTo(832, 514); c.lineTo(819, 488); c.lineTo(838, 487); c.closePath();
  c.lineWidth = 6; c.strokeStyle = "white"; c.stroke(); c.fillStyle = "black"; c.fill();
  currentImage = sample; emptyState.hidden = true; detect(sample);
});

shareScreenButton.addEventListener("click", async () => {
  if (displayStream) {
    stopScreenShare();
    return;
  }
  if (!navigator.mediaDevices?.getDisplayMedia) {
    resultLabel.textContent = "Screen sharing is not supported in this browser";
    return;
  }

  shareScreenButton.disabled = true;
  resultLabel.textContent = "Choose a screen, window, or tab to share…";
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: 30, max: 60 } },
      audio: false
    });
    stopVideo();
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = undefined;
    }
    video.removeAttribute("src");
    video.load();
    displayStream = stream;
    isLiveScreen = true;
    currentImage = null;
    temporalHistory = [];
    tracePoints = [];
    resetInpainting();
    lastBoxes = [];
    smoothedInferenceMs = undefined;
    videoControls.hidden = true;
    emptyState.hidden = true;
    video.srcObject = stream;
    shareScreenButton.textContent = "Stop sharing";
    shareScreenButton.classList.add("sharing");
    shareScreenButton.setAttribute("aria-pressed", "true");
    stream.getVideoTracks()[0].addEventListener("ended", () => stopScreenShare(), { once: true });
    await video.play();
    resultLabel.textContent = "Detecting cursors on your shared screen";
    videoLoop();
  } catch (error) {
    if (error.name !== "NotAllowedError") resultLabel.textContent = `Could not share screen: ${error.message}`;
    else resultLabel.textContent = "Screen sharing cancelled";
  } finally {
    shareScreenButton.disabled = false;
  }
});

demoVideoButton.addEventListener("click", async () => {
  demoVideoButton.disabled = true;
  resultLabel.textContent = "Loading sample video…";
  try {
    const response = await fetch(`${BASE}video-for-demo.mp4`);
    if (!response.ok) throw new Error(`Could not load sample video (${response.status})`);
    const blob = await response.blob();
    openFile(new File([blob], "video-for-demo.mp4", { type: "video/mp4" }));
  } catch (error) {
    resultLabel.textContent = error.message;
  } finally {
    demoVideoButton.disabled = false;
  }
});

playPause.addEventListener("click", async () => {
  if (video.paused) {
    await video.play();
    if (!videoFrameRequest) videoLoop();
  } else {
    stopVideo();
    updateTransport();
  }
});
timeline.addEventListener("input", () => {
  const previewTime = (video.duration || 0) * Number(timeline.value) / 1000;
  stopVideo();
  currentTimeLabel.textContent = formatTime(previewTime);
  video.currentTime = previewTime;
  clearTimeout(skimTimer);
  // Coalesce dense pointer events while still updating throughout the drag.
  skimTimer = setTimeout(() => detect(video), 40);
});
timeline.addEventListener("change", () => seekTo((video.duration || 0) * Number(timeline.value) / 1000));
stepBack.addEventListener("click", () => { stopVideo(); seekTo(video.currentTime - FRAME_SECONDS); });
stepForward.addEventListener("click", () => { stopVideo(); seekTo(video.currentTime + FRAME_SECONDS); });
playbackRate.addEventListener("change", () => { video.playbackRate = Number(playbackRate.value); });
video.addEventListener("ended", updateTransport);
video.addEventListener("seeked", () => {
  resetInpainting();
  updateTransport();
  detect(video);
});

loadModel();
