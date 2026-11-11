// The browser half of the harness view. No framework and no build step: the
// server compiles this file into the binary and serves it as-is.
//
// Everything here is a reader. The page polls the task directory through the
// API and never writes, so having it open cannot affect a run.

"use strict";

const state = {
  runId: null,
  detail: null,
  events: [],
  lastSeq: 0,
  selected: null,
  structure: null,
  camera: { yaw: 0.6, pitch: 0.25, zoom: 1 },
  timer: null,
};

const el = (id) => document.getElementById(id);

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

function fourDecimals(value) {
  return value === null || value === undefined ? "—" : value.toFixed(4);
}

// ---------------------------------------------------------------- run picker

async function loadRuns() {
  const runs = await getJson("/api/runs");
  const picker = el("runs");
  picker.innerHTML = "";

  if (runs.length === 0) {
    const option = document.createElement("option");
    option.textContent = "no runs yet — try: fh run task.yaml";
    picker.append(option);
    return;
  }

  for (const run of runs) {
    const option = document.createElement("option");
    option.value = run.id;
    const loss = run.best_loss === null ? "" : ` — best ${fourDecimals(run.best_loss)}`;
    option.textContent =
      `${run.name}  (${run.cycles_completed}/${run.cycles_planned})${loss}`;
    picker.append(option);
  }

  const wanted = new URLSearchParams(location.search).get("run");
  state.runId = runs.some((run) => run.id === wanted) ? wanted : runs[0].id;
  picker.value = state.runId;
}

// ------------------------------------------------------------------- loading

async function loadRun(reset) {
  if (!state.runId) return;
  if (reset) {
    state.events = [];
    state.lastSeq = 0;
    state.selected = null;
    state.structure = null;
  }

  state.detail = await getJson(`/api/runs/${state.runId}`);

  const fresh = await getJson(
    `/api/runs/${state.runId}/trace?after=${state.lastSeq}`,
  );
  if (fresh.length > 0) {
    state.lastSeq = fresh[fresh.length - 1].seq;
    state.events.push(...fresh);
  }

  const members = state.detail?.population?.members ?? [];
  if (state.selected === null && members.length > 0) {
    await select(members[0].id);
  }

  render();
  schedule();
}

function schedule() {
  clearTimeout(state.timer);
  const running = state.detail?.population?.running === true;
  // Poll while a run is live; a finished run never changes.
  if (running) state.timer = setTimeout(() => loadRun(false), 1500);
}

async function select(candidateId) {
  state.selected = candidateId;
  state.structure = null;
  try {
    state.structure = await getJson(
      `/api/runs/${state.runId}/structure/${candidateId}`,
    );
  } catch {
    // A candidate the backend returned no trace for. The pane says so.
  }
  render();
}

// ------------------------------------------------------------------ the page

function render() {
  const detail = state.detail;
  if (!detail) return;
  renderFacts(detail);
  renderCurve(detail.population);
  renderPopulation(detail.population);
  renderSequence(detail);
  renderEvents();
  drawStructure();
}

function renderFacts(detail) {
  const manifest = detail.manifest;
  const population = detail.population;
  const running = population?.running === true;

  const pill = el("state");
  pill.textContent = running ? "running" : "finished";
  pill.className = `pill ${running ? "running" : "finished"}`;

  const facts = [
    ["cycle", population ? `${population.cycle}/${population.cycles_planned}` : null],
    ["best loss", fourDecimals(population?.stats?.best_loss)],
    ["mean", fourDecimals(population?.stats?.mean_loss)],
    ["folds", population?.folds],
    ["backend", manifest?.backend],
    ["model", manifest?.model],
    ["seed", manifest?.seed],
    ["config", manifest?.config_sha256?.slice(0, 12)],
    ["scope", detail.scope?.summary],
  ];

  const list = el("facts");
  list.innerHTML = "";
  for (const [label, value] of facts) {
    if (value === null || value === undefined || value === "") continue;
    const row = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const definition = document.createElement("dd");
    definition.textContent = String(value);
    row.append(term, definition);
    list.append(row);
  }

  el("disclaimer").textContent = manifest?.backend?.startsWith("fixture")
    ? "The fixture backend predicts nothing: it scores against an optimum planted from the run seed. This shape is synthetic."
    : "A prediction is not a measurement. Computational results need experimental validation.";
}

function renderCurve(population) {
  const curve = population?.loss_curve ?? [];
  const host = el("curve");
  const note = el("curve-note");

  if (curve.length < 2) {
    host.innerHTML = "";
    note.hidden = false;
    return;
  }
  note.hidden = true;

  const width = 520;
  const height = 200;
  const pad = { top: 14, right: 14, bottom: 26, left: 56 };
  const low = Math.min(...curve);
  const high = Math.max(...curve);
  const span = high - low || 1;
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const x = (index) => pad.left + (index / (curve.length - 1)) * plotWidth;
  const y = (value) => pad.top + (1 - (value - low) / span) * plotHeight;

  const path = curve.map((value, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  const dots = curve
    .map((value, index) => `<circle class="dot" cx="${x(index).toFixed(1)}" cy="${y(value).toFixed(1)}" r="2" />`)
    .join("");

  host.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" role="img"
         aria-label="best loss per cycle, ${fourDecimals(high)} down to ${fourDecimals(low)}">
      <line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
      <line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" />
      <text x="4" y="${pad.top + 4}">${fourDecimals(high)}</text>
      <text x="4" y="${height - pad.bottom}">${fourDecimals(low)}</text>
      <text x="${pad.left}" y="${height - 6}">cycle 1</text>
      <text x="${width - pad.right}" y="${height - 6}" text-anchor="end">${curve.length}</text>
      <path class="curve" d="${path}" />
      ${dots}
    </svg>`;
}

function renderPopulation(population) {
  const members = population?.members ?? [];
  el("population-count").textContent = members.length
    ? `${members.length} retained`
    : "";

  const body = el("population").querySelector("tbody");
  body.innerHTML = "";

  members.forEach((candidate, rank) => {
    const metrics = candidate.metrics ?? {};
    const row = document.createElement("tr");
    if (candidate.id === state.selected) row.classList.add("selected");
    if (rank === 0) row.classList.add("best");

    const origin =
      candidate.origin === "founder" ? "founder" : candidate.origin?.skill ?? "—";

    for (const cell of [
      rank + 1,
      `c${String(candidate.id).padStart(4, "0")}`,
      candidate.parent === null ? "—" : `c${String(candidate.parent).padStart(4, "0")}`,
      origin,
      fourDecimals(candidate.loss),
      metrics.iptm?.toFixed(3) ?? "—",
      metrics.interface_plddt?.toFixed(1) ?? "—",
      metrics.clashes ?? "—",
      candidate.substitutions?.length ?? 0,
    ]) {
      const td = document.createElement("td");
      td.textContent = String(cell);
      row.append(td);
    }

    row.addEventListener("click", () => select(candidate.id));
    body.append(row);
  });
}

function renderSequence(detail) {
  const host = el("sequence");
  host.innerHTML = "";

  const scope = detail.scope;
  const members = detail.population?.members ?? [];
  const candidate = members.find((member) => member.id === state.selected);
  el("sequence-of").textContent = candidate
    ? `c${String(candidate.id).padStart(4, "0")}`
    : "";

  if (!scope || !candidate) {
    host.textContent = scope
      ? "Select a candidate."
      : "task.yaml is not readable, so the mutable scope is unknown.";
    return;
  }

  const perRow = 60;
  for (const chain of scope.chains) {
    if (chain.role !== "binder") continue;
    const current = candidate.sequence?.chains?.[chain.id] ?? chain.founder;

    const heading = document.createElement("div");
    heading.className = "note";
    heading.textContent = `chain ${chain.label} — ${current.length} residues`;
    host.append(heading);

    for (let start = 0; start < current.length; start += perRow) {
      const row = document.createElement("div");
      row.className = "row";

      const at = document.createElement("span");
      at.className = "at";
      at.textContent = String(start + 1);
      row.append(at);

      const line = document.createElement("span");
      for (let offset = 0; offset < perRow && start + offset < current.length; offset += 1) {
        const position = start + offset;
        const residue = current[position];
        const mutable = chain.mutable[position] === true;
        const changed = chain.founder[position] !== residue;

        const span = document.createElement("b");
        span.textContent = residue;
        span.className = changed
          ? "res-changed"
          : mutable
            ? "res-mutable"
            : "res-fixed";
        line.append(span);
      }
      row.append(line);
      host.append(row);
    }
  }
}

function renderEvents() {
  const list = el("events");
  const first = state.events[0]?.at_ms ?? 0;
  el("event-count").textContent = state.events.length
    ? `${state.events.length}`
    : "";

  // Newest last, matching the trace file, and only the tail is worth drawing.
  const tail = state.events.slice(-400);
  list.innerHTML = "";
  for (const record of tail) {
    const item = document.createElement("li");

    const at = document.createElement("span");
    at.className = "at";
    at.textContent = `${((record.at_ms - first) / 1000).toFixed(1)}s`;

    const kind = document.createElement("span");
    kind.className = `kind kind-${record.event}`;
    kind.textContent = record.event;

    const text = document.createElement("span");
    text.textContent = summarise(record);

    item.append(at, kind, text);
    list.append(item);
  }
  const scroller = list.parentElement;
  scroller.scrollTop = scroller.scrollHeight;
}

// The trace is a tagged union; this says enough about each arm to follow a run.
function summarise(record) {
  const r = record;
  switch (r.event) {
    case "task_started":
      return `${r.name} on ${r.backend} (${r.model}), ${r.cycles} cycles`;
    case "founder_scored":
      return `founder c${String(r.candidate).padStart(4, "0")} loss ${fourDecimals(r.loss)}`;
    case "cycle_started":
      return `cycle ${r.cycle}: parent c${String(r.parent).padStart(4, "0")}, population ${r.population}`;
    case "skill_routed":
      return `legal [${r.legal.join(", ")}]${r.blocked.length ? `, blocked ${r.blocked.length}` : ""}`;
    case "skill_chosen":
      return `chose ${r.skill}: ${r.rationale}`;
    case "proposals":
    case "proposals_generated":
      return `${r.skill} produced ${r.count} proposal(s)`;
    case "scored":
      return `c${String(r.candidate).padStart(4, "0")} loss ${fourDecimals(r.loss)} (ipTM ${r.iptm.toFixed(3)})`;
    case "admitted":
      return `admitted c${String(r.candidate).padStart(4, "0")} loss ${fourDecimals(r.loss)} [${r.substitutions}]`;
    case "rejected":
      return `rejected c${String(r.candidate).padStart(4, "0")}: ${r.reason}`;
    case "fold_failed":
      return `fold failed for c${String(r.candidate).padStart(4, "0")}: ${r.detail}`;
    case "cycle_finished":
      return `cycle ${r.cycle}: +${r.admitted} / -${r.rejected}, best ${fourDecimals(r.best_loss)}${r.improved ? " (improved)" : ""}`;
    case "stagnation":
      return `${r.cycles_without_improvement} cycles without improvement`;
    case "note":
      return `${r.source}: ${r.text}`;
    case "task_finished":
      return `${r.cycles_completed} cycles, ${r.folds} folds, best ${fourDecimals(r.best_loss)}`;
    case "task_failed":
      return r.detail;
    default:
      return "";
  }
}

// --------------------------------------------------------------- structure 3d

function band(plddt) {
  if (plddt >= 90) return "#4b8ef0";
  if (plddt >= 70) return "#4ec9d8";
  if (plddt >= 50) return "#d9b166";
  return "#e2605e";
}

function project(trace, camera, width, height) {
  const count = trace.length;
  const centre = [0, 0, 0];
  for (const residue of trace) {
    centre[0] += residue.xyz[0] / count;
    centre[1] += residue.xyz[1] / count;
    centre[2] += residue.xyz[2] / count;
  }

  const cosYaw = Math.cos(camera.yaw);
  const sinYaw = Math.sin(camera.yaw);
  const cosPitch = Math.cos(camera.pitch);
  const sinPitch = Math.sin(camera.pitch);

  const points = trace.map((residue) => {
    const x0 = residue.xyz[0] - centre[0];
    const y0 = residue.xyz[1] - centre[1];
    const z0 = residue.xyz[2] - centre[2];

    const x1 = x0 * cosYaw + z0 * sinYaw;
    const z1 = -x0 * sinYaw + z0 * cosYaw;
    const y2 = y0 * cosPitch - z1 * sinPitch;
    const z2 = y0 * sinPitch + z1 * cosPitch;

    return { x: x1, y: y2, depth: z2, residue };
  });

  let extent = 1e-6;
  for (const point of points) {
    extent = Math.max(extent, Math.abs(point.x), Math.abs(point.y));
  }
  const scale = (Math.min(width, height) / 2 - 12) * (camera.zoom / extent);

  for (const point of points) {
    point.sx = width / 2 + point.x * scale;
    point.sy = height / 2 - point.y * scale;
  }
  return points;
}

function drawStructure() {
  const canvas = el("structure");
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);

  const snapshot = state.structure;
  const label = el("structure-of");

  if (!snapshot || snapshot.ca_trace.length === 0) {
    label.textContent = "";
    context.fillStyle = "#8b94a3";
    context.font = "13px ui-monospace, monospace";
    context.fillText(
      state.selected === null
        ? "Select a candidate."
        : "No stored backbone for this candidate.",
      16,
      24,
    );
    return;
  }

  label.textContent =
    `c${String(snapshot.candidate).padStart(4, "0")} — ${snapshot.ca_trace.length} residues`;

  const points = project(snapshot.ca_trace, state.camera, width, height);

  // Segments between consecutive residues of the same chain, drawn back to
  // front so nearer parts of the chain cover farther ones.
  const segments = [];
  for (let index = 0; index + 1 < points.length; index += 1) {
    const from = points[index];
    const to = points[index + 1];
    if (from.residue.chain_index !== to.residue.chain_index) continue;
    if (to.residue.index !== from.residue.index + 1) continue;
    segments.push({ from, to, depth: (from.depth + to.depth) / 2 });
  }
  segments.sort((a, b) => a.depth - b.depth);

  const depths = points.map((point) => point.depth);
  const nearest = Math.max(...depths);
  const farthest = Math.min(...depths);
  const depthSpan = nearest - farthest || 1;

  context.lineCap = "round";
  for (const segment of segments) {
    const designed = segment.from.residue.designed;
    const near = (segment.depth - farthest) / depthSpan;
    context.globalAlpha = designed ? 0.45 + 0.55 * near : 0.25 + 0.3 * near;
    context.lineWidth = designed ? 1.4 + 2.2 * near : 1 + 1.2 * near;
    context.strokeStyle = designed ? band(segment.from.residue.plddt) : "#5a6273";
    context.beginPath();
    context.moveTo(segment.from.sx, segment.from.sy);
    context.lineTo(segment.to.sx, segment.to.sy);
    context.stroke();
  }
  context.globalAlpha = 1;
}

function bindStructure() {
  const canvas = el("structure");
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const scale = 0.01;
    state.camera.yaw += (event.clientX - lastX) * scale;
    // Stop short of the poles, where the projection degenerates.
    state.camera.pitch = Math.max(
      -1.4,
      Math.min(1.4, state.camera.pitch + (event.clientY - lastY) * scale),
    );
    lastX = event.clientX;
    lastY = event.clientY;
    drawStructure();
  });

  for (const name of ["pointerup", "pointercancel", "pointerleave"]) {
    canvas.addEventListener(name, () => {
      dragging = false;
    });
  }

  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
      state.camera.zoom = Math.max(0.25, Math.min(6, state.camera.zoom * factor));
      drawStructure();
    },
    { passive: false },
  );

  canvas.addEventListener("dblclick", () => {
    state.camera = { yaw: 0.6, pitch: 0.25, zoom: 1 };
    drawStructure();
  });
}

// ----------------------------------------------------------------- bootstrap

async function main() {
  bindStructure();
  el("runs").addEventListener("change", (event) => {
    state.runId = event.target.value;
    const url = new URL(location.href);
    url.searchParams.set("run", state.runId);
    history.replaceState(null, "", url);
    loadRun(true);
  });

  try {
    await loadRuns();
    await loadRun(true);
  } catch (error) {
    el("facts").textContent = `could not load: ${error.message}`;
  }
}

main();
