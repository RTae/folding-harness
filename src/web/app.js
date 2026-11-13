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
  colourMode: "confidence",
  spinning: false,
  timer: null,
  frame: null,
};

const el = (id) => document.getElementById(id);

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

const four = (value) =>
  value === null || value === undefined ? "—" : value.toFixed(4);
const cid = (id) => `c${String(id).padStart(4, "0")}`;

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
    const loss = run.best_loss === null ? "" : ` — best ${four(run.best_loss)}`;
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
  // Poll while a run is live; a finished run never changes.
  if (state.detail?.population?.running === true) {
    state.timer = setTimeout(() => loadRun(false), 1500);
  }
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
  if (!state.detail) return;
  renderFacts(state.detail);
  renderCurve(state.detail.population);
  renderCycles(state.detail.population);
  renderPopulation(state.detail.population);
  renderLineage(state.detail.population);
  renderBreakdown(state.detail);
  renderSequence(state.detail);
  renderEvents();
  renderLegend();
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
    ["best loss", four(population?.stats?.best_loss)],
    ["mean", four(population?.stats?.mean_loss)],
    ["backend", manifest?.backend],
    ["model", manifest?.model],
    ["seed", manifest?.seed],
    ["config", manifest?.config_sha256?.slice(0, 12)],
    // The server renders this line with Usage::summary, the same call the
    // terminal report makes, so the two interfaces cannot say different
    // things about whether the model counts were measured.
    ["spent", detail.spend ?? null],
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

// ------------------------------------------------------------- the loss curve

function renderCurve(population) {
  const records = population?.records ?? [];
  const host = el("curve");
  const note = el("curve-note");

  const points = records
    .map((record, index) => ({
      cycle: record.cycle ?? index + 1,
      best: record.best_loss,
      mean: record.mean_loss,
      improved: record.improved,
      skill: record.skill,
      admitted: record.admitted,
      rejected: record.rejected,
    }))
    .filter((point) => point.best !== null && point.best !== undefined);

  if (points.length < 2) {
    host.innerHTML = "";
    note.hidden = false;
    return;
  }
  note.hidden = true;

  const width = 560;
  const height = 230;
  const pad = { top: 16, right: 16, bottom: 30, left: 62 };
  const values = points.flatMap((point) =>
    [point.best, point.mean].filter((value) => value !== null && value !== undefined),
  );
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || 1;
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const x = (index) =>
    pad.left + (points.length === 1 ? 0 : (index / (points.length - 1)) * plotWidth);
  const y = (value) => pad.top + (1 - (value - low) / span) * plotHeight;

  const line = (key) =>
    points
      .map((point, index) =>
        point[key] === null || point[key] === undefined
          ? null
          : `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(point[key]).toFixed(1)}`,
      )
      .filter(Boolean)
      .join(" ");

  // A band between best and mean shows how spread the population is.
  const haveMean = points.every(
    (point) => point.mean !== null && point.mean !== undefined,
  );
  const band = haveMean
    ? `${points.map((point, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(point.best).toFixed(1)}`).join(" ")} ` +
      `${points
        .map(
          (point, index) =>
            `L${x(points.length - 1 - index).toFixed(1)},${y(points[points.length - 1 - index].mean).toFixed(1)}`,
        )
        .join(" ")} Z`
    : "";

  const improvedDots = points
    .map((point, index) =>
      point.improved
        ? `<circle class="dot" cx="${x(index).toFixed(1)}" cy="${y(point.best).toFixed(1)}" r="3" />`
        : "",
    )
    .join("");

  const hits = points
    .map(
      (point, index) =>
        `<rect class="hit" data-index="${index}" x="${(x(index) - plotWidth / points.length / 2).toFixed(1)}"
               y="${pad.top}" width="${(plotWidth / points.length).toFixed(1)}" height="${plotHeight}" />`,
    )
    .join("");

  host.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img"
         aria-label="loss per cycle, best from ${four(high)} to ${four(low)}">
      <line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" />
      <line class="axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" />
      <text x="6" y="${pad.top + 4}">${four(high)}</text>
      <text x="6" y="${height - pad.bottom}">${four(low)}</text>
      <text x="${pad.left}" y="${height - 8}">cycle 1</text>
      <text x="${width - pad.right}" y="${height - 8}" text-anchor="end">${points[points.length - 1].cycle}</text>
      ${band ? `<path class="band" d="${band}" />` : ""}
      ${haveMean ? `<path class="curve-mean" d="${line("mean")}" />` : ""}
      <path class="curve" d="${line("best")}" />
      ${improvedDots}
      <g id="curve-marker"></g>
      ${hits}
    </svg>`;

  const svg = host.querySelector("svg");
  svg.addEventListener("mousemove", (event) => {
    const target = event.target.closest(".hit");
    if (!target) return;
    const point = points[Number(target.dataset.index)];
    const index = Number(target.dataset.index);
    el("curve-marker").innerHTML =
      `<line class="marker" x1="${x(index)}" y1="${pad.top}" x2="${x(index)}" y2="${height - pad.bottom}" />`;
    showTooltip(
      event,
      [
        `cycle ${point.cycle}  ${point.skill ?? ""}`,
        `best ${four(point.best)}${point.improved ? "  (improved)" : ""}`,
        `mean ${four(point.mean)}`,
        `+${point.admitted} / -${point.rejected}`,
      ].join("\n"),
    );
  });
  svg.addEventListener("mouseleave", () => {
    el("curve-marker").innerHTML = "";
    hideTooltip();
  });
}

function renderCycles(population) {
  const records = population?.records ?? [];
  const host = el("cycles");
  host.innerHTML = "";
  if (records.length === 0) return;

  const most = Math.max(
    1,
    ...records.map((record) => (record.admitted ?? 0) + (record.rejected ?? 0)),
  );

  for (const record of records) {
    const admitted = record.admitted ?? 0;
    const rejected = record.rejected ?? 0;
    const cell = document.createElement("div");
    cell.className = `cycle${record.improved ? " improved" : ""}`;

    const good = document.createElement("div");
    good.className = "admitted";
    good.style.height = `${(admitted / most) * 100}%`;
    const bad = document.createElement("div");
    bad.className = "rejected";
    bad.style.height = `${(rejected / most) * 100}%`;
    cell.append(bad, good);

    cell.addEventListener("mousemove", (event) =>
      showTooltip(
        event,
        `cycle ${record.cycle}  ${record.skill ?? ""}\n+${admitted} admitted / -${rejected} rejected`,
      ),
    );
    cell.addEventListener("mouseleave", hideTooltip);
    host.append(cell);
  }
}

// --------------------------------------------------------------- the lineage

function renderLineage(population) {
  const members = population?.members ?? [];
  const host = el("lineage");
  if (members.length === 0) {
    host.innerHTML = "";
    return;
  }

  const present = new Set(members.map((member) => member.id));
  const children = new Map();
  for (const member of members) {
    const parent = present.has(member.parent) ? member.parent : null;
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent).push(member);
  }
  for (const list of children.values()) {
    list.sort((a, b) => (a.loss ?? 1e9) - (b.loss ?? 1e9));
  }

  // Lay out as a tree: depth across, siblings down.
  const nodes = [];
  let row = 0;
  const walk = (parent, depth) => {
    for (const member of children.get(parent) ?? []) {
      nodes.push({ member, depth, row: row++ });
      walk(member.id, depth + 1);
    }
  };
  walk(null, 0);
  if (nodes.length === 0) {
    host.innerHTML = "";
    return;
  }

  const losses = members
    .map((member) => member.loss)
    .filter((loss) => loss !== null && loss !== undefined);
  const low = Math.min(...losses);
  const high = Math.max(...losses);
  const colourFor = (loss) => {
    if (loss === null || loss === undefined) return "#5a6273";
    const t = high === low ? 0 : (loss - low) / (high - low);
    // Green for the best loss through to red for the worst.
    const hue = 140 - 140 * t;
    return `hsl(${hue.toFixed(0)} 55% 55%)`;
  };

  const depth = Math.max(...nodes.map((node) => node.depth));
  const stepX = 58;
  const stepY = 18;
  const pad = 22;
  const width = pad * 2 + depth * stepX + 60;
  const height = pad * 2 + nodes.length * stepY;
  const at = new Map(nodes.map((node) => [node.member.id, node]));
  const px = (node) => pad + node.depth * stepX;
  const py = (node) => pad + node.row * stepY;

  const edges = nodes
    .filter((node) => present.has(node.member.parent) && at.has(node.member.parent))
    .map((node) => {
      const parent = at.get(node.member.parent);
      const x1 = px(parent);
      const y1 = py(parent);
      const x2 = px(node);
      const y2 = py(node);
      const mid = (x1 + x2) / 2;
      return `<path class="edge" d="M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}" />`;
    })
    .join("");

  const circles = nodes
    .map((node) => {
      const selected = node.member.id === state.selected ? " selected" : "";
      return `<circle class="node${selected}" data-id="${node.member.id}"
                r="5" cx="${px(node)}" cy="${py(node)}"
                fill="${colourFor(node.member.loss)}" />
              <text class="node-label" x="${px(node) + 9}" y="${py(node) + 3}">${cid(node.member.id)}</text>`;
    })
    .join("");

  host.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img"
       aria-label="lineage of ${nodes.length} retained candidates">${edges}${circles}</svg>`;

  for (const circle of host.querySelectorAll(".node")) {
    const id = Number(circle.dataset.id);
    const member = members.find((candidate) => candidate.id === id);
    circle.addEventListener("click", () => select(id));
    circle.addEventListener("mousemove", (event) =>
      showTooltip(
        event,
        `${cid(id)}  loss ${four(member?.loss)}\ncycle ${member?.cycle}  ${originOf(member)}`,
      ),
    );
    circle.addEventListener("mouseleave", hideTooltip);
  }
}

function originOf(member) {
  if (!member) return "";
  return member.origin === "founder" ? "founder" : (member.origin?.skill ?? "");
}

// ------------------------------------------------------------- the population

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

    for (const cell of [
      rank + 1,
      cid(candidate.id),
      candidate.parent === null ? "—" : cid(candidate.parent),
      originOf(candidate),
      four(candidate.loss),
      metrics.iptm?.toFixed(3) ?? "—",
      metrics.interface_plddt?.toFixed(1) ?? "—",
      metrics.pae_interface?.toFixed(1) ?? "—",
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

// --------------------------------------------------------- the loss breakdown

/// Mirrors the weights and normalisation in src/metrics.rs. Kept here rather
/// than sent over the wire so the pane works for any stored run.
const LOSS_TERMS = [
  ["iptm", 3.0, (m) => clamp01(1 - m.iptm)],
  ["interface_plddt", 2.0, (m) => clamp01(1 - m.interface_plddt / 100)],
  ["pae_interface", 1.0, (m) => clamp01(m.pae_interface / 30)],
  ["interface_contacts", 1.0, (m) => clamp01(1 - m.interface_contacts / 60)],
  ["clashes", 2.0, (m) => clamp01(m.clashes / 10)],
  ["hydrophobic_patch", 0.5, (m) => clamp01(m.hydrophobic_patch)],
  ["net_charge", 0.25, (m) => clamp01(Math.abs(m.net_charge) / 10)],
  ["plddt_mean", 0.5, (m) => clamp01(1 - m.plddt_mean / 100)],
];

const clamp01 = (value) => Math.min(1, Math.max(0, value));

function renderBreakdown(detail) {
  const host = el("breakdown");
  host.innerHTML = "";
  const members = detail.population?.members ?? [];
  const candidate = members.find((member) => member.id === state.selected);
  el("detail-of").textContent = candidate ? cid(candidate.id) : "";

  const metrics = candidate?.metrics;
  if (!metrics) {
    host.innerHTML = '<p class="note">Select a candidate.</p>';
    return;
  }

  const weights = LOSS_TERMS.reduce((sum, [, weight]) => sum + weight, 0);
  const terms = LOSS_TERMS.map(([name, weight, penalty]) => ({
    name,
    value: (penalty(metrics) * weight) / weights,
  })).sort((a, b) => b.value - a.value);
  const total = terms.reduce((sum, term) => sum + term.value, 0) || 1;

  for (const term of terms) {
    const row = document.createElement("div");
    row.className = "term";

    const name = document.createElement("span");
    name.className = "name";
    name.textContent = term.name;

    const track = document.createElement("span");
    const bar = document.createElement("span");
    bar.className = `bar${term.value > total * 0.25 ? " heavy" : ""}`;
    bar.style.display = "block";
    bar.style.width = `${Math.max(1, (term.value / total) * 100)}%`;
    track.append(bar);

    const value = document.createElement("span");
    value.className = "value";
    value.textContent = term.value.toFixed(4);

    row.append(name, track, value);
    host.append(row);
  }
}

// ------------------------------------------------------------- the sequence

function renderSequence(detail) {
  const host = el("sequence");
  host.innerHTML = "";

  const scope = detail.scope;
  const members = detail.population?.members ?? [];
  const candidate = members.find((member) => member.id === state.selected);

  if (!scope || !candidate) {
    host.innerHTML = `<p class="note">${
      scope ? "Select a candidate." : "task.yaml is not readable, so the mutable scope is unknown."
    }</p>`;
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
        span.className = changed ? "res-changed" : mutable ? "res-mutable" : "res-fixed";
        span.title = `${chain.label}:${position + 1} ${chain.founder[position]}→${residue}`;
        line.append(span);
      }
      row.append(line);
      host.append(row);
    }
  }
}

// ---------------------------------------------------------------- the events

function renderEvents() {
  const list = el("events");
  const first = state.events[0]?.at_ms ?? 0;
  el("event-count").textContent = state.events.length ? `${state.events.length}` : "";

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
function summarise(r) {
  switch (r.event) {
    case "task_started":
      return `${r.name} on ${r.backend} (${r.model}), ${r.cycles} cycles`;
    case "task_resumed":
      return `resumed after cycle ${r.cycle} of ${r.cycles_planned}, seed ${r.seed}; the generator was re-seeded, so choices from here diverge from one uninterrupted run`;
    case "founder_scored":
      return `founder ${cid(r.candidate)} loss ${four(r.loss)}`;
    case "cycle_started":
      return `cycle ${r.cycle}: parent ${cid(r.parent)}, population ${r.population}`;
    case "skill_routed":
      return `legal [${r.legal.join(", ")}]${r.blocked.length ? `, blocked ${r.blocked.length}` : ""}`;
    case "skill_chosen":
      return `chose ${r.skill}: ${r.rationale}`;
    case "proposals":
      return `${r.skill} produced ${r.count} proposal(s)`;
    case "scored":
      return `${cid(r.candidate)} loss ${four(r.loss)} (ipTM ${r.iptm.toFixed(3)})`;
    case "admitted":
      return `admitted ${cid(r.candidate)} loss ${four(r.loss)} [${r.substitutions}]`;
    case "rejected":
      return `rejected ${cid(r.candidate)}: ${r.reason}`;
    case "fold_failed":
      return `fold failed for ${cid(r.candidate)}: ${r.detail}`;
    case "cycle_finished":
      return `cycle ${r.cycle}: +${r.admitted} / -${r.rejected}, best ${four(r.best_loss)}${r.improved ? " (improved)" : ""}`;
    case "stagnation":
      return `${r.cycles_without_improvement} cycles without improvement`;
    case "note":
      return `${r.source}: ${r.text}`;
    case "task_finished":
      return `${r.cycles_completed} cycles, ${r.folds} folds, best ${four(r.best_loss)}`;
    case "task_failed":
      return r.detail;
    default:
      return "";
  }
}

// --------------------------------------------------------------- structure 3d

const BANDS = [
  ["very high (≥90)", "#4b8ef0"],
  ["confident (70–90)", "#4ec9d8"],
  ["low (50–70)", "#d9b166"],
  ["very low (<50)", "#e2605e"],
];

function confidenceColour(plddt) {
  if (plddt >= 90) return BANDS[0][1];
  if (plddt >= 70) return BANDS[1][1];
  if (plddt >= 50) return BANDS[2][1];
  return BANDS[3][1];
}

/// What each residue of the trace should be coloured, under the chosen mode.
function colourTable() {
  const snapshot = state.structure;
  if (!snapshot) return new Map();

  const scope = state.detail?.scope;
  const members = state.detail?.population?.members ?? [];
  const candidate = members.find((member) => member.id === state.selected);
  const byId = new Map((scope?.chains ?? []).map((chain) => [chain.id, chain]));

  const table = new Map();
  for (const residue of snapshot.ca_trace) {
    const key = `${residue.chain_index}:${residue.index}`;
    if (!residue.designed) {
      table.set(key, "#4d5566");
      continue;
    }

    const chain = byId.get(residue.chain_index);
    const current = candidate?.sequence?.chains?.[residue.chain_index];
    const founder = chain?.founder;
    const mutable = chain?.mutable?.[residue.index] === true;
    const changed =
      current && founder && current[residue.index] !== founder[residue.index];

    switch (state.colourMode) {
      case "changed":
        table.set(key, changed ? "#f0b429" : "#3f4756");
        break;
      case "scope":
        table.set(key, mutable ? "#63c98a" : "#3f4756");
        break;
      case "chain":
        table.set(key, residue.chain_index === 0 ? "#4ec9d8" : "#b98fd0");
        break;
      default:
        table.set(key, confidenceColour(residue.plddt));
    }
  }
  return table;
}

function renderLegend() {
  const host = el("legend");
  const entries =
    state.colourMode === "confidence"
      ? [...BANDS, ["target (fixed)", "#4d5566"]]
      : state.colourMode === "changed"
        ? [
            ["changed from the start", "#f0b429"],
            ["unchanged", "#3f4756"],
            ["target (fixed)", "#4d5566"],
          ]
        : state.colourMode === "scope"
          ? [
              ["may change", "#63c98a"],
              ["fixed", "#3f4756"],
              ["target (fixed)", "#4d5566"],
            ]
          : [
              ["first binder chain", "#4ec9d8"],
              ["other binder chain", "#b98fd0"],
              ["target (fixed)", "#4d5566"],
            ];

  host.innerHTML = entries
    .map(
      ([label, colour]) =>
        `<li><i class="band" style="background:${colour}"></i>${label}</li>`,
    )
    .join("");
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
  const scale = (Math.min(width, height) / 2 - 14) * (camera.zoom / extent);

  for (const point of points) {
    point.sx = width / 2 + point.x * scale;
    point.sy = height / 2 - point.y * scale;
  }
  return points;
}

/// Match the backing store to the element's size and the display's pixel
/// density, so the backbone is crisp rather than blurred by upscaling.
function sizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 520;
  const height = Math.round(width * 0.78);
  if (canvas.width !== Math.round(width * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  canvas.style.height = `${height}px`;
  return { width, height, ratio };
}

function drawStructure() {
  const canvas = el("structure");
  const { width, height, ratio } = sizeCanvas(canvas);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
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
      26,
    );
    return;
  }

  const designed = snapshot.ca_trace.filter((residue) => residue.designed).length;
  label.textContent = `${cid(snapshot.candidate)} — ${snapshot.ca_trace.length} residues, ${designed} designed`;

  const points = project(snapshot.ca_trace, state.camera, width, height);
  const colours = colourTable();

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
  const spread = nearest - farthest || 1;

  context.lineCap = "round";
  context.lineJoin = "round";
  for (const segment of segments) {
    const residue = segment.from.residue;
    const near = (segment.depth - farthest) / spread;
    context.globalAlpha = residue.designed ? 0.4 + 0.6 * near : 0.18 + 0.25 * near;
    context.lineWidth = residue.designed ? 1.6 + 3 * near : 1 + 1.4 * near;
    context.strokeStyle = colours.get(`${residue.chain_index}:${residue.index}`) ?? "#5a6273";
    context.beginPath();
    context.moveTo(segment.from.sx, segment.from.sy);
    context.lineTo(segment.to.sx, segment.to.sy);
    context.stroke();
  }

  // Mark the residues the mode is drawing attention to, so they read as
  // points on the chain rather than only as a change of colour.
  if (state.colourMode === "changed" || state.colourMode === "scope") {
    const wanted = state.colourMode === "changed" ? "#f0b429" : "#63c98a";
    for (const point of points) {
      const key = `${point.residue.chain_index}:${point.residue.index}`;
      if (colours.get(key) !== wanted) continue;
      const near = (point.depth - farthest) / spread;
      context.globalAlpha = 0.5 + 0.5 * near;
      context.fillStyle = wanted;
      context.beginPath();
      context.arc(point.sx, point.sy, 2 + 1.6 * near, 0, Math.PI * 2);
      context.fill();
    }
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
    state.camera.yaw += (event.clientX - lastX) * 0.01;
    // Stop short of the poles, where the projection degenerates.
    state.camera.pitch = Math.max(
      -1.4,
      Math.min(1.4, state.camera.pitch + (event.clientY - lastY) * 0.01),
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

  canvas.addEventListener("dblclick", resetView);
  el("reset-view").addEventListener("click", resetView);
  window.addEventListener("resize", drawStructure);

  el("colour-mode").addEventListener("change", (event) => {
    state.colourMode = event.target.value;
    renderLegend();
    drawStructure();
  });

  el("spin").addEventListener("change", (event) => {
    state.spinning = event.target.checked;
    if (state.spinning) spin();
    else cancelAnimationFrame(state.frame);
  });
}

function resetView() {
  state.camera = { yaw: 0.6, pitch: 0.25, zoom: 1 };
  drawStructure();
}

function spin() {
  if (!state.spinning) return;
  state.camera.yaw += 0.006;
  drawStructure();
  state.frame = requestAnimationFrame(spin);
}

// ---------------------------------------------------------------- tooltips

function showTooltip(event, text) {
  const tip = el("tooltip");
  tip.textContent = text;
  tip.hidden = false;
  const pad = 14;
  const right = event.clientX + pad + tip.offsetWidth > window.innerWidth;
  tip.style.left = `${right ? event.clientX - tip.offsetWidth - pad : event.clientX + pad}px`;
  tip.style.top = `${Math.min(event.clientY + pad, window.innerHeight - tip.offsetHeight - 8)}px`;
}

function hideTooltip() {
  el("tooltip").hidden = true;
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
