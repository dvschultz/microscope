// Microscope viewer. Columns: architecture | units of the current layer | main panel.
// Routes: /<model>, /<model>/<layer>[?b=<branch>], /<model>/<layer>/<unit>

const cols = document.getElementById("cols");
const crumbs = document.getElementById("crumbs");
const tools = document.getElementById("bar-tools");
const jsonCache = {};
let manifest = null;

const DEFAULT_PREFS = { size: "M", show: "vis", fold: {}, labels: "off", circuit: "gradient", lens: "ch" };
const prefs = loadPrefs();
function loadPrefs() {
  try { return Object.assign({}, DEFAULT_PREFS, JSON.parse(localStorage.getItem("microscope") || "{}")); }
  catch { return { ...DEFAULT_PREFS }; }
}
function savePrefs() {
  try { localStorage.setItem("microscope", JSON.stringify(prefs)); } catch {}
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fill = (tpl, o) => tpl.replace(/\{(\w+)\}/g, (_, k) => o[k]);
const fmt = x => x >= 100 ? x.toFixed(0) : x.toFixed(1);
const pct = x => `${(x * 100).toFixed(Math.abs(x) < 0.1 ? 1 : 0)}%`;
const HIDE = `onerror="this.style.visibility='hidden'"`;

// Missing files count as {}. On Cloudflare Pages a missing file is not a 404: the SPA fallback
// (_redirects: /* /index.html 200) answers with index.html, so check the content type too.
async function getJSON(url) {
  if (!(url in jsonCache)) {
    jsonCache[url] = fetch(url).then(async r => {
      if (!r.ok || !(r.headers.get("content-type") || "").includes("json")) return {};
      try { return await r.json(); } catch { return {}; }
    });
  }
  return jsonCache[url];
}
async function getManifest() {
  if (!manifest) {
    manifest = await getJSON("/manifest.json");
    // optional extra lenses, e.g. SAE features; each points back at its base model
    const sae = await getJSON("/manifest_sae.json");
    for (const m of sae.models || []) manifest.models.push({ ...m, base: m.base || manifest.models[0].name });
  }
  return manifest;
}
const baseOf = model => manifest.models.find(m => m.name === (model.base || model.name));
const saeOf = base => manifest.models.find(m => m.base === base.name);
const baseLayerName = layer => layer.base_layer || layer.name;
const assetUrl = (model, layer, file) => `/assets/${model.name}/${layer}/${file}`;
// SAE features are a view inside their base model's URLs: /<base>/<layer>/sae[/<feature>]
const layerPath = (model, layerName) => model.base ? `/${model.base}/${layerName}/sae` : `/${model.name}/${layerName}`;
const unitPath = (model, layerName, u) => `${layerPath(model, layerName)}/${u}`;
const saeLayerOf = (base, layerName) => saeOf(base)?.layers.find(l => baseLayerName(l) === layerName);
const getLabels = (model, layer) => getJSON(assetUrl(model, layer, "labels.json"));
const visUrl = (model, layer, unit) => fill(model.paths.vis, { layer, unit });
const visImg = (model, layer, unit, attrs = "") => `<img loading="lazy" src="${visUrl(model, layer, unit)}" alt="" ${HIDE}${attrs ? " " + attrs : ""}>`;
// A dataset-example crop. Dev: one JPEG per crop. Published build (paths.sprite set): one sprite sheet per unit,
// cols × cols cells, shown as a background-positioned div so a unit's 16 crops cost one request. CSS backgrounds ignore
// loading="lazy", so the sheet URL sits in data-bg and lazySprites() sets it only when the cell nears the viewport.
function exImg(model, layer, unit, rank, label = "") {
  const url = fill(model.paths.example, { layer, unit, rank });
  const sp = model.paths.sprite;
  if (!sp) return `<img loading="lazy" src="${url}" alt="${esc(label)}">`;
  const n = sp.cols - 1, col = rank % sp.cols, row = Math.floor(rank / sp.cols);
  return `<div class="sprite" role="img" aria-label="${esc(label)}" data-bg="${url}" style="` +
    `background-size:${sp.cols * 100}% ${sp.cols * 100}%;background-position:${col * 100 / n}% ${row * 100 / n}%"></div>`;
}
const spriteObserver = "IntersectionObserver" in window && new IntersectionObserver(entries => {
  for (const e of entries) if (e.isIntersecting) {
    e.target.style.backgroundImage = `url('${e.target.dataset.bg}')`;
    delete e.target.dataset.bg;
    spriteObserver.unobserve(e.target);
  }
}, { rootMargin: "400px" });
function lazySprites(root) {
  for (const el of root.querySelectorAll(".sprite[data-bg]")) {
    if (spriteObserver) spriteObserver.observe(el);
    else { el.style.backgroundImage = `url('${el.dataset.bg}')`; delete el.dataset.bg; }
  }
}

function setCrumbs(parts) {
  crumbs.innerHTML = parts.map(([label, href]) =>
    href ? `<a href="${href}" data-link>${esc(label)}</a>` : `<span>${esc(label)}</span>`
  ).join('<span class="sep">/</span>');
}

function navigate(href, replace) {
  if (replace) history.replaceState(null, "", href); else history.pushState(null, "", href);
  route();
}
document.addEventListener("click", e => {
  const fold = e.target.closest("[data-fold]");
  if (fold) {
    const k = fold.dataset.fold;
    prefs.fold[k] = !prefs.fold[k];
    savePrefs();
    fold.closest(".col").classList.toggle("folded", prefs.fold[k]);
    e.preventDefault();
    return;
  }
  const a = e.target.closest("a[data-link]");
  if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
  e.preventDefault();
  navigate(a.getAttribute("href"));
});
window.addEventListener("popstate", route);
window.addEventListener("hashchange", route);

let keyHandler = null;
document.addEventListener("keydown", e => {
  if (keyHandler && !e.target.closest("input")) keyHandler(e);
});

// ---------- columns ----------

const scrollMemo = {};
function column(key, title, body, cls = "") {
  const folded = key !== "main" && prefs.fold[key];
  const btn = key === "main" ? "" : `<button class="fold" data-fold="${key}" title="Collapse">«</button>`;
  return `<section class="col ${key} ${cls} ${folded ? "folded" : ""}" data-col="${key}">
    <div class="head"><span>${esc(title)}</span>${btn}</div>
    <div class="body">${body}</div></section>`;
}
let keepMainScroll = false;  // set by view toggles so a re-render doesn't jump to the top
function mount(html) {
  cols.querySelectorAll(".col").forEach(c => scrollMemo[c.dataset.col] = c.querySelector(".body").scrollTop);
  cols.innerHTML = html;
  cols.querySelectorAll(".col").forEach(c => {
    const keep = c.dataset.col !== "main" || keepMainScroll;
    if (keep && scrollMemo[c.dataset.col] != null) c.querySelector(".body").scrollTop = scrollMemo[c.dataset.col];
  });
  keepMainScroll = false;
  lazySprites(cols);
}
function rerender(fn) { keepMainScroll = true; fn(); }

// Architecture strip, drawn top (output) to bottom (input) like the original Microscope.
function archColumn(view, cur) {
  const model = baseOf(view);
  const sae = saeOf(model);
  const useSae = view !== model || prefs.lens === "sae";
  const layerHref = l => layerPath(useSae && saeLayerOf(model, l.name) ? sae : model, l.name);
  const op = (label, ch) => `<div class="row op"><span class="ch">${ch ?? ""}</span><span class="glyph"></span><span>${label}</span></div>`;
  const layerRow = l => `<a class="row layer ${l.name === cur.layer ? "cur" : ""} ${l.rendered ? "" : "empty"}"
      href="${layerHref(l)}" data-link title="${l.rendered}/${l.channels} rendered${saeLayerOf(model, l.name) ? " · has SAE features" : ""}">
      <span class="ch">${l.channels}</span><span class="glyph"></span><span>${l.name}${saeLayerOf(model, l.name)
        ? `<span class="tag ${useSae ? "on" : ""}">SAE</span>` : ""}</span></a>`;
  const block = l => {
    const b = Object.fromEntries(l.branches.map(x => [x.name, x]));
    const top = ["1x1", "3x3", "5x5", "pool"].map(n => {
      const x = b[n], label = n === "pool" ? "1×1" : n.replace("x", "×");
      const isCur = l.name === cur.layer && cur.branch === n;
      return `<a class="chip ${isCur ? "cur" : ""}" href="/${model.name}/${l.name}?b=${n}" data-link
        title="${l.name} ${n} branch: units ${x.start}–${x.stop - 1}">${label}<small>${x.stop - x.start}</small></a>`;
    }).join("");
    const bottom = `<span class="chip neck none"></span>
      <span class="chip neck" title="1×1 bottleneck">${b["3x3"].bottleneck}</span>
      <span class="chip neck" title="1×1 bottleneck">${b["5x5"].bottleneck}</span>
      <span class="chip neck pool" title="3×3 max pool">pool</span>`;
    return `<div class="block ${l.name === cur.layer ? "cur" : ""}">${top}${bottom}</div>`;
  };
  let html;
  if (model.architecture) {
    // stage-based networks (VGG blocks, ResNet stages), drawn output at top like the Inception strip
    const arch = model.architecture;
    html = arch.kind === "residual" ? op("attention pool → embedding", 1024) : op("fc6–fc8 → classes", 1000);
    for (const st of [...arch.stages].reverse()) {
      const ls = st.layers.map(n => model.layers.find(l => l.name === n)).filter(Boolean);
      const isCur = ls.some(l => l.name === cur.layer);
      html += `<div class="stage ${isCur ? "cur" : ""}"><div class="stage-head"><b>${esc(st.name)}</b><small>${esc(st.note || "")}</small></div>
        ${[...ls].reverse().map(layerRow).join("")}</div>`;
    }
    html += op("input", 3);
  } else {
    const pools = { mixed5a: ["maxpool10", 832], mixed4a: ["maxpool4", 480], mixed3a: ["maxpool1", 192], conv2d1: ["maxpool0", 64] };
    html = op("softmax2", 1008) + op("avgpool0", 1024);
    for (const l of [...model.layers].reverse()) {
      html += layerRow(l);
      if (l.branches) html += block(l);
      if (pools[l.name]) html += op(...pools[l.name]);
    }
    html += op("input", 3);
  }
  return column("arch", model.title, `<nav class="net">${html}</nav>`);
}

function unitsColumn(model, layer, unit) {
  const items = Array.from({ length: layer.channels }, (_, c) =>
    `<a href="${unitPath(model, layer.name, c)}" data-link class="${c === unit ? "cur" : ""}" id="u${c}">
      ${visImg(model, layer.name, c)}<span class="n">${c}</span></a>`).join("");
  return column("units", layer.name, `<div class="unit-list">${items}</div>`);
}

function lensSwitch(model, layer) {
  const base = baseOf(model), sae = saeOf(base);
  const name = baseLayerName(layer);
  const available = sae?.layers.map(l => l.name).join(", ");
  if (!saeLayerOf(base, name))
    return segmented([["ch", "Channels"], ["sae", "SAE features"]], "ch", null,
      available ? `SAE features exist for ${available}` : "No SAE features yet");
  const el = segmented([["ch", "Channels"], ["sae", "SAE features"]], model === base ? "ch" : "sae", v => {
    prefs.lens = v; savePrefs();
    navigate(layerPath(v === "sae" ? sae : base, name));
  });
  el.title = "SAE features: a sparse autoencoder splits this layer into ~8× more, mostly single-meaning directions";
  return el;
}

const searchBox = document.getElementById("search");
searchBox?.addEventListener("keydown", e => {
  if (e.key === "Enter" && searchBox.value.trim())
    navigate(`/${(currentBase || manifest.models[0]).name}/search?q=${encodeURIComponent(searchBox.value.trim())}`);
});

async function searchPage(model, q) {
  setCrumbs([[model.title, `/${model.name}`], [`Search: ${q}`]]);
  document.title = `${q} · Search · Microscope`;
  if (searchBox) searchBox.value = q;
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  const all = await Promise.all(model.layers.map(async l => [l, await getLabels(model, l.name)]));
  let total = 0, labeled = 0;
  const sections = all.map(([l, labels]) => {
    labeled += Object.keys(labels).length;
    const hits = Object.entries(labels).filter(([, v]) => {
      const text = `${v.label} ${v.description}`.toLowerCase();
      return terms.every(t => text.includes(t));
    });
    total += hits.length;
    if (!hits.length) return "";
    const tiles = hits.map(([u, v]) => `<a class="tile" href="${unitPath(model, l.name, u)}" data-link title="${esc(v.description)}">
      ${visImg(model, l.name, u)}<span class="lab">${esc(v.label)}</span><span class="n">${u}</span></a>`).join("");
    return `<div class="branch-head"><b>${l.name}</b><span>${hits.length} match${hits.length > 1 ? "es" : ""}</span></div>
      <div class="grid" style="--tile:150px">${tiles}</div>`;
  }).join("");
  mount(archColumn(model, {}) + column("main", "Search", `<h1>“${esc(q)}”</h1>
    <p class="sub">${total} units whose automatic label or description contains all of: ${terms.map(esc).join(", ")}.
      ${labeled ? `${labeled} units have labels.` : "No labels found yet: run <code>python label_units.py</code>."}
      Labels are written by a vision model from each unit's feature visualization and dataset examples, so treat them as hints.</p>${sections}`));
}

// ---------- routes ----------

async function route() {
  const m = await getManifest();
  const [modelName, layerName, third, fourth] = location.pathname.split("/").filter(Boolean);
  const branch = new URLSearchParams(location.search).get("b");
  keyHandler = null;
  tools.innerHTML = "";
  if (!modelName) return homePage(m);
  if (modelName === "glossary") return glossaryPage(m.models[0]);
  let model = m.models.find(x => x.name === modelName);
  if (!model) return notFound(null, `No model "${modelName}".`);
  currentBase = baseOf(model);
  // old /inceptionv1_sae/<layer>/<f> links -> /inceptionv1/<layer>/sae/<f>
  if (model.base) return navigate(layerName ? layerPath(model, layerName) + (third !== undefined ? `/${third}` : "") : `/${model.base}`, true);
  if (!layerName) return modelPage(model);
  if (layerName === "search") return searchPage(baseOf(model), new URLSearchParams(location.search).get("q") || "");
  let layer = model.layers.find(l => l.name === layerName);
  if (!layer) return notFound(model, `No layer "${layerName}" in ${model.title}.`);
  let unitStr = third;
  if (third === "sae") {
    const saeLayer = saeLayerOf(model, layerName);
    if (!saeLayer) return notFound(model, `No SAE features for ${layerName} yet.`);
    model = saeOf(model); layer = saeLayer; unitStr = fourth;
  }
  if (unitStr === undefined) return layerPage(model, layer, branch);
  const unit = Number(unitStr);
  if (!Number.isInteger(unit) || unit < 0 || unit >= layer.channels)
    return notFound(model, `${layer.name} has units 0–${layer.channels - 1}.`);
  return unitPage(model, layer, unit);
}

function notFound(model, msg) {
  setCrumbs([]);
  document.title = "Not found · Microscope";
  mount((model ? archColumn(model, {}) : "") + column("main", "Not found", `<p class="sub">${esc(msg)}</p>`));
}

let currentBase = null;
function homePage(m) {
  setCrumbs([]);
  document.title = "Microscope";
  currentBase = null;
  const nets = m.models.filter(x => !x.base);
  const cards = nets.map(x => {
    const units = x.layers.reduce((a, l) => a + l.channels, 0);
    const mid = x.layers[Math.floor(x.layers.length * 0.6)];
    const thumbs = Array.from({ length: 10 }, (_, c) => visImg(x, mid.name, c)).join("");
    const sae = saeOf(x);
    return `<a class="net-card" href="/${x.name}" data-link><div class="strip">${thumbs}</div>
      <div class="meta"><b>${esc(x.title)}</b><span>${x.layers.length} layers · ${units.toLocaleString()} units${sae ? ` · SAE features on ${sae.layers.length} layers` : ""}</span>
      <p>${esc(x.description)}</p></div></a>`;
  }).join("");
  mount(column("main", "Networks", `<h1>Networks</h1>
    <p class="sub">Pick a network to browse its layers and units. See the <a href="/glossary" data-link>glossary</a> for what each view shows.</p>
    <div class="net-cards">${cards}</div>`));
}

function modelPage(model) {
  setCrumbs([[model.title]]);
  document.title = `${model.title} · Microscope`;
  const rows = [...model.layers].reverse().map(l => {
    const href = layerPath(model, l.name);
    const meta = `<div class="meta"><b>${l.name}</b>
      <div>${l.channels} units · ${l.spatial[0]}×${l.spatial[1]}</div>
      <div>${l.rendered ? `${l.rendered}/${l.channels} rendered` : "not rendered"}</div></div>`;
    if (!l.rendered) return `<a class="layer-row empty" href="${href}" data-link>${meta}
      <div class="strip">Run <code>python precompute.py --layer ${l.name}</code></div></a>`;
    const imgs = Array.from({ length: Math.min(l.channels, 40) }, (_, c) =>
      visImg(model, l.name, c)).join("");
    return `<a class="layer-row" href="${href}" data-link>${meta}<div class="strip">${imgs}</div></a>`;
  }).join("");
  mount(archColumn(model, {}) + column("main", model.title, `
    <h1>${esc(model.title)}</h1>
    <p class="sub">${esc(model.description)} Layers run from edges and colors at the bottom (${model.layers[0].name}) to object parts and
      whole objects at the top (${model.layers.at(-1).name}). Each unit page shows which units in the layer below build it and which units above use it.
      Dataset examples: ${esc(model.dataset)}.</p>
    <div class="layers">${rows}</div>`));
}

const SIZES = { S: 72, M: 112, L: 176 };
function segmented(options, value, onChange, disabled = "") {
  const el = document.createElement("span");
  el.className = "seg" + (disabled ? " disabled" : "");
  if (disabled) el.title = disabled;
  el.innerHTML = options.map(([v, label]) => `<button data-v="${v}" class="${v === value ? "on" : ""}">${label}</button>`).join("");
  el.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b || disabled) return;
    el.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    onChange(b.dataset.v);
  });
  return el;
}

async function layerPage(model, layer, branch) {
  const base = baseOf(model);
  setCrumbs([[base.title, `/${base.name}`], [layer.name]]);
  document.title = `${layer.name} · ${model.title} · Microscope`;
  const [units, labels] = await Promise.all([getJSON(fill(model.paths.units, { layer: layer.name })), getLabels(model, layer.name)]);
  tools.append(lensSwitch(model, layer));

  const jump = document.createElement("label");
  jump.innerHTML = `Unit <input type="number" min="0" max="${layer.channels - 1}">`;
  jump.querySelector("input").addEventListener("keydown", e => {
    const v = Number(e.target.value);
    if (e.key === "Enter" && Number.isInteger(v) && v >= 0 && v < layer.channels) navigate(unitPath(model, layer.name, v));
  });
  tools.append(jump,
    segmented([["vis", "Feature vis"], ["ex", "Top example"], ["both", "Both"]], prefs.show, v => { prefs.show = v; savePrefs(); rerender(draw); }),
    segmented([["S", "S"], ["M", "M"], ["L", "L"]], prefs.size, v => { prefs.size = v; savePrefs(); rerender(draw); }));
  const hasLabels = Object.keys(labels).length > 0;
  tools.append(segmented([["off", "No labels"], ["on", "Labels"]], hasLabels ? prefs.labels : "off",
    v => { prefs.labels = v; savePrefs(); rerender(draw); }, hasLabels ? "" : "No labels for this view yet"));

  const groups = layer.branches
    ? layer.branches.map(b => ({ ...b, title: b.name === "pool" ? "Pool branch" : `${b.name.replace("x", "×")} branch`,
        note: b.name === "pool" ? "3×3 max pool, then 1×1 conv"
          : b.bottleneck ? `1×1 bottleneck to ${b.bottleneck}, then ${b.name.replace("x", "×")} conv` : "1×1 conv on the layer below" }))
    : [{ name: "all", start: 0, stop: layer.channels, title: "", note: "" }];

  function draw() {
    const tile = SIZES[prefs.size] * (prefs.show === "both" ? 2 : 1);
    const tileHtml = c => {
      const vis = `${visImg(model, layer.name, c)}`;
      const ex = units[c] ? exImg(model, layer.name, c, 0) : "";
      const body = prefs.show === "vis" ? vis : prefs.show === "ex" ? ex : vis + ex;
      const lab = labels[c];
      const title = lab ? ` title="${esc(lab.label)}: ${esc(lab.description)}"` : units[c] ? ` title="top example: ${esc(units[c][0].label)}"` : "";
      const cap = lab && prefs.labels === "on" ? `<span class="lab">${esc(lab.label)}</span>` : "";
      return `<a class="tile${prefs.show === "both" ? " pair" : ""}" href="${unitPath(model, layer.name, c)}" data-link${title}>${body}${cap}<span class="n">${c}</span></a>`;
    };
    const body = groups.map(g => {
      const tiles = [];
      for (let c = g.start; c < g.stop; c++) tiles.push(tileHtml(c));
      const head = g.title ? `<div class="branch-head" id="b-${g.name}"><b>${g.title}</b>
        <span>units ${g.start}–${g.stop - 1} · ${g.note}</span></div>` : "";
      return head + `<div class="grid" style="--tile:${tile}px">${tiles.join("")}</div>`;
    }).join("");
    mount(archColumn(model, { layer: baseLayerName(layer), branch }) + column("main", layer.name, `
      <h1>${layer.name}</h1>
      <p class="sub">${layer.channels} ${model.base ? "features" : "units"} · ${layer.spatial[0]}×${layer.spatial[1]} at 224px input ·
        receptive field ${layer.receptive_field}px · ${layer.rendered}/${layer.channels} feature vis,
        ${layer.examples}/${layer.channels} with dataset examples</p>${body}`));
  }
  draw();
  if (branch) document.getElementById(`b-${branch}`)?.scrollIntoView({ block: "start" });
}

// what the last shown layer feeds, by architecture kind
const headText = model => ({ residual: "CLIP's attention pooling, which produces the image embedding",
  sequential: "the fully connected classifier layers" })[model.architecture?.kind] || "average pooling and the 1008-way softmax";

function linkTiles(model, layerName, pairs, sign, scale, labels = {}) {
  if (!pairs?.length) return `<div class="links-empty">none</div>`;
  return `<div class="links">${pairs.map(([c, w]) => `
    <a class="link ${sign < 0 ? "neg" : ""}" href="${unitPath(model, layerName, c)}" data-link title="${layerName} unit ${c}: ${pct(w)}">
      <div class="img">${visImg(model, layerName, c)}<span class="n">${c}</span></div>
      <div class="w"><i style="width:${Math.max(4, Math.round(Math.abs(w) / scale * 48))}px"></i><span>${pct(w)}</span></div>
      ${labels[c] ? `<div class="lbl">${esc(labels[c].label)}</div>` : ""}</a>`).join("")}</div>`;
}

function circuitSide(model, title, note, layerName, data, labels, missing) {
  if (!data) return `<div class="side"><h3>${title}</h3><p class="links-empty">Not computed yet: <code>${missing}</code></p></div>`;
  const scale = Math.max(1e-6, ...[...data.pos, ...data.neg].map(([, w]) => Math.abs(w)));
  return `<div class="side"><h3>${title}</h3><p>${note}</p>
    <h3>Excites</h3>${linkTiles(model, layerName, data.pos, 1, scale, labels)}
    <h3>Inhibits</h3>${linkTiles(model, layerName, data.neg, -1, scale, labels)}</div>`;
}

// SAE features that lean on each channel, built once per SAE layer from its features.json
const saeUsersCache = {};
function saeUsers(sae, layerName) {
  return saeUsersCache[layerName] ??= getJSON(assetUrl(sae, layerName, "features.json")).then(feats => {
    const byChannel = {};
    for (const [f, v] of Object.entries(feats))
      for (const [c, w] of v.top_channels || []) if (w > 0.1) (byChannel[c] ??= []).push([Number(f), w]);
    for (const list of Object.values(byChannel)) list.sort((x, y) => y[1] - x[1]);
    return byChannel;
  });
}

async function unitPage(model, layer, unit) {
  const isSae = !!model.base;
  const base = baseOf(model);
  const kind = isSae ? "feature" : "unit";
  setCrumbs([[base.title, `/${base.name}`], [layer.name, layerPath(model, layer.name)], [`${isSae ? "Feature" : "Unit"} ${unit}`]]);
  document.title = `${layer.name}:${unit} · ${model.title} · Microscope`;
  tools.append(lensSwitch(model, layer));

  const i = model.layers.indexOf(layer);
  const prev = isSae ? null : model.layers[i - 1], next = isSae ? null : model.layers[i + 1];
  const saeLayer = !isSae && saeLayerOf(model, layer.name), sae = saeLayer && saeOf(model);
  const canAblate = !isSae && !!layer.circuits_ablation;
  const method = canAblate ? prefs.circuit : "gradient";
  const none = {};
  const [units, labels, features, chLabels, circuits, prevLabels, nextLabels, users, text] = await Promise.all([
    getJSON(fill(model.paths.units, { layer: layer.name })),
    getLabels(model, layer.name),
    isSae ? getJSON(assetUrl(model, layer.name, "features.json")) : none,
    isSae ? getLabels(base, baseLayerName(layer)) : none,
    isSae ? none : getJSON(assetUrl(model, layer.name, method === "ablation" ? "circuits_ablation.json" : "circuits.json")),
    prev ? getLabels(model, prev.name) : none,
    next ? getLabels(model, next.name) : none,
    saeLayer ? saeUsers(sae, layer.name) : none,
    layer.text && model.paths.text ? getJSON(fill(model.paths.text, { layer: layer.name })) : none,
  ]);
  const rows = units[unit] || [];
  const lab = labels[unit];
  const feat = features[unit];
  const urlBase = layerPath(model, layer.name);
  const prevU = unit > 0 ? unit - 1 : null, nextU = unit < layer.channels - 1 ? unit + 1 : null;
  const br = layer.branches?.find(b => unit >= b.start && unit < b.stop);

  const counts = {};
  rows.forEach(r => counts[r.label] = (counts[r.label] || 0) + 1);
  const classes = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 6)
    .map(([k, v]) => `${esc(k)}${v > 1 ? ` ×${v}` : ""}`).join(", ");
  const examples = rows.map((r, k) => `<div class="ex" title="${esc(r.label)} · activation ${fmt(r.act)} · val image ${r.img}">
      ${exImg(model, layer.name, unit, k, r.label)}
      <div class="cap"><span>${esc(r.label)}</span><span>${fmt(r.act)}</span></div></div>`).join("");

  // ---- circuit section: channels get gradient/ablation circuits, SAE features get their channel recipe
  let circuitHtml;
  if (isSae) {
    const baseLayer = baseLayerName(layer);
    const tops = feat?.top_channels || [];
    const scale = Math.max(1e-6, ...tops.map(([, w]) => Math.abs(w)));
    circuitHtml = `<h2>Made of channels <a class="help" href="/glossary#sae" data-link title="What is this?">?</a></h2>
      <p class="sub">The ${baseLayer} channels this feature's direction is built from (its SAE decoder weights). A feature that mixes several
        channels is a direction the network uses that no single channel captures.</p>
      ${tops.length ? linkTiles(base, baseLayer, tops.filter(([, w]) => w > 0), 1, scale, chLabels) : `<p class="links-empty">No decoder data.</p>`}
      ${tops.some(([, w]) => w < 0) ? `<h3>Negative weights</h3>${linkTiles(base, baseLayer, tops.filter(([, w]) => w < 0), -1, scale, chLabels)}` : ""}`;
  } else {
    const circ = circuits[unit] || {};
    const missing = method === "ablation" ? "python ablate.py" : "python circuits.py";
    const archNote = model.circuits_note ? ` <span class="muted">${esc(model.circuits_note)}</span>` : "";
    const how = method === "ablation"
      ? `Measured by switching off each candidate input on this unit's top ${rows.length || 16} dataset images and recording how much the
         unit's peak activation drops (positive) or rises (negative). This is a causal test of the gradient estimate.`
      : `Measured on this unit's top ${rows.length || 16} dataset images: activation × gradient at the position where the unit peaks.
         Percentages are each unit's share of the activation it feeds.${archNote}`;
    circuitHtml = `<div class="circuit-head"><h2>Circuit <a class="help" href="/glossary#circuit" data-link title="What is this?">?</a></h2><span id="circuit-method"></span></div>
      <p class="sub">${how}</p>
      <div class="circuit">
        ${prev ? circuitSide(model, `Built from ${prev.name}`, `Units in ${prev.name} that raise or lower this unit.`, prev.name, circ.in, prevLabels, missing)
               : `<div class="side"><h3>Built from pixels</h3><p class="muted">${layer.name} reads the image directly.</p></div>`}
        ${next ? circuitSide(model, `Feeds into ${next.name}`, `Units in ${next.name} this one helps build, or suppresses.`, next.name, circ.out, nextLabels, missing)
               : `<div class="side"><h3>Feeds the output</h3><p class="muted">${layer.name} is the last layer shown; it feeds ${headText(model)}.</p></div>`}
      </div>`;
  }

  let saeHtml = "";
  if (saeLayer) {
    const top = (users[unit] || []).slice(0, 12);
    saeHtml = `<h2>SAE features that use this channel <a class="help" href="/glossary#sae" data-link title="What is this?">?</a></h2>
      <p class="sub">Features from the sparse autoencoder on ${layer.name} whose direction leans on unit ${unit}, by decoder weight.
        A weight near 100% means the feature is essentially this channel; several mid-weight features mean the channel takes part in
        several distinct directions (often a sign it responds to unrelated things).</p>
      ${top.length ? linkTiles(sae, layer.name, top, 1, top[0][1])
                     : `<p class="links-empty">No feature puts much weight on this channel.</p>`}`;
  }

  let textHtml = "";
  const words = text[unit] || [];
  if (words.length) {
      const max = words[0][1] || 1;
      textHtml = `<h2>Words that drive this unit <a class="help" href="/glossary#text-view" data-link title="What is this?">?</a></h2>
        <p class="sub">About 5,000 English words were rendered as black-on-white text images and shown to CLIP. These made this unit fire most:
          the same unit often responds to a concept in photos and in writing.</p>
        <div class="words">${words.map(([w, a]) => `<span style="--w:${(a / max).toFixed(2)}">${esc(w)}<em>${fmt(a)}</em></span>`).join("")}</div>`;
  }

  const facts = [
    lab && [`Auto label <a class="help" href="/glossary#auto-label" data-link title="What is this?">?</a>`, `${esc(lab.label)} <span class="muted">· ${esc(lab.confidence)} confidence${lab.polysemantic ? " · responds to unrelated things" : ""}</span>`],
    !br && layer.group && ["Stage", `${esc(layer.group)}${layer.note ? ` · ${esc(layer.note)}` : ""}`],
    layer.pre_relu_units?.includes(unit) && ["Feature vis", `rendered before the final ReLU: this unit was off on the random starting image,
      so the usual objective gave no signal`],
    br && ["Branch", `${br.name === "pool" ? "pool → 1×1" : br.name.replace("x", "×")}${br.bottleneck ? ` (via ${br.bottleneck}-unit bottleneck)` : ""}, units ${br.start}–${br.stop - 1}`],
    feat && ["Density", `active at ${feat.density >= 0.001 ? pct(feat.density) : (feat.density * 100).toPrecision(2) + "%"} of positions`],
    ["Max activation", rows[0] ? fmt(rows[0].act) : "–"],
    ["Top classes", classes || "–"],
    ["Crop", `${layer.crop_size}px of 256px image, centered on the spatial argmax`],
  ].filter(Boolean).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  mount(archColumn(model, { layer: baseLayerName(layer), branch: br?.name }) + unitsColumn(model, layer, unit) + column("main", `${isSae ? "Feature" : "Unit"} ${unit}`, `
    <div class="nav-btns">
      ${prevU !== null ? `<a href="${urlBase}/${prevU}" data-link>← ${prevU}</a>` : ""}
      ${nextU !== null ? `<a href="${urlBase}/${nextU}" data-link>${nextU} →</a>` : ""}
    </div>
    <h1>${layer.name} ${kind} ${unit}${lab ? ` <span class="unit-label">${esc(lab.label)}</span>` : ""}</h1>
    ${lab ? `<p class="sub">${esc(lab.description)}</p>` : ""}
    <div class="unit-top">
      <section class="vis">
        <h3>Feature visualization · ${isSae ? "SAE feature objective" : "channel objective"}</h3>
        <img src="${visUrl(model, layer.name, unit)}" alt="feature visualization"
          onerror="this.replaceWith(Object.assign(document.createElement('p'), {className: 'muted', textContent: 'Not rendered yet.'}))">
        <table class="facts">${facts}</table>
      </section>
      <section>
        <h3>Dataset examples · top ${rows.length} of ${esc(model.dataset?.split(",")[0] || "ImageNet")}</h3>
        <div class="examples">${examples || '<p class="muted">No dataset examples yet.</p>'}</div>
      </section>
    </div>
    ${textHtml}${circuitHtml}${saeHtml}`));

  const slot = document.getElementById("circuit-method");
  if (slot) slot.append(segmented([["gradient", "Gradient estimate"], ["ablation", "Ablation test"]], method,
    v => { prefs.circuit = v; savePrefs(); rerender(route); }, canAblate ? "" : "Ablation test not computed for this layer"));
  document.getElementById(`u${unit}`)?.scrollIntoView({ block: "nearest" });
  keyHandler = e => {
    if (e.key === "ArrowLeft" && prevU !== null) navigate(`${urlBase}/${prevU}`);
    if (e.key === "ArrowRight" && nextU !== null) navigate(`${urlBase}/${nextU}`);
    if (e.key === "Escape") navigate(urlBase);
  };
}

// ---------- glossary ----------

// glossary figures use InceptionV1 (and its SAE lens) through the same URL builders as the viewer
const gModel = sae => { const base = manifest.models.find(m => m.name === "inceptionv1"); return sae ? saeOf(base) : base; };
const gVis = (layer, u, sae) => visUrl(gModel(sae), layer, u);
const gExImg = (layer, u, r, sae) => exImg(gModel(sae), layer, u, r);
const gHref = (layer, u, sae) => unitPath(gModel(sae), layer, u);
const gUnits = (layer, sae) => getJSON(fill(gModel(sae).paths.units, { layer }));
const gLabels = layer => getLabels(gModel(), layer);
const gJSON = (layer, file, sae) => getJSON(assetUrl(gModel(sae), layer, file));

// one unit: big feature vis, then a few real examples with their ImageNet class
async function figUnitWithExamples(layer, u, { n = 6, sae = false, caption = "", classes = false } = {}) {
  const units = await gUnits(layer, sae);
  const rows = (units[u] || []).slice(0, n);
  return `<figure class="gfig">
    <div class="g-unit">
      <a href="${gHref(layer, u, sae)}" data-link class="g-vis"><img loading="lazy" src="${gVis(layer, u, sae)}" alt=""><span class="n">${layer}${sae ? " feature" : ""} ${u}</span></a>
      <div class="g-exs">${rows.map((r, k) => `<div>${gExImg(layer, u, k, sae)}${classes ? `<span>${esc(r.label)}</span>` : ""}</div>`).join("")}</div>
    </div>
    ${caption ? `<figcaption>${caption}</figcaption>` : ""}</figure>`;
}

async function figLayers() {
  const picks = [["conv2d0", 3], ["conv2d1", 3], ["conv2d2", 8], ["mixed3a", 8], ["mixed3b", 379], ["mixed4a", 50],
                 ["mixed4b", 373], ["mixed4c", 447], ["mixed4d", 28], ["mixed4e", 8], ["mixed5a", 6], ["mixed5b", 24]];
  const labels = await Promise.all(picks.map(([l]) => gLabels(l)));
  return `<figure class="gfig"><div class="g-ladder">${picks.map(([l, u], i) => `
      <a href="${gHref(l, u)}" data-link><img loading="lazy" src="${gVis(l, u)}" alt="">${gExImg(l, u, 0)}
        <b>${l}</b><span>${esc(labels[i][u]?.label || "")}</span></a>`).join("")}</div>
    <figcaption>One unit per layer, input on the left, output on the right: feature visualization above, its top real example below.
      Edges and colors become curves and rims, then wheels and car bodies, then dog faces.</figcaption></figure>`;
}

function figBranch(layer) {
  const b = Object.fromEntries(layer.branches.map(x => [x.name, x]));
  const w = n => b[n].stop - b[n].start;
  const box = (x, y, wd, label, cls = "") => `<g class="${cls}"><rect x="${x}" y="${y}" width="${wd}" height="26" rx="3"/><text x="${x + wd / 2}" y="${y + 17}">${label}</text></g>`;
  const line = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
  const xs = [20, 120, 220, 320], cx = xs.map(x => x + 40);
  return `<figure class="gfig"><svg class="g-svg small" viewBox="0 0 420 230" role="img" aria-label="Inception block diagram">
      ${cx.map(x => line(x, 196, x, 26) + line(x, 26, 210, 20)).join("")}${cx.map(x => line(210, 210, x, 196)).join("")}
      ${box(110, 4, 200, `${layer.name} · ${layer.channels} units`, "out")}
      ${box(xs[0], 60, 80, `1×1 · ${w("1x1")}`)}${box(xs[1], 60, 80, `3×3 · ${w("3x3")}`)}${box(xs[2], 60, 80, `5×5 · ${w("5x5")}`)}${box(xs[3], 60, 80, `1×1 · ${w("pool")}`)}
      ${box(xs[1], 128, 80, `1×1 · ${b["3x3"].bottleneck}`, "neck")}${box(xs[2], 128, 80, `1×1 · ${b["5x5"].bottleneck}`, "neck")}${box(xs[3], 128, 80, "max pool", "neck pool")}
      ${box(110, 200, 200, "layer below", "in")}
    </svg><figcaption>${layer.name}'s four branches, read bottom to top. The 1×1 boxes in the middle row are bottlenecks that shrink the
      input before the larger 3×3 and 5×5 convolutions. The outputs are stacked: ${w("1x1")} + ${w("3x3")} + ${w("5x5")} + ${w("pool")} = ${layer.channels}.</figcaption></figure>`;
}

async function figCircuit() {
  const [c, l3, l4a, l4b] = await Promise.all([gJSON("mixed4a", "circuits.json"),
    gLabels("mixed3b"), gLabels("mixed4a"), gLabels("mixed4b")]);
  const u = 50, circ = c[u];
  const ins = [...circ.in.pos.slice(0, 3).map(x => [...x, 1]), [...circ.in.neg[0], -1]];
  const outs = circ.out.pos.slice(0, 3).map(x => [...x, 1]);
  // columns: [thumb + label] -> curves -> center -> curves -> [thumb + label]
  const W = 820, S = 56, rowY = i => 26 + i * 72, cy = 26 + 1.5 * 72 + S / 2;
  const inX = 0, inEnd = 210, cx = 340, C = 104, outX = 600;
  const maxW = Math.max(...[...ins, ...outs].map(([, w]) => Math.abs(w)));
  const short = t => esc((t || "").length > 22 ? t.slice(0, 21) + "…" : t || "");
  const thumb = (x, y, layer, unit, label) => `<a href="${gHref(layer, unit)}" data-link><image href="${gVis(layer, unit)}" x="${x}" y="${y}" width="${S}" height="${S}"/>
      <rect class="frame" x="${x}" y="${y}" width="${S}" height="${S}"/></a>
      <text class="cap strong" x="${x + S + 8}" y="${y + 22}">${layer.replace("mixed", "")}:${unit}</text>
      <text class="cap" x="${x + S + 8}" y="${y + 37}">${short(label)}</text>`;
  // percentage sits at the outer (unit) end of each line so labels never stack at the center
  const edge = (x1, y1, x2, y2, w, sign, atEnd) => `<path class="edge ${sign < 0 ? "neg" : ""}" d="M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}"
      stroke-width="${1.5 + 9 * Math.abs(w) / maxW}"/><text class="pct" x="${atEnd ? x2 - 30 : x1 + 4}" y="${(atEnd ? y2 : y1) - 7}">${(w * 100).toFixed(0)}%</text>`;
  return `<figure class="gfig"><svg class="g-svg circuit" viewBox="0 0 ${W} 330" role="img" aria-label="Circuit for mixed4a unit 50">
      <text class="col" x="${inX}" y="12">built from mixed3b</text><text class="col" x="${cx}" y="12">mixed4a unit 50</text><text class="col" x="${outX}" y="12">feeds into mixed4b</text>
      ${ins.map(([, w, sg], i) => edge(inEnd, rowY(i) + S / 2, cx, cy, w, sg)).join("")}
      ${outs.map(([, w], i) => edge(cx + C, cy, outX, rowY(i) + S / 2, w, 1, true)).join("")}
      ${ins.map(([iu], i) => thumb(inX, rowY(i), "mixed3b", iu, l3[iu]?.label)).join("")}
      <a href="${gHref("mixed4a", u)}" data-link><image href="${gVis("mixed4a", u)}" x="${cx}" y="${cy - C / 2}" width="${C}" height="${C}"/>
        <rect class="frame cur" x="${cx}" y="${cy - C / 2}" width="${C}" height="${C}"/></a>
      <text class="cap" x="${cx}" y="${cy + C / 2 + 14}">${short(l4a[u]?.label)}</text>
      ${outs.map(([ou], i) => thumb(outX, rowY(i), "mixed4b", ou, l4b[ou]?.label)).join("")}
    </svg><figcaption>The real circuit for <a href="/inceptionv1/mixed4a/50" data-link>mixed4a unit 50</a>. Line thickness is the share of
      the receiving unit's activation. Green excites, red inhibits: three curve detectors build this rim detector, a wavy-line detector
      suppresses it, and it in turn feeds rounder, more object-like units in mixed4b.</figcaption></figure>`;
}

async function figGradAbl() {
  const [g, a] = await Promise.all([gJSON("mixed4a", "circuits.json"), gJSON("mixed4a", "circuits_ablation.json")]);
  const gi = Object.fromEntries([...g[50].in.pos, ...g[50].in.neg]);
  const ai = Object.fromEntries([...(a[50]?.in.pos || []), ...(a[50]?.in.neg || [])]);
  const rows = [...g[50].in.pos.slice(0, 5), ...g[50].in.neg.slice(0, 1)].map(([c]) => [c, gi[c], ai[c]]);
  const max = Math.max(...rows.map(([, x, y]) => Math.max(Math.abs(x), Math.abs(y ?? 0))));
  const bar = (v, cls) => v == null ? `<i class="none">n/a</i>` : `<i class="${cls} ${v < 0 ? "neg" : ""}" style="width:${Math.abs(v) / max * 100}%"></i><em>${(v * 100).toFixed(1)}%</em>`;
  return `<figure class="gfig"><div class="g-pairs">${rows.map(([c, gv, av]) => `
      <a href="${gHref("mixed3b", c)}" data-link><img loading="lazy" src="${gVis("mixed3b", c)}" alt=""></a>
      <div><div class="pbar">${bar(gv, "grad")}</div><div class="pbar">${bar(av, "abl")}</div></div>`).join("")}</div>
    <figcaption>Inputs to mixed4a unit 50. <span class="key grad"></span> gradient estimate, <span class="key abl"></span> measured by
      switching the input off. The pairs match almost exactly.</figcaption></figure>`;
}

function figSaeDiagram() {
  const ch = Array.from({ length: 40 }, (_, i) => 0.25 + 0.75 * Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.37)));
  const lit = new Set([7, 19, 23, 41, 58, 66, 90, 101]);
  const feats = Array.from({ length: 112 }, (_, i) => i);
  return `<figure class="gfig"><svg class="g-svg small" viewBox="0 0 420 220" role="img" aria-label="Sparse autoencoder diagram">
      <text class="col" x="10" y="12">508 channels</text>
      ${ch.map((v, i) => `<rect class="chbar" x="${10 + i * 2.6}" y="${120 - v * 90}" width="1.8" height="${v * 90}"/>`).join("")}
      <text class="cap" x="10" y="138">many active at once</text>
      <path class="arrow" d="M120,80 L160,80"/><text class="cap" x="122" y="72">encode</text>
      <text class="col" x="170" y="12">~4,000 features</text>
      ${feats.map(i => `<rect class="feat ${lit.has(i) ? "on" : ""}" x="${170 + (i % 16) * 15}" y="${24 + Math.floor(i / 16) * 15}" width="12" height="12" rx="1"/>`).join("")}
      <text class="cap" x="170" y="138">only a few light up (32 of ~4,000)</text>
      <path class="arrow" d="M290,150 L290,175"/><text class="cap" x="298" y="168">decode</text>
      <text class="cap" x="170" y="196">each lit feature adds back its own mix</text>
      <text class="cap" x="170" y="210">of channels, rebuilding most of the layer</text>
    </svg><figcaption>The autoencoder rewrites each position's channel values using a handful of features from a much larger set.
      Forcing it to use so few makes each feature stand for one thing.</figcaption></figure>`;
}

async function figSaeSplit() {
  const ch = 2, picks = [763, 1103];
  const [feats, l4a] = await Promise.all([gJSON("mixed4a", "features.json", true), gLabels("mixed4a")]);
  const w = f => (feats[f]?.top_channels || []).find(([c]) => c === ch)?.[1] || 0;
  const crops = (f, sae, n) => Array.from({ length: n }, (_, k) => gExImg("mixed4a", f, k, sae)).join("");
  return `<figure class="gfig"><div class="g-split">
      <a class="from" href="${gHref("mixed4a", ch)}" data-link><img class="v" loading="lazy" src="${gVis("mixed4a", ch)}" alt="">
        <div class="crops">${crops(ch, false, 4)}</div><b>channel ${ch}</b><span>${esc(l4a[ch]?.label || "")}</span></a>
      <div class="arrowcol">→</div>
      ${picks.map(f => `<a class="to" href="${gHref("mixed4a", f, true)}" data-link><img class="v" loading="lazy" src="${gVis("mixed4a", f, true)}" alt="">
        <div class="crops">${crops(f, true, 4)}</div><b>feature ${f}</b><span>uses channel ${ch} at ${(w(f) * 100).toFixed(0)}%</span></a>`).join("")}
    </div><figcaption>Channel 2 fires on clusters of round dots wherever they appear: its top crops come from unrelated photos.
      Two SAE features that lean on it each keep to one use: spotted big-cat coats and keyboard keys.</figcaption></figure>`;
}

async function figMadeOf() {
  const feats = await gJSON("mixed4a", "features.json", true);
  const f = 1289, tops = feats[f].top_channels.filter(([, w]) => w > 0).slice(0, 4);
  return `<figure class="gfig"><div class="g-eq">${tops.map(([c, w], i) => `${i ? "<i>+</i>" : ""}<a href="${gHref("mixed4a", c)}" data-link>
      <img loading="lazy" src="${gVis("mixed4a", c)}" alt=""><span>ch ${c} × ${(w * 100).toFixed(0)}%</span></a>`).join("")}<i>+ …</i><i>=</i>
      <a href="${gHref("mixed4a", f, true)}" data-link class="res"><img loading="lazy" src="${gVis("mixed4a", f, true)}" alt=""><span>feature ${f}</span></a></div>
    <figcaption>Feature 1289 (speedometer dials) as a weighted mix of channels. No single channel is a speedometer detector.</figcaption></figure>`;
}

const GLOSSARY = [
  ["The basics", [
    ["unit", "Unit (channel)", `One detector inside a layer. A channel looks at every position of the image and reports how strongly its
      pattern is present there. "mixed3b unit 379" means channel 379 of layer mixed3b, a curve detector.`,
      () => figUnitWithExamples("mixed3b", 379, { caption: "mixed3b unit 379: its feature visualization and real crops where it fires most." })],
    ["layer", "Layer", `A stage of the network. The image enters at conv2d0, which sees raw pixels, and exits at mixed5b, just before the
      classifier. Early layers detect edges and colors, middle layers curves, patterns and parts, late layers whole objects.`, figLayers],
    ["branch", "Branch (inception block)", `Each "mixed" layer is an inception block: four small paths run side by side over the layer below
      and their outputs are stacked into one layer. The boxes in the architecture strip are these branches; the numbers are how many
      units each contributes. A unit's branch tells you how large a neighborhood it combines.`,
      () => figBranch(gModel().layers.find(l => l.name === "mixed4a"))],
    ["activation", "Activation", `How strongly a unit fires on an image at a position. Higher means more of its pattern. Values are only
      comparable within one view: channels and SAE features use different scales.`],
    ["receptive-field", "Receptive field", `The patch of the input image a unit at one position can see. It grows with depth, from a few
      pixels in conv2d0 to most of the image by mixed5b. Dataset-example crops are sized from it.`],
  ]],
  ["Ways of seeing a unit", [
    ["feature-vis", "Feature visualization", `A synthetic image optimized to make the unit fire as hard as possible, starting from noise.
      It shows what the unit looks for, usually tiled across the image because every position counts. Read it with the dataset
      examples: the visualization can exaggerate, the examples keep it honest.`,
      () => figUnitWithExamples("mixed4a", 485, { n: 8, caption: "mixed4a unit 485: radiating spokes in the visualization; wheels, thatch and twig piles in real photos." })],
    ["dataset-examples", "Dataset examples", `The 16 crops, from 50,000 ImageNet photos, where the unit fired most, each centered on the spot
      where it fired. The class name under each is the photo's ImageNet label. It is only a hint: units usually respond to a part
      or texture in the crop, not the labeled object. The crops are cropped and downsized from the ImageNet validation set and are
      shown for non-commercial research and education under the ImageNet terms of access.`],
    ["auto-label", "Auto label", `A short description written by a vision model after looking at the feature visualization and dataset
      examples. Labels make units searchable. They are hints, not ground truth; "confidence" is the model's own estimate.`],
    ["text-view", "Words that drive this unit (CLIP)", `CLIP was trained on images paired with captions, so many of its units
      respond to a concept whether it appears as a photo or as written words. To show this, about 5,000 common words were rendered
      as plain text images and shown to the network; each high-level unit lists the words that made it fire most. When those words
      match its photos, as with a weather unit firing on "rain" and on rainy streets, it is a multimodal unit.`],
    ["polysemantic", "Polysemantic / monosemantic", `A monosemantic unit means one thing, like curves of one orientation. A polysemantic
      unit responds to unrelated things. Networks pack more concepts than they have channels, so many channels end up mixed.
      SAE features are one way to untangle them.`,
      () => figUnitWithExamples("mixed4c", 41, { n: 8, classes: true, caption: "mixed4c unit 41 fires on both car bodies and dog muzzles." })],
  ]],
  ["Circuits: how units are built from units", [
    ["circuit", "Circuit", `A unit plus the connections that build it. Every unit is computed from the units of the layer directly below,
      so a wheel detector is a recipe of lower-level ingredients. The circuit panel on each unit page shows the recipe both ways:
      <b>Built from</b> lists units below that raise (excite) or lower (inhibit) this one; <b>Feeds into</b> lists units above that
      this one helps build. Click through them to walk from edges to objects. The idea comes from the Distill Circuits thread (2020).`,
      figCircuit],
    ["gradient", "Gradient estimate", `The default way the circuit numbers are measured. On the unit's 16 top photos, at the spot where it
      fires hardest, each lower unit's contribution is its activation times how sensitive this unit is to it. The percentage is its
      share of this unit's activation, so shares sum to about 100%.`],
    ["ablation", "Ablation test", `A direct check of the gradient estimate: switch a lower unit off on the same photos, re-run the network,
      and measure how much this unit actually drops. In InceptionV1 the two agree almost exactly, because between neighboring layers
      the network is weighted sums plus on/off switches. At its peak a unit responds in exact proportion to its inputs.`, figGradAbl],
    ["excite-inhibit", "Excites / inhibits", `Excites: turning that unit up turns this one up. Inhibits: it pushes this one down, often
      meaning "fire for X, but not when Y is present", like a curve detector suppressed by wavy lines.`],
  ]],
  ["SAE features: splitting channels into cleaner parts", [
    ["sae", "SAE features (sparse autoencoder)", `A second way to divide up a layer. A sparse autoencoder is a small model trained to rewrite
      each position's channel values as a combination of a few features from a set about 8× larger, with only 32 active at a time.
      Because so few can be active, each feature tends to capture one clean idea, including ideas the network spreads across several
      channels. Switch between <b>Channels</b> and <b>SAE features</b> at the top right, on any layer tagged <b>SAE</b> in the architecture strip. SAEs come from
      Anthropic's work on language models (2023–24); Gorton (2024) applied them to InceptionV1.`, figSaeDiagram],
    ["sae-split", "Splitting a mixed channel", `When a channel is polysemantic, several SAE features usually lean on it, each picking out
      one of its meanings. On a channel page, "SAE features that use this channel" lists them.`, figSaeSplit],
    ["made-of", "Made of channels", `On an SAE feature page: the channels whose mix defines the feature (its decoder weights). One channel
      near 100% means the feature is essentially that channel. Several mid-sized weights mean a new direction no single channel captures.`,
      figMadeOf],
    ["density", "Density", `The share of all image positions where an SAE feature is active. Most features are rare, well under 1%,
      which is part of why they are cleaner.`],
    ["reconstruction", "How faithful are SAE features?", `They reconstruct about 88% (mixed3b) and 84% (mixed4a) of the layer's activity
      on held-out images: a good but not perfect account of the layer. Some features latch onto dataset quirks, such as feature 2320,
      which fires on a photographer's watermark that appears on canoe photos.`,
      () => figUnitWithExamples("mixed4a", 2320, { sae: true, n: 4, caption: "SAE feature 2320: the shared watermark, not the canoe." })],
  ]],
];

async function glossaryPage(model) {
  setCrumbs([[model.title, `/${model.name}`], ["Glossary"]]);
  document.title = "Glossary · Microscope";
  const sections = GLOSSARY.map(([section, items]) => `<h2>${section}</h2>${items.map(([id, term, text, fig]) => `
    <section class="gitem ${fig ? "has-fig" : ""}" id="${id}">
      <div class="gtext"><h3>${term}</h3><p>${text}</p></div>
      ${fig ? `<div class="gfig-slot" data-fig="${id}"></div>` : ""}
    </section>`).join("")}`).join("");
  mount(archColumn(model, {}) + column("main", "Glossary", `<h1>Glossary</h1>
    <p class="sub">What each part of this viewer shows and how it was computed. Figures use this network's real units; click any image to open it.</p>
    <div class="gloss-body">${sections}</div>`));
  // figures load after the text so the page is readable immediately
  await Promise.all(GLOSSARY.flatMap(([, items]) => items).filter(x => x[3]).map(async ([id, , , fn]) => {
    const slot = document.querySelector(`[data-fig="${id}"]`);
    try { if (slot) { slot.innerHTML = await fn(); lazySprites(slot); } } catch (e) { if (slot) slot.innerHTML = ""; console.warn(id, e); }
  }));
  const id = location.hash.slice(1);
  if (id) { const el = document.getElementById(id); el?.scrollIntoView({ block: "start" }); el?.classList.add("hit"); }
}

route();
