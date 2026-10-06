// compare extension: registers on window.MS (see the extension-points comment at the top of app.js)
// /compare: a picker; /compare?a=<model>/<layer>/<unit> lists that unit's closest matches in every network;
// /compare?a=…&b=… shows two units side by side.
// Data: site/assets/<model>/<layer>/similar/<unit // 256>.json from similar.py, {unit: {model: [[layer, unit, r, rare?], ...]}}: the top 8 units in
// every other model and the top 5 in the unit's own model's other layers, by Pearson r of per-image spatial-max activation.
window.MS ||= {};
(() => {
  const ms = window.MS;
  ms.routes ||= {};
  const N_IMAGES = "8,000";  // similar.py collect --n
  const SHARD = 256;  // similar.py SHARD
  const simUrl = (model, layer, unit) => assetUrl(model, layer, `similar/${Math.floor(unit / SHARD)}.json`);
  const findModel = name => manifest.models.find(m => m.name === name);
  const ref = (model, layer, unit) => `${model.name}/${layer}/${unit}`;
  const compareHref = (a, b) => `/compare?a=${a}&b=${b}`;
  const r3 = r => r.toFixed(2);

  // own network's other layers last; the SAE lens and its channels count as one network
  const family = m => m.base || m.name;
  function orderKeys(model, entry) {
    const keys = Object.keys(entry).filter(k => findModel(k) && entry[k].length);
    const rank = k => k === model.name ? 3 : family(findModel(k)) === family(model) ? 2 : 1;
    return keys.sort((x, y) => rank(x) - rank(y));
  }
  function groupTitle(model, key) {
    const m = findModel(key);
    if (key === model.name) return `${model.base ? "Other SAE layers" : "Other layers"} of ${baseOf(model).title}`;
    if (m.base) return `${baseOf(m).title} SAE features`;
    return family(m) === family(model) ? `${m.title} channels` : m.title;
  }

  // labels for every (model, layer) the matches touch, fetched in parallel
  async function labelsFor(entry) {
    const want = {};
    for (const [k, lst] of Object.entries(entry)) for (const [l] of lst) if (findModel(k)) want[`${k}\t${l}`] = [findModel(k), l];
    const got = await Promise.all(Object.values(want).map(([m, l]) => getLabels(m, l)));
    const out = {};
    Object.keys(want).forEach((k, i) => out[k] = got[i]);
    return (key, layer, unit) => out[`${key}\t${layer}`]?.[unit];
  }

  function simTiles(src, srcRef, key, lst, label) {
    const m = findModel(key);
    return `<div class="links sim-links">${lst.map(([l, u, r, rare]) => {
      const lab = label(key, l, u);
      const note = rare ? " · rare: active on under 1% of the images, so r is noisy" : "";
      return `<div class="link sim${rare ? " rare" : ""}" title="${esc(`${m.title} ${l} ${u}${lab ? `: ${lab.label}` : ""} · r = ${r3(r)}${note}`)}">
        <a class="img" href="${unitPath(m, l, u)}" data-link>${visImg(m, l, u)}<span class="n">${esc(l)} ${u}</span></a>
        <div class="w"><i style="width:${Math.max(4, Math.round(Math.max(0, r) * 48))}px"></i><span>r ${r3(r)}</span></div>
        <div class="lbl">${rare ? `<em>rare (noisy)</em> ` : ""}${lab ? esc(lab.label) : rare ? "" : "&nbsp;"}</div>
        <a class="cmp" href="${compareHref(srcRef, ref(m, l, u))}" data-link>compare</a></div>`;
    }).join("")}</div>`;
  }

  // ---------- /compare

  function parseRef(s) {
    const [name, layerName, u] = (s || "").split("/");
    const model = findModel(name);
    const layer = model?.layers.find(l => l.name === layerName);
    const unit = Number(u);
    if (!model) return { error: `No model "${name || ""}".` };
    if (!layer) return { error: `No layer "${layerName || ""}" in ${model.title}.` };
    if (!Number.isInteger(unit) || unit < 0 || unit >= layer.channels) return { error: `${layer.name} has units 0–${layer.channels - 1}.` };
    return { model, layer, unit, ref: ref(model, layer.name, unit) };
  }
  const titleOf = s => `${s.model.base ? baseOf(s.model).title + " SAE" : s.model.title} · ${s.layer.name} ${s.model.base ? "feature" : "unit"} ${s.unit}`;
  const topClasses = rows => {
    const counts = {};
    rows.forEach(r => counts[r.label] = (counts[r.label] || 0) + 1);
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  };

  async function side(s, other, sim, label) {
    const { model, layer, unit } = s;
    const [units, labels] = await Promise.all([getJSON(fill(model.paths.units, { layer: layer.name })), getLabels(model, layer.name)]);
    const rows = units[unit] || [], lab = labels[unit];
    const otherClasses = new Set(other.rows.map(r => r.label));
    const classes = topClasses(rows).slice(0, 8).map(([k, v]) =>
      `<span class="${otherClasses.has(k) ? "shared" : ""}">${esc(k)}${v > 1 ? ` ×${v}` : ""}</span>`).join(", ");
    const examples = rows.map((r, k) => {
      const href = ms.exampleHref?.(model, layer.name, unit, k, r);
      const tag = href ? `a href="${href}" data-link` : "div";
      return `<${tag} class="ex" title="${esc(r.label)} · activation ${fmt(r.act)} · val image ${r.img}">
        ${exImg(model, layer.name, unit, k, r.label)}<div class="cap"><span>${esc(r.label)}</span><span>${fmt(r.act)}</span></div></${href ? "a" : "div"}>`;
    }).join("");
    // this unit's other close matches in the other side's model, to switch the comparison
    const alts = (sim[unit]?.[other.model.name] || []).filter(([l, u]) => !(l === other.layer.name && u === other.unit));
    const altHtml = alts.length ? `<h3>Other matches in ${esc(groupTitle(model, other.model.name))}</h3>
      <div class="cmp-alts">${alts.map(([l, u, r, rare]) => `<a href="${compareHref(s.ref, ref(other.model, l, u))}" data-link title="${esc((label(other.model.name, l, u)?.label || "") + (rare ? " · rare (noisy r)" : ""))}">
        ${visImg(other.model, l, u)}<span class="n">${esc(l)} ${u} · ${r3(r)}${rare ? " · rare" : ""}</span></a>`).join("")}</div>` : "";
    return `<section class="cmp-side">
      <h2><a href="${unitPath(model, layer.name, unit)}" data-link>${esc(titleOf(s))}</a></h2>
      <p class="sub">${lab ? `<b>${esc(lab.label)}</b> · ${esc(lab.description)}` : "No label."}</p>
      <div class="cmp-vis">${visImg(model, layer.name, unit)}</div>
      <table class="facts"><tr><td>Max activation</td><td>${rows[0] ? fmt(rows[0].act) : "–"}</td></tr>
        <tr><td>Top classes</td><td class="cmp-classes">${classes || "–"}</td></tr></table>
      <h3>Dataset examples · top ${rows.length}</h3>
      <div class="examples cmp-ex">${examples || '<p class="muted">No dataset examples yet.</p>'}</div>
      ${altHtml}</section>`;
  }

  // ---- picker: network (channels or SAE lens), layer, unit for each side
  const PAIRS = [
    ["inceptionv1/mixed3b/379", "vgg19/conv4_2/306", "a curve detector and its closest VGG19 unit"],
    ["inceptionv1/mixed4a/50", "vgg19/conv4_1/473", "round-rim detectors in two networks"],
    ["clip_rn50/layer4_1/1933", "inceptionv1/mixed4b/429", "dog faces in CLIP and InceptionV1"],
    ["clip_rn50/layer4_1/1996", "inceptionv1/mixed4e/715", "faces in CLIP and InceptionV1"],
    ["inceptionv1/mixed4a/2", "inceptionv1_sae/mixed4a/763", "a mixed channel and an SAE feature that uses it"],
  ];
  const lensTitle = m => m.base ? `${baseOf(m).title} · SAE features` : m.title;
  function picker(id, init) {
    const opts = manifest.models.map(m => `<option value="${m.name}"${m.name === init?.model.name ? " selected" : ""}>${esc(lensTitle(m))}</option>`).join("");
    return `<fieldset class="cmp-pick" data-side="${id}"><legend>Unit ${id.toUpperCase()}</legend>
      <label>Network <select name="model">${opts}</select></label>
      <label>Layer <select name="layer"></select></label>
      <label>Unit <input name="unit" type="number" min="0" value="${init ? init.unit : 0}"></label></fieldset>`;
  }
  function wirePicker(el, init) {
    const [mSel, lSel, uIn] = ["model", "layer", "unit"].map(n => el.querySelector(`[name=${n}]`));
    const fillLayers = layerName => {
      const m = findModel(mSel.value);
      lSel.innerHTML = m.layers.map(l => `<option value="${l.name}"${l.name === layerName ? " selected" : ""}>${l.name} (${l.channels})</option>`).join("");
      uIn.max = m.layers.find(l => l.name === lSel.value).channels - 1;
    };
    mSel.addEventListener("change", () => fillLayers());
    lSel.addEventListener("change", () => { uIn.max = findModel(mSel.value).layers.find(l => l.name === lSel.value).channels - 1; });
    fillLayers(init?.layer.name);
    return () => `${mSel.value}/${lSel.value}/${Math.min(Math.max(0, Number(uIn.value) || 0), Number(uIn.max))}`;
  }

  async function landing(a) {
    setCrumbs([["Compare"]]);
    document.title = "Compare · Microscope";
    mount(column("main", "Compare", `<h1>Compare units</h1>
      <p class="sub">Put any two units side by side: feature visualizations, top dataset examples, labels and how closely they track each other
        across ${N_IMAGES} ImageNet validation images. Pick one unit and press <b>Find matches</b> to see its closest units in every network.</p>
      <div class="cmp-form">${picker("a", a)}${picker("b")}
        <div class="cmp-actions"><button data-go="pair">Compare A and B</button><button data-go="matches">Find matches for A</button></div></div>
      <h3>Try these</h3>
      <div class="cmp-pairs">${PAIRS.map(([x, y, why]) => `<a href="${compareHref(x, y)}" data-link><b>${esc(why)}</b><span>${esc(x)} vs ${esc(y)}</span></a>`).join("")}</div>`));
    const getA = wirePicker(document.querySelector('[data-side="a"]'), a), getB = wirePicker(document.querySelector('[data-side="b"]'));
    document.querySelector(".cmp-actions").addEventListener("click", e => {
      const go = e.target.closest("button")?.dataset.go;
      if (go === "pair") navigate(compareHref(getA(), getB()));
      if (go === "matches") navigate(`/compare?a=${getA()}`);
    });
  }

  async function matches(a) {
    setCrumbs([["Compare", "/compare"], [`Matches for ${a.ref}`]]);
    document.title = `Matches for ${a.layer.name}:${a.unit} · Compare · Microscope`;
    const sim = await getJSON(simUrl(a.model, a.layer.name, a.unit));
    const entry = sim[a.unit] || {};
    const label = await labelsFor(entry);
    const groups = orderKeys(a.model, entry).map(k => `<h3>${esc(groupTitle(a.model, k))}</h3>${simTiles(a.model, a.ref, k, entry[k], label)}`).join("");
    mount(column("main", "Compare", `<h1>Closest units to <a href="${unitPath(a.model, a.layer.name, a.unit)}" data-link>${esc(titleOf(a))}</a></h1>
      <p class="sub">Units whose response rises and falls with this one across ${N_IMAGES} ImageNet validation images: r is the correlation of
        their peak activations, image by image. A high r means the two respond to the same images, which often (not always) means the same
        feature. Press <b>compare</b> under any unit to see the pair side by side.</p>
      <div class="cmp-unit">${visImg(a.model, a.layer.name, a.unit)}</div>
      ${groups || '<p class="muted">No match data for this unit.</p>'}`));
  }

  ms.routes.compare = async ({ params }) => {
    const pa = params.get("a"), pb = params.get("b");
    const a = pa ? parseRef(pa) : null, b = pb ? parseRef(pb) : null;
    if (a?.error || b?.error) return notFound(null, `Compare needs ?a=<model>/<layer>/<unit>&b=<model>/<layer>/<unit>. ${a?.error || b?.error}`);
    if (!a) return landing();
    if (!b) return matches(a);
    setCrumbs([["Compare"], [`${a.ref} vs ${b.ref}`]]);
    document.title = `${a.layer.name}:${a.unit} vs ${b.layer.name}:${b.unit} · Compare · Microscope`;
    const here = location.search;
    const [simA, simB, unitsA, unitsB] = await Promise.all([simUrl(a.model, a.layer.name, a.unit), simUrl(b.model, b.layer.name, b.unit),
      fill(a.model.paths.units, { layer: a.layer.name }), fill(b.model.paths.units, { layer: b.layer.name })].map(getJSON));
    a.rows = unitsA[a.unit] || []; b.rows = unitsB[b.unit] || [];
    const entry = { ...(simA[a.unit] || {}) };
    for (const [k, v] of Object.entries(simB[b.unit] || {})) entry[k] = [...(entry[k] || []), ...v];
    const label = await labelsFor(entry);
    const [htmlA, htmlB] = await Promise.all([side(a, b, simA, label), side(b, a, simB, label)]);
    if (location.pathname !== "/compare" || location.search !== here) return;  // navigated away meanwhile

    const find = (sim, s, t) => sim[s.unit]?.[t.model.name]?.find(([l, u]) => l === t.layer.name && u === t.unit);
    const ab = find(simA, a, b), ba = find(simB, b, a);
    const r = (ab || ba)?.[2];
    const shared = topClasses(a.rows).map(([k]) => k).filter(k => b.rows.some(x => x.label === k));
    const verdict = r != null
      ? `r = <b>${r3(r)}</b> across ${N_IMAGES} ImageNet validation images${ab ? ` · #${simA[a.unit][b.model.name].indexOf(ab) + 1} match for A in ${esc(groupTitle(a.model, b.model.name))}` : ""}${ba ? ` · #${simB[b.unit][a.model.name].indexOf(ba) + 1} match for B in ${esc(groupTitle(b.model, a.model.name))}` : ""}`
      : `Not among each other's top matches, so r isn't stored for this pair`;
    mount(column("main", "Compare", `
      <div class="cmp-head"><h1>Compare</h1><a class="cmp-swap" href="${compareHref(b.ref, a.ref)}" data-link title="Swap sides">swap ⇄</a></div>
      <p class="sub">${verdict}. ${shared.length ? `Top-example classes in common: ${shared.slice(0, 8).map(esc).join(", ")}.` : "No top-example classes in common."}
        r is the correlation of the two units' peak activations, image by image (<code>similar.py</code>).</p>
      <div class="cmp-grid">${htmlA}${htmlB}</div>`));
  };
})();
