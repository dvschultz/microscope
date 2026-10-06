// graph extension: registers on window.MS (see the extension-points comment at the top of app.js)
// Circuit graph on unit pages: layers as columns (input left, output right), the unit in the middle, its strongest
// inputs and outputs, optionally a second hop chained from the neighbouring layers' circuits. Nodes are HTML links over
// an SVG of edges. A Graph | List toggle swaps it with app.js's tile lists (the .circuit sibling of #circuit-ext).
window.MS ||= {};
MS.circuitViews ||= [];

(() => {
  const TOPS = [["4", "Top 4"], ["6", "Top 6"], ["8", "Top 8"]];
  const SIZE = { 0: 72, 1: 44, 2: 36 };   // thumbnail size by hop distance
  const FAN2 = [2, 1];                    // second hop: strongest excitatory, inhibitory per intermediate unit
  const COL2_MAX = 12;                    // cap on a second-hop column
  const short = (s, n) => !s ? "" : s.length > n ? s.slice(0, n - 1) + "…" : s;
  const abbr = name => name.replace(/^mixed/, "");

  const circFile = (model, method, layer) =>
    method === "ablation" && layer.circuits_ablation ? "circuits_ablation.json" : "circuits.json";

  // strongest entries of one side ({pos, neg}): n split about 2:1 between excitatory and inhibitory
  function strongest(side, n) {
    if (!side) return [];
    const nNeg = Math.floor(n / 3);
    return [...(side.pos || []).slice(0, n - nNeg), ...(side.neg || []).slice(0, nNeg)];
  }

  MS.circuitViews.push(async ctx => {
    const { model, layer, unit, el, circuits, method, labels } = ctx;
    if (!el || ctx.isSae) return;
    const list = el.parentElement.querySelector(":scope > .circuit");
    const circ = circuits?.[unit];
    if (!circ || (!circ.in && !circ.out)) return;   // nothing computed: app.js's list already says so

    const i = model.layers.indexOf(layer);
    const L = k => model.layers[i + k];
    prefs.circuitView ||= "graph"; prefs.graphHops ||= "1"; prefs.graphTop ||= "6";

    el.innerHTML = `<div class="cg-bar"></div><div class="cg-wrap"></div>`;
    const bar = el.firstElementChild, wrap = el.lastElementChild;
    const controls = document.createElement("span");
    controls.className = "cg-controls";
    controls.innerHTML = `<label class="cg-min" title="Hide connections weaker than this share of the receiving unit's activation">
      min <input type="range" min="0" max="1" step="0.01"><output></output></label>
      <span class="cg-legend"><i class="pos"></i>excites <i class="neg"></i>inhibits</span>`;
    bar.append(
      segmented([["graph", "Graph"], ["list", "List"]], prefs.circuitView, v => { prefs.circuitView = v; savePrefs(); applyView(); }),
      controls);
    controls.prepend(
      segmented([["1", "1 hop"], ["2", "2 hops"]], prefs.graphHops, v => { prefs.graphHops = v; savePrefs(); build(); }),
      segmented(TOPS, prefs.graphTop, v => { prefs.graphTop = v; savePrefs(); build(); }));
    const slider = controls.querySelector("input"), out = controls.querySelector("output");

    function applyView() {
      const g = prefs.circuitView === "graph";
      wrap.hidden = controls.hidden = !g;
      if (list) list.hidden = g;
      if (g) wrap.firstChild ? draw() : build();   // redraw: the width may have changed while hidden
    }

    let graph = null, sliderMax = 0.2;
    async function build() {
      const hops = +prefs.graphHops, top = +prefs.graphTop;
      // second hop: prev layer's `in` lists reach two layers down, next layer's `out` lists two layers up
      const far = hops === 2 ? await Promise.all([
        L(-1) && L(-2) ? getJSON(assetUrl(model, L(-1).name, circFile(model, method, L(-1)))) : {},
        L(1) && L(2) ? getJSON(assetUrl(model, L(1).name, circFile(model, method, L(1)))) : {},
      ]) : [{}, {}];
      const labs = await Promise.all([-2, -1, 1, 2].map(k =>
        L(k) && (Math.abs(k) === 1 || hops === 2) ? getLabels(model, L(k).name) : {}));
      if (!el.isConnected) return;
      const labelOf = { [-2]: labs[0], [-1]: labs[1], 0: labels, 1: labs[2], 2: labs[3] };

      const nodes = new Map(), edges = [];
      const node = (k, u) => {
        const id = `${k}:${u}`;
        if (!nodes.has(id)) nodes.set(id, { id, k, u, layer: L(k).name, label: labelOf[k]?.[u]?.label || "" });
        return nodes.get(id);
      };
      const center = node(0, unit);
      // 1 hop: this unit's own circuit
      for (const [c, w] of strongest(circ.in, top)) if (L(-1)) edges.push({ a: node(-1, c), b: center, w });
      for (const [c, w] of strongest(circ.out, top)) if (L(1)) edges.push({ a: center, b: node(1, c), w });
      // 2 hops: the strongest inputs of each input, the strongest outputs of each output, then every listed edge
      // between shown units (several inputs often share an input, which is what the list can't show)
      for (const [dir, k, data, key] of [[-1, -1, far[0], "in"], [1, 1, far[1], "out"]]) {
        if (!data || !Object.keys(data).length) continue;
        const mids = [...nodes.values()].filter(n => n.k === k);
        const score = new Map();
        for (const m of mids) {
          const side = data[m.u]?.[key];
          if (!side) continue;
          for (const [c, w] of [...(side.pos || []).slice(0, FAN2[0]), ...(side.neg || []).slice(0, FAN2[1])])
            score.set(c, Math.max(score.get(c) || 0, Math.abs(w)));
        }
        const keep = new Set([...score].sort((x, y) => y[1] - x[1]).slice(0, COL2_MAX).map(([c]) => c));
        for (const m of mids) {
          const side = data[m.u]?.[key];
          for (const [c, w] of side ? [...(side.pos || []), ...(side.neg || [])] : [])
            if (keep.has(c)) edges.push(dir < 0 ? { a: node(2 * dir, c), b: m, w } : { a: m, b: node(2 * dir, c), w });
        }
      }
      sliderMax = Math.max(0.01, ...edges.map(e => Math.abs(e.w)));
      slider.max = String(sliderMax);
      slider.step = String(sliderMax / 100);
      slider.value = String(Math.min(minW, sliderMax));
      graph = { nodes, edges, center, hops };
      draw();
    }

    function draw() {
      if (!graph) return;
      const min = +slider.value;
      out.textContent = min ? `≥ ${pct(min)}` : "all";
      const { edges, center, hops } = graph;
      // keep edges over the threshold, and only nodes still connected to the unit through them
      const kept = edges.filter(e => Math.abs(e.w) >= min);
      const live = new Set([center.id]);
      for (const k of [1, 2]) for (const e of kept) {
        if (e.b.k === -k + 1 && live.has(e.b.id) && e.a.k === -k) live.add(e.a.id);
        if (e.a.k === k - 1 && live.has(e.a.id) && e.b.k === k) live.add(e.b.id);
      }
      const shown = kept.filter(e => live.has(e.a.id) && live.has(e.b.id));
      const cols = {};
      for (const n of graph.nodes.values()) if (live.has(n.id)) (cols[n.k] ||= []).push(n);

      // order: direct neighbours by weight into/out of the unit, second hop by the mean height of their neighbours
      const direct = new Map(shown.filter(e => e.a === center || e.b === center).map(e => [e.a === center ? e.b.id : e.a.id, e.w]));
      // excitatory first, strongest on top, then inhibitory by strength
      const dw = n => direct.get(n.id) ?? 0;
      for (const k of [-1, 1]) cols[k]?.sort((x, y) => (dw(x) < 0) - (dw(y) < 0) || Math.abs(dw(y)) - Math.abs(dw(x)));

      const W = Math.min(wrap.clientWidth || 800, 1180);
      const ks = [-2, -1, 0, 1, 2].filter(k => Math.abs(k) <= hops && L(k));
      const side = hops === 1 && W >= 560 ? Math.min(220, W * 0.22) : 0;   // room for labels at 1 hop
      // column centres spread between the margins: labels at 1 hop, or the "reads the image" / "feeds the output" note
      const END = 120, kMin = ks[0], kMax = ks.at(-1);
      const c0 = (L(-1) ? side : END) + 4 + SIZE[Math.abs(kMin)] / 2, c1 = W - (L(1) ? side : END) - 4 - SIZE[Math.abs(kMax)] / 2;
      const CX = k => c0 + (k - kMin) / Math.max(1, kMax - kMin) * (c1 - c0);
      const gap = 10, head = 22;
      const colH = k => (cols[k] || []).reduce((h, n) => h + SIZE[Math.abs(k)] + gap, -gap);
      const H = Math.max(SIZE[0] + 14, ...ks.map(colH)) + head + 8;
      const place = k => {
        let y = head + (H - head - colH(k)) / 2;
        for (const n of cols[k] || []) {
          n.s = SIZE[Math.abs(k)]; n.x = CX(k) - n.s / 2; n.y = y; y += n.s + gap;
        }
      };
      place(0); place(-1); place(1);
      for (const k of [-2, 2]) {
        const near = k / 2;
        const bary = n => {
          const es = shown.filter(e => e.a === n || e.b === n);
          const t = es.reduce((s, e) => s + Math.abs(e.w), 0) || 1;
          return es.reduce((s, e) => { const m = e.a === n ? e.b : e.a; return s + (m.k === near ? m.y : 0) * Math.abs(e.w); }, 0) / t;
        };
        cols[k]?.sort((x, y) => bary(x) - bary(y));
        place(k);
      }

      const maxW = sliderMax;
      const path = e => {
        const ax = e.a.x + e.a.s, ay = e.a.y + e.a.s / 2, bx = e.b.x, by = e.b.y + e.b.s / 2, mx = (ax + bx) / 2;
        const r = Math.abs(e.w) / maxW;
        return `<path class="${e.w < 0 ? "neg" : "pos"}" data-a="${e.a.id}" data-b="${e.b.id}" d="M${ax},${ay} C${mx},${ay} ${mx},${by} ${bx},${by}"
          stroke-width="${(1 + 8 * r).toFixed(2)}" stroke-opacity="${(0.22 + 0.68 * Math.sqrt(r)).toFixed(2)}"/>`;
      };
      // strongest edges last so they sit on top
      const svg = `<svg class="cg-edges" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true">
        ${[...shown].sort((x, y) => Math.abs(x.w) - Math.abs(y.w)).map(path).join("")}</svg>`;
      const heads = ks.map(k => {
        return `<div class="cg-head${k === 0 ? " cur" : ""}" style="left:${CX(k)}px">${esc(L(k).name)}${k === 0 ? ` · ${unit}` : ""}</div>`;
      }).join("");
      const ends = [!L(-1) && `<div class="cg-end" style="left:4px;top:${head + (H - head) / 2 - 8}px">← reads the image</div>`,
        !L(1) && `<div class="cg-end r" style="right:4px;top:${head + (H - head) / 2 - 8}px">feeds the output →</div>`].filter(Boolean).join("");
      const nodeHtml = [...graph.nodes.values()].filter(n => live.has(n.id)).map(n => {
        const w = direct.get(n.id);
        const lbl = side && n.k !== 0 && n.label
          ? `<span class="cg-lbl ${n.k < 0 ? "l" : "r"}" style="width:${side - 8}px">${esc(short(n.label, 34))}</span>` : "";
        const wt = w !== undefined ? `<span class="cg-w ${n.k < 0 ? "r" : "l"} ${w < 0 ? "neg" : ""}">${pct(w)}</span>` : "";
        const tag = n.k === 0 ? `div aria-current="page"` : `a href="${unitPath(model, n.layer, n.u)}" data-link`;
        return `<${tag} class="cg-node${n.k === 0 ? " cur" : ""}" data-id="${n.id}"
            style="left:${n.x}px;top:${n.y}px;width:${n.s}px;height:${n.s}px" aria-label="${esc(`${n.layer} unit ${n.u}${n.label ? ": " + n.label : ""}`)}">
          ${visImg(model, n.layer, n.u)}<span class="n">${n.u}</span>${wt}${lbl}</${n.k === 0 ? "div" : "a"}>`;
      }).join("");
      wrap.innerHTML = `<div class="cg" style="height:${H}px;width:${W}px">${svg}${heads}${ends}${nodeHtml}<div class="cg-tip" hidden></div></div>`;
      graph.shown = shown;
    }

    // hover: highlight a node's edges and neighbours, tooltip with its label and connections
    wrap.addEventListener("mouseover", e => {
      const a = e.target.closest(".cg-node");
      const g = wrap.querySelector(".cg");
      if (!a || !g || !graph?.shown) return;
      const id = a.dataset.id, n = graph.nodes.get(id);
      const mine = graph.shown.filter(x => x.a.id === id || x.b.id === id);
      const near = new Set(mine.flatMap(x => [x.a.id, x.b.id]));
      g.classList.add("hovering");
      g.querySelectorAll("path").forEach(p => p.classList.toggle("hl", p.dataset.a === id || p.dataset.b === id));
      g.querySelectorAll(".cg-node").forEach(x => x.classList.toggle("near", near.has(x.dataset.id)));
      const rows = mine.sort((x, y) => Math.abs(y.w) - Math.abs(x.w)).slice(0, 6).map(x => {
        const m = x.a.id === id ? x.b : x.a;
        return `<div><span class="${x.w < 0 ? "neg" : "pos"}">${pct(x.w)}</span> ${x.a.id === id ? "→" : "←"} ${esc(abbr(m.layer))}:${m.u}</div>`;
      }).join("");
      const tip = g.querySelector(".cg-tip");
      tip.innerHTML = `<b>${esc(n.layer)} unit ${n.u}</b>${n.label ? `<div class="cg-tl">${esc(n.label)}</div>` : ""}${rows}`;
      tip.hidden = false;
      const left = n.x + n.s + 8, flip = left + 220 > g.clientWidth;
      tip.style.left = `${flip ? Math.max(0, n.x - 228) : left}px`;
      tip.style.top = `${Math.max(0, n.y - 4)}px`;
    });
    wrap.addEventListener("mouseout", e => {
      if (e.relatedTarget?.closest?.(".cg-node")) return;
      const g = wrap.querySelector(".cg");
      if (!g) return;
      g.classList.remove("hovering");
      g.querySelector(".cg-tip").hidden = true;
    });
    // the threshold is per page (weights differ a lot between units), so it isn't saved in prefs
    let minW = 0;
    slider.addEventListener("input", () => { minW = +slider.value; draw(); });

    let lastW = 0;
    new ResizeObserver(() => {
      const w = wrap.clientWidth;
      if (w && Math.abs(w - lastW) > 8 && !wrap.hidden) { lastW = w; draw(); }
    }).observe(wrap);
    applyView();
  });
})();
