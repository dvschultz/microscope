// images extension: registers on window.MS (see the extension-points comment at the top of app.js)
// /image/<id>: one ImageNet-val photo and every unit, in every network and the SAE lens, that has it among its top 16
// dataset examples. Data: site/assets/_images/<id // 256>.json from image_index.py (one shard per page), thumbnails
// at <media>/_images/thumbs/<id>.webp. Dataset-example crops on unit pages link here via MS.exampleHref.
window.MS ||= {};
(() => {
  const ms = window.MS;
  ms.routes ||= {};
  const SHARD = 256, N_VAL = 50000;
  const shardUrl = id => `/assets/_images/${String(Math.floor(id / SHARD)).padStart(3, "0")}.json`;
  // same prefix as the feature vis: /assets in dev, /media (or the --asset-base origin) in the published build
  const fmtAct = x => x >= 10 ? fmt(x) : x.toPrecision(2);  // SAE activations can be well below 1
  const thumbUrl = (m, id) => `${m.models[0].paths.vis.split(`/${m.models[0].name}/`)[0]}/_images/thumbs/${id}.webp`;

  ms.exampleHref = (model, layerName, unit, rank, row) => row?.img != null ? `/image/${row.img}` : null;

  ms.routes.image = async ({ manifest: m, segments }) => {
    if (segments[0] === undefined) return randomImage();
    const id = Number(segments[0]);
    if (!Number.isInteger(id) || id < 0 || id >= N_VAL) return notFound(null, `Image ids run 0–${N_VAL - 1} (ImageNet validation index).`);
    setCrumbs([["Images"], [`Image ${id}`]]);
    document.title = `Image ${id} · Microscope`;
    const shard = await getJSON(shardUrl(id));
    if (location.pathname.split("/").filter(Boolean)[1] !== String(id)) return;  // navigated away meanwhile
    const img = shard.images?.[id];
    const ids = Object.keys(shard.images || {}).map(Number).sort((a, b) => a - b);
    const prev = ids.filter(x => x < id).at(-1) ?? (id > 0 ? id - 1 : null);
    const next = ids.find(x => x > id) ?? (id < N_VAL - 1 ? id + 1 : null);
    const nav = `<div class="nav-btns">${prev !== null ? `<a href="/image/${prev}" data-link>← ${prev}</a>` : ""}
      ${next !== null ? `<a href="/image/${next}" data-link>${next} →</a>` : ""}</div>`;
    keyHandler = e => {
      if (e.key === "ArrowLeft" && prev !== null) navigate(`/image/${prev}`);
      if (e.key === "ArrowRight" && next !== null) navigate(`/image/${next}`);
    };
    if (!img) return mount(column("main", `Image ${id}`, `${nav}<h1>ImageNet val image ${id}</h1>
      <p class="sub">This photo is not among the top 16 dataset examples of any unit or SAE feature shown here.</p>`));

    // shard keys -> manifest models; entries grouped network -> layer
    const models = shard.models.map(n => m.models.find(x => x.name === n));
    const groups = new Map();  // model name -> Map(layer -> entries)
    img.e.forEach(([k, unit, rank, act, top, left, size], i) => {
      const [mi, layer] = shard.layers[k], model = models[mi];
      if (!model) return;
      if (!groups.has(model.name)) groups.set(model.name, new Map());
      const g = groups.get(model.name);
      if (!g.has(layer)) g.set(layer, []);
      g.get(layer).push({ i, model, layer, unit, rank, act, box: [top, left, size] });
    });
    // networks in manifest order, each SAE lens right after its base network; layers output first, like the architecture strip
    const order = m.models.filter(x => !x.base).flatMap(b => [b, saeOf(b)]).filter(x => x && groups.has(x.name));
    const all = [...groups.values()].flatMap(g => [...g.values()].flat());
    const title = model => model.base ? `${baseOf(model).title} · SAE features` : model.title;
    const kind = model => model.base ? "feature" : "unit";

    const entry = e => `<a class="im-e" href="${unitPath(e.model, e.layer, e.unit)}" data-link data-i="${e.i}"
        title="${esc(title(e.model))} · ${e.layer} ${kind(e.model)} ${e.unit} · #${e.rank + 1} of 16 · activation ${fmtAct(e.act)}">
        <div class="im-pair">${visImg(e.model, e.layer, e.unit)}${exImg(e.model, e.layer, e.unit, e.rank)}</div>
        <div class="cap"><span class="lab" data-lab="${e.model.name}/${e.layer}/${e.unit}">${kind(e.model)} ${e.unit}</span>
          <span>#${e.rank + 1} · ${fmtAct(e.act)}</span></div></a>`;
    const sections = order.map(model => {
      const g = groups.get(model.name);
      const layers = [...model.layers].reverse().filter(l => g.has(l.name));
      const n = layers.reduce((a, l) => a + g.get(l.name).length, 0);
      return `<section class="im-net" id="im-${model.name}"><h2>${esc(title(model))} <span class="muted">${n} ${kind(model)}${n > 1 ? "s" : ""}</span></h2>
        ${layers.map(l => { const es = g.get(l.name); return `<div class="branch-head im-lh" data-layer="${model.name}/${l.name}"><b>${l.name}</b>
          <span>${es.length} ${kind(model)}${es.length > 1 ? "s" : ""}${l.receptive_field ? ` · receptive field ${l.receptive_field}px` : ""}</span></div>
          <div class="im-grid">${es.map(entry).join("")}</div>`; }).join("")}</section>`;
    }).join("");
    const summary = order.map(model => {
      const g = groups.get(model.name), es = [...g.values()].flat();
      const best = es.reduce((a, e) => e.rank < a.rank ? e : a);
      return `<tr><td><button class="im-jump" data-jump="im-${model.name}">${esc(title(model))}</button></td>
        <td>${es.length} ${kind(model)}${es.length > 1 ? "s" : ""} in ${g.size} layer${g.size > 1 ? "s" : ""} · best #${best.rank + 1}
          (${best.layer} ${best.unit})</td></tr>`;
    }).join("");

    mount(column("main", `Image ${id}`, `${nav}
      <h1>ImageNet val image ${id} <span class="unit-label">${esc(img.label)}</span></h1>
      <p class="sub">Every unit whose top 16 dataset examples include this photo, across all networks. Each entry shows the unit's
        feature visualization, its crop of this photo, the photo's rank among the unit's examples and the activation. Hover an entry
        to see its crop on the photo; hover a layer name to see all of that layer's crops.</p>
      <div class="im-layout">
        <aside class="im-side">
          <div class="im-photo"><img src="${thumbUrl(m, id)}" alt="${esc(img.label)}"
            onerror="this.closest('.im-photo').classList.add('missing')"><div class="im-boxes"></div></div>
          <table class="facts">
            <tr><td>ImageNet class</td><td>${esc(img.label)}</td></tr>
            <tr><td>Val index</td><td>${id}</td></tr>
            <tr><td>In the top 16 of</td><td>${all.length} unit${all.length > 1 ? "s" : ""}${order.some(x => x.base) ? " and SAE features" : ""}</td></tr>
            ${summary}
          </table>
        </aside>
        <div class="im-main">${sections}</div>
      </div>`));

    // crop boxes on the photo (box = [top, left, size] in 256px coordinates)
    const boxes = document.querySelector(".im-boxes");
    const show = es => boxes.innerHTML = es.map(e => `<i style="top:${e.box[0] / 2.56}%;left:${e.box[1] / 2.56}%;` +
      `width:${e.box[2] / 2.56}%;height:${e.box[2] / 2.56}%"></i>`).join("");
    const byI = Object.fromEntries(all.map(e => [e.i, e]));
    const main = document.querySelector(".im-main");
    main.addEventListener("mouseover", e => {
      const a = e.target.closest(".im-e"), h = e.target.closest(".im-lh");
      if (a) show([byI[a.dataset.i]]);
      else if (h) { const [mn, ln] = h.dataset.layer.split("/"); show(groups.get(mn).get(ln)); }
    });
    main.addEventListener("mouseleave", () => show([]));
    document.querySelector(".im-side").addEventListener("click", e => {
      const b = e.target.closest("[data-jump]");
      if (b) document.getElementById(b.dataset.jump)?.scrollIntoView({ block: "start", behavior: "smooth" });
    });

    // auto labels arrive after the page is up (one labels.json per layer, cached; missing files are {})
    const layerKeys = [...new Set(all.map(e => `${e.model.name}/${e.layer}`))];
    const labels = await Promise.all(layerKeys.map(k => { const [mn, ln] = k.split("/"); return getLabels(models.find(x => x?.name === mn), ln); }));
    const byLayer = Object.fromEntries(layerKeys.map((k, j) => [k, labels[j]]));
    for (const el of main.querySelectorAll("[data-lab]")) {
      const [mn, ln, u] = el.dataset.lab.split("/");
      const lab = byLayer[`${mn}/${ln}`]?.[u];
      if (lab) { el.textContent = lab.label; el.closest(".im-e").title += `\n${lab.label}: ${lab.description}`; }
    }
  };

  async function randomImage() {
    const s = Math.floor(Math.random() * Math.ceil(N_VAL / SHARD));
    const shard = await getJSON(shardUrl(s * SHARD));
    const ids = Object.keys(shard.images || {});
    navigate(`/image/${ids.length ? ids[Math.floor(Math.random() * ids.length)] : s * SHARD}`, true);
  }
})();
