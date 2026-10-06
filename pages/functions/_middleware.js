// Social previews. Link crawlers (Twitter/X, Facebook, Slack, iMessage) don't run the viewer's JavaScript,
// so every URL would otherwise share index.html's bare <head>. For HTML responses this adds per-URL
// title, description and Open Graph / Twitter card tags. Cards are rendered ahead of time by og_cards.py
// into R2 (served at /media/og/...); titles come from small per-layer og.json files next to the assets.
const SITE = "Microscope";
const DEFAULT_DESC = "Browse every unit of InceptionV1, VGG19 and CLIP ResNet-50: feature visualizations, " +
  "dataset examples, circuits and sparse-autoencoder features. A rebuild of OpenAI Microscope.";

let manifests = null;   // per isolate
async function json(env, origin, path) {
  const r = await env.ASSETS.fetch(new URL(path, origin));
  if (!r.ok || !(r.headers.get("content-type") || "").includes("json")) return {};
  try { return await r.json(); } catch { return {}; }
}

// parsed og.json files, kept per isolate (a few hundred KB each), oldest dropped beyond 12
const metaCache = new Map();
function layerMeta(env, origin, path) {
  if (!metaCache.has(path)) {
    if (metaCache.size >= 12) metaCache.delete(metaCache.keys().next().value);
    metaCache.set(path, json(env, origin, path));
  }
  return metaCache.get(path);
}

async function pageMeta(url, env) {
  const og = p => `${url.origin}/media/og/${p}`;
  const fallback = { title: SITE, description: DEFAULT_DESC, image: og("home.jpg") };
  const [model, layer, third, fourth] = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  if (!model) return fallback;
  if (model === "glossary") return { ...fallback, title: `Glossary · ${SITE}`, description: "What each view in Microscope shows and how it was computed." };

  if (model === "image") {
    const id = Number(layer);
    return Number.isInteger(id) && id >= 0
      ? { ...fallback, title: `ImageNet val image ${id} · ${SITE}`,
          description: "One photo, and every unit across InceptionV1, VGG19 and CLIP ResNet-50 that ranks it among its top examples." }
      : { ...fallback, title: `Images · ${SITE}` };
  }
  if (model === "compare") return { ...fallback, title: `Compare units · ${SITE}`,
    description: "Two units side by side: feature visualizations, top dataset examples and labels." };
  manifests ??= Promise.all([json(env, url.origin, "/manifest.json"), json(env, url.origin, "/manifest_sae.json")]);
  const [main, saeMan] = await manifests;
  const nets = main.models || [];
  const m = nets.find(x => x.name === model);
  if (!m) return fallback;
  if (!layer) return { title: `${m.title} · ${SITE}`, description: m.description, image: og(`${m.name}/model.jpg`) };
  if (layer === "search") return { ...fallback, title: `Search · ${m.title} · ${SITE}` };
  const l = m.layers.find(x => x.name === layer);
  if (!l) return fallback;

  // SAE features live at /<model>/<layer>/sae[/<f>]; the SAE manifest belongs to the first (base) network
  const sae = third === "sae" && m === nets[0] ? (saeMan.models || [])[0] : null;
  const name = sae ? sae.name : m.name;
  const unitStr = sae ? fourth : third;
  if (third === "sae" && !sae?.layers.some(x => (x.base_layer || x.name) === layer)) return fallback;
  const what = sae ? "SAE features" : "units";
  if (unitStr === undefined) {
    const n = sae ? sae.layers.find(x => (x.base_layer || x.name) === layer).channels : l.channels;
    return { title: `${m.title} · ${layer}${sae ? " · SAE features" : ""} · ${SITE}`,
             description: `${n.toLocaleString("en-US")} ${what} of ${m.title} ${layer}: feature visualizations, dataset examples${sae ? "" : " and circuits"}.`,
             image: og(`${name}/${layer}/layer.jpg`) };
  }
  const u = Number(unitStr);
  if (!Number.isInteger(u) || u < 0) return fallback;
  const [label, desc] = (await layerMeta(env, url.origin, `/assets/${name}/${layer}/og.json`))[u] || ["", ""];
  const kind = sae ? "SAE feature" : "unit";
  return { title: `${m.title} ${layer} ${kind} ${u}${label ? `: ${label}` : ""} · ${SITE}`,
           description: desc || DEFAULT_DESC, image: og(`${name}/${layer}/${u}.jpg`) };
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export async function onRequest({ request, env, next }) {
  const res = await next();
  if (request.method !== "GET" || !(res.headers.get("content-type") || "").includes("text/html")) return res;
  const url = new URL(request.url);
  let meta;
  try { meta = await pageMeta(url, env); } catch { return res; }
  const tags = [
    ["property", "og:site_name", SITE], ["property", "og:type", "website"], ["property", "og:url", url.origin + url.pathname],
    ["property", "og:title", meta.title], ["property", "og:description", meta.description],
    ["property", "og:image", meta.image], ["property", "og:image:width", "900"], ["property", "og:image:height", "472"],
    ["name", "twitter:card", "summary_large_image"], ["name", "twitter:title", meta.title],
    ["name", "twitter:description", meta.description], ["name", "twitter:image", meta.image],
    ["name", "description", meta.description],
  ].map(([attr, key, value]) => `<meta ${attr}="${key}" content="${esc(value)}">`).join("\n");
  return new HTMLRewriter()
    .on("title", { element(e) { e.setInnerContent(meta.title); } })
    .on("head", { element(e) { e.append(tags, { html: true }); } })
    .transform(res);
}
