// Serves /media/<key> from the R2 bucket bound as MEDIA, so images share the site's hostname
// (R2 custom domains need the zone on Cloudflare; artificial-images.com's DNS is on Netlify).
// Responses go through the edge cache, so repeat views of an image skip R2.
// by extension: the bucket's stored content-type isn't reliable (uploads were tagged image/webp)
const TYPES = { webp: "image/webp", jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png" };

export async function onRequestGet({ params, env, request, waitUntil }) {
  const key = [].concat(params.path || []).join("/");
  if (!key || key.includes("..")) return new Response("Not found", { status: 404 });

  // the key carries a version so a fix to headers isn't masked by week-old cached copies
  const cacheKey = new Request(`${new URL(request.url).origin}/media/${key}?v=2`);
  const cache = caches.default;
  const hit = await cache.match(cacheKey);
  if (hit) return hit;

  const obj = await env.MEDIA.get(key);
  if (!obj) return new Response("Not found", { status: 404 });

  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  headers.set("etag", obj.httpEtag);
  headers.set("content-type", TYPES[key.split(".").pop().toLowerCase()] || "application/octet-stream");
  headers.set("cache-control", "public, max-age=604800");  // a week; re-renders keep their paths
  const res = new Response(obj.body, { headers });
  waitUntil(cache.put(cacheKey, res.clone()));
  return res;
}

// HEAD (link checkers, crawlers): same status and headers as GET, no body.
export async function onRequestHead(context) {
  const res = await onRequestGet(context);
  return new Response(null, { status: res.status, headers: res.headers });
}
