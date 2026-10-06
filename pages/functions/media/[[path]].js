// Serves /media/<key> from the R2 bucket bound as MEDIA, so images share the site's hostname
// (R2 custom domains need the zone on Cloudflare; artificial-images.com's DNS is on Netlify).
// Responses go through the edge cache, so repeat views of an image skip R2.
export async function onRequestGet({ params, env, request, waitUntil }) {
  const key = [].concat(params.path || []).join("/");
  if (!key || key.includes("..")) return new Response("Not found", { status: 404 });

  const cache = caches.default;
  const hit = await cache.match(request);
  if (hit) return hit;

  const obj = await env.MEDIA.get(key);
  if (!obj) return new Response("Not found", { status: 404 });

  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  headers.set("etag", obj.httpEtag);
  headers.set("content-type", "image/webp");
  headers.set("cache-control", "public, max-age=604800");  // a week; re-renders keep their paths
  const res = new Response(obj.body, { headers });
  waitUntil(cache.put(request, res.clone()));
  return res;
}

// HEAD (link checkers, crawlers): same status and headers as GET, no body.
export async function onRequestHead(context) {
  const res = await onRequestGet(context);
  return new Response(null, { status: res.status, headers: res.headers });
}
