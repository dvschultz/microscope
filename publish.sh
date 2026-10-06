#!/usr/bin/env bash
# Publish the viewer at microscope.artificial-images.com: images to Cloudflare R2, the site to
# Cloudflare Pages. pages/functions/media serves /media/* from the R2 bucket, so everything
# shares one hostname (no R2 custom domain needed; the zone's DNS is on Netlify).
#
#   ./publish.sh                 # build, upload images, deploy site
#   ./publish.sh --dry-run       # build, then show what would be uploaded; deploy nothing
#   ./publish.sh --images-only   # build + upload images, skip the Pages deploy
#
# Needs: `wrangler login` (done once) and an rclone remote named `r2` with
# Object Read & Write keys for the bucket (`rclone config`, type s3, provider Cloudflare).
# Override with env vars: ASSET_BASE, BUCKET, PAGES_PROJECT, RCLONE_REMOTE.
set -euo pipefail
cd "$(dirname "$0")"

ASSET_BASE="${ASSET_BASE:-/media}"   # same host, via the Pages Function
BUCKET="${BUCKET:-microscope-media}"
PAGES_PROJECT="${PAGES_PROJECT:-microscope}"
REMOTE="${RCLONE_REMOTE:-r2}"
DRY=""; DEPLOY=1
for a in "$@"; do
  case "$a" in
    --dry-run) DRY="--dry-run"; DEPLOY=0 ;;
    --images-only) DEPLOY=0 ;;
    *) sed -n 2,11p "$0"; exit 1 ;;
  esac
done

command -v rclone >/dev/null || { echo "rclone missing: brew install rclone"; exit 1; }
rclone listremotes | grep -qx "$REMOTE:" || { echo "no rclone remote '$REMOTE': run rclone config"; exit 1; }
rclone lsd "$REMOTE:$BUCKET" >/dev/null 2>&1 || rclone lsf "$REMOTE:$BUCKET" --max-depth 1 >/dev/null \
  || { echo "can't reach $REMOTE:$BUCKET: check the API token's bucket scope"; exit 1; }

echo "== build (images at $ASSET_BASE)"
.venv/bin/python build_site.py --asset-base "$ASSET_BASE"

echo "== upload dist/media -> $REMOTE:$BUCKET"
# copy, not sync: never deletes objects from the bucket. Unchanged files are skipped by size+modtime.
rclone copy dist/media "$REMOTE:$BUCKET" --transfers 32 --checkers 32 --fast-list --progress \
  --header-upload "Content-Type: image/webp" $DRY

if [ "$DEPLOY" = 1 ]; then
  echo "== deploy dist/site -> Pages project $PAGES_PROJECT"
  wrangler pages project list 2>/dev/null | grep -q "\b$PAGES_PROJECT\b" \
    || wrangler pages project create "$PAGES_PROJECT" --production-branch main
  # run from pages/: wrangler.toml there points at ../dist/site and binds the R2 bucket; functions/ deploys with it
  (cd pages && wrangler pages deploy --branch main --commit-dirty=true)
fi
echo "done"
