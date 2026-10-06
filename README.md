# Microscope

**Live: [microscope.artificial-images.com](https://microscope.artificial-images.com)**

A rebuild of [OpenAI Microscope](https://openai.com/index/microscope/) (2020, since taken down): browse every unit of a vision network, with its feature visualization, the ImageNet crops that excite it most, an auto-generated label, and the **circuit** connecting it to the layers below and above. It also adds **sparse-autoencoder (SAE) features**, a second way of dividing up a layer into cleaner, mostly single-meaning directions. The first network is InceptionV1 (GoogLeNet, the TF `inception5h` weights that Lucid and the Distill Circuits thread used). See `RESEARCH.md` for how the original was built and where this rebuild differs, and the site's [glossary](https://microscope.artificial-images.com/glossary) for what each view shows.

This repo holds the code only. The rendered images, dataset-example crops and caches (about 9 GB) aren't checked in; the pipeline below regenerates them. ImageNet validation images are needed for the dataset examples. Built with [lucent](https://github.com/greentfrapp/lucent), the PyTorch port of [Lucid](https://github.com/tensorflow/lucid).

Networks:

| Network | Layers | Units | Extras |
|---|---:|---:|---|
| InceptionV1 (GoogLeNet) | 12 | 5,804 | circuits + ablation test, auto-labels, SAE features on mixed3b–mixed5b |
| VGG19 | 16 | 5,504 | circuits, auto-labels |
| CLIP ResNet-50 | 7 | 6,976 | circuits, auto-labels, "words that drive this unit" (text view) on layer4_1 and layer4 |

Every unit has a feature visualization and 16 ImageNet-val dataset examples. Pick a network on the home page (`/`). Each network's search box searches that network's labels. `/glossary` explains every view.

## Commands

```bash
# one-time: uv venv --python 3.12 && uv pip install -r requirements.txt

python precompute.py --layer mixed4a          # feature vis + dataset examples + manifest; resumable
python precompute.py --layer mixed4b --batch 64 --device cuda
python precompute.py --layer mixed4a --channels 0-31,40   # subset
python precompute.py --layer mixed4a --recrop  # redo example crops from the cached top-k
python precompute.py --layer manifest          # only rewrite site/manifest.json

python serve.py                                # http://127.0.0.1:8000/inceptionv1/mixed4a/123
python check.py --layer mixed4a --random 5     # contact sheet: vis next to top examples
python circuits.py                             # circuit data for every layer (~2 min on an A5000)
```

On a remote CUDA box (these renders used 2× RTX A5000), `remote.sh` handles copying, running and pulling:

```bash
export AI_HOST=user@gpu-host
./remote.sh setup                               # rsync code + data + assets; venv inherits conda env `comfy` torch
CUDA_VISIBLE_DEVICES=0 ./remote.sh run mixed4b --batch 64
CUDA_VISIBLE_DEVICES=1 ./remote.sh run mixed4c --batch 64   # second GPU, different layer
./remote.sh status mixed4b
./remote.sh pull                                # assets + timings back to the Mac, rebuild manifest
```

Don't run two GPU jobs on the Mac at once. Doing that crashed it once.

## What the pipeline does

- **Dataset**: ImageNet ILSVRC2012 val, all 50,000 images, from the ungated HF mirror `benjamin-paine/imagenet-1k-256x256`. Images are center-cropped to a square and resized to 256px (Lanczos). They're fed to the network at 256px, without a further crop.
- **Dataset examples**: one streaming pass computes the top-32 per channel for **all 12 layers** at once, keyed on the spatial max of the post-ReLU activation (`cache/topk_<layer>.npz`, 56 s on an A5000). Each saved crop is centered on the argmax cell and sized at 0.75 × the theoretical receptive field (mixed4a: 128px of the 256px image). Microscope used fixed crops of 40/60/80/120px.
- **Feature vis**: lucent's `param.image(224)`: FFT, decorrelated color, the same `standard_transforms`, `x*255-117` preprocessing, Adam lr 0.05, 512 steps. Channels are optimized together in batches; each image's loss only touches its own channel, so this is equivalent to separate runs. `microscope/featurevis.py` replaces lucent's `render_vis`, which adds a hook to every module on every call and never removes them. It hooks only the target layer and stops the forward pass there.
- **lucent fixes**:
  - (1) Device patch for MPS (`microscope/device.py`).
  - (2) LRN fix (`microscope/model.py`). lucent's port is 9× too weak, which makes activations about 3× Microscope's. With the fix they match Microscope's published values to within about 10%.

## Viewer

The layout follows the original Microscope: an architecture strip (output at the top, input at the bottom; each inception block is drawn with its 1×1/3×3/5×5/pool branches), a column of the current layer's units, and the main panel. Layer pages group units by branch.

Each unit page has a **circuit panel**:
- *Built from the layer below:* the units in the previous layer that most excite or inhibit this one.
- *Feeds into the layer above:* the units this one most helps build or suppresses.

The panel is measured on the unit's top 16 dataset images, as activation × gradient at the unit's peak, and each entry is shown as a share of the target unit's activation (`circuits.py`). As a check against Distill's Curve Detectors article: 11 of its 14 mixed4a curve units have a mixed3b curve detector among their top 8 inputs, against 13% of random units.

### Exploring across units and networks

Each of these lives in its own `site/ext/*.js` file. The extension hooks are documented at the top of `site/app.js`.
- **Circuit graph** (`ext/graph.js`): the circuit panel can draw as a network graph, with layers as columns and edge thickness showing strength. Green edges excite, red inhibit. It expands to 2 hops, has a strength slider, and clicking a node recenters on it. Graph | List switches back to the tiles.
- **Image pages** (`ext/images.js`, `image_index.py`): `/image/<id>` shows one ImageNet-val photo with every unit, in every network and the SAE lens, that ranks it among its top 16. Hovering shows each unit's crop on the photo, and dataset-example crops on unit pages link here. The index is sharded (`site/assets/_images/*.json`, 196 files), with 256px thumbnails (49k WebP, about 530 MB).
- **Similar units in other networks** (`ext/compare.js`, `similar.py`): each unit page lists the units in the other networks, and in other layers of its own, whose peak activations correlate with it best (Pearson r over 8,000 val images, computed in about 10 min on two A5000s). `/compare?a=<model>/<layer>/<unit>&b=…` shows two units side by side. The data is sharded per 256 units (`similar/<k>.json`), and SAE features active on under 1% of images are marked as noisy.

## Timings

The dataset pass for all layers takes 56 s on an A5000 (103 s on the M5 Pro). Crops take about 30 s per layer.

Feature vis per channel at 512 steps, batch 64. A5000 times are measured; the full mixed4a run took 1.19 s per channel. M5 Pro (MPS) times are about 3.5–4× slower: mixed4a measured 4.5 s per channel at batch 32.

| Layer | Units | s/unit (A5000) | Layer time, 1× A5000 | Est. on M5 Pro |
|---|---:|---:|---:|---:|
| conv2d0 | 64 | 0.38 | 0.4 min | 2 min |
| conv2d1 | 64 | 0.47 | 0.5 min | 2 min |
| conv2d2 | 192 | 0.66 | 2.1 min | 8 min |
| mixed3a | 256 | 0.94 | 4.0 min | 15 min |
| mixed3b | 480 | 1.12 | 9.0 min | 30 min |
| **mixed4a** | **508** | **1.19** | **10.1 min (done)** | **38 min** |
| mixed4b | 512 | 1.26 | 10.8 min | 40 min |
| mixed4c | 512 | 1.33 | 11.4 min | 42 min |
| mixed4d | 528 | 1.39 | 12.2 min | 45 min |
| mixed4e | 832 | 1.47 | 20.4 min | 75 min |
| mixed5a | 832 | 1.52 | 21.1 min | 80 min |
| mixed5b | 1024 | 1.56 | 26.6 min | 100 min |

The remaining 11 layers take about **2 h on one A5000, or about 1 h split across both GPUs**. On the Mac that's roughly 7 h. Disk use is about 125 MB per 500 units (63 MB vis PNG + 63 MB JPEG crops), so all layers come to roughly 1.5 GB.

## Differences from the original Microscope

- 512 steps and 1 render per unit. Microscope used 4096 steps × 3 repeats and cropped back to 224 after the transforms, which gave its renders a dark vignette. Ours fill the frame; the motifs match (see `RESEARCH.md` and `cache/check/`).
- 16 dataset examples per unit, against Microscope's 100. There are no neuron-objective renders, YFCC examples, or other views.

## Publishing

```bash
python build_site.py                                        # images served from /media on the same host
python build_site.py --asset-base https://media.example.com # images on object storage (R2/S3)
```

`build_site.py` reads `site/` and doesn't change it. It writes `dist/`:
- `dist/site/` is the static part: `index.html`, `app.js`, `style.css`, the manifests, all JSON, and a Pages `_redirects` file for deep links. Deploy it to Cloudflare Pages, Vercel or Netlify.
- `dist/media/` holds the images. Each feature vis is a WebP (q85), and each unit's 16 dataset-example crops are combined into one 4×4 WebP sprite (128px cells, q80). Upload it to object storage.

Re-running only rebuilds missing or outdated files (5 s when nothing changed; 44 s for a full build on the M5 Pro).

Measured on 2026-10-05, with InceptionV1 channels for all 12 layers plus SAE features for mixed3b and mixed4a:

| | Files | Size |
|---|---:|---:|
| `site/assets` images (dev: PNG + JPEG) | 232,951 | 3.40 GB |
| `dist/media` (WebP + sprites) | 27,406 | 1.02 GB |
| `dist/site` (static) | 59 | 24.5 MB |

**Live setup: microscope.artificial-images.com.** `./publish.sh` builds with images at `/media`, uploads `dist/media` to the R2 bucket `microscope-media` with rclone, and deploys `dist/site` to the Pages project `microscope` from `pages/`. The `wrangler.toml` there binds the bucket, and `pages/functions/media/[[path]].js` serves `/media/*` from it through the edge cache. That keeps everything on one hostname even though artificial-images.com's DNS is on Netlify: R2 custom domains need the DNS on Cloudflare, so a single CNAME record `microscope` → `microscope.pages.dev` in Netlify DNS is enough. Function calls are free up to 100k a day. `./publish.sh --dry-run` shows what would upload without changing anything.

**Social previews.** Link crawlers don't run JavaScript, so `pages/functions/_middleware.js` adds per-URL `<title>`, description, Open Graph and Twitter card tags to each HTML response. The card images are 900×472 JPEGs (laid out at 1200×630) from `og_cards.py`: one per unit and SAE feature (its feature vis, six dataset examples and its label), plus layer, network and home cards. That's about 38k cards, around 3.2 GB, in R2 under `/media/og/`. The titles come from a small `og.json` per layer. `publish.sh` runs it; it only renders missing cards. The static files listed in `_routes.json` skip the functions, so the only function calls are page loads and `/media` images.

The older two-host variant, with images on a separate R2 domain:

```bash
python build_site.py --asset-base https://media.example.com      # the R2 bucket's public/custom domain
rclone copy dist/media r2:microscope-media --transfers 32         # or: wrangler r2 object put ... per file
npx wrangler pages deploy dist/site --project-name microscope
```

R2 needs a CORS-free public bucket or a custom domain; images are loaded as `<img>` and CSS backgrounds, so CORS isn't required. Pages caps a deployment at 20k files, which is why the images live outside `dist/site`. To serve everything from one host instead, copy `dist/media` to `dist/site/media` and build without `--asset-base`. That host must allow about 27k files.

**License caveat:** the dataset-example crops come from ImageNet, whose terms allow non-commercial research use. The feature visualizations, labels and circuit data are generated here.
