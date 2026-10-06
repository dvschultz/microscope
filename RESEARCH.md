# How OpenAI Microscope was built (research notes)

Research date: 2026-10-04. Goal: rebuild Microscope locally for InceptionV1 using `lucent` (installed: `torch-lucent 0.1.8`, `kornia 0.4.1`, `torch 2.14.1`).

**Main finding: the original Microscope InceptionV1 assets can still be downloaded** from a byte-identical public mirror at `https://openaipublic.blob.core.windows.net/microscopeprod/...`. The original host (`oaiggoh.blob.core.windows.net`) now returns 403. See §4.

Claims are labeled: **(verified)** means I checked the source or file myself; **(inferred)** means it follows from evidence but no source states it; **(unconfirmed)** means I couldn't check it.

---

## 1. Lucid (TF1) and lucent (PyTorch)

- **Lucid** (`github.com/tensorflow/lucid`) is the TF1 library behind Distill's Feature Visualization (2017), Building Blocks (2018), Activation Atlas and the Circuits thread. The Microscope About page says it's "used to generate all the visualizations in Microscope". Microscope's backend technique ids match Lucid's names: `lucid.feature_vis`, `lucid.dataset_examples`, `lucid.caricature`, `lucid.deep_dream`, `lucid.synthetic_tuning` and `lucid.text_feature_vis` **(verified, archived JS bundle and model JSON)**.
- **Microscope's feature-vis recipe is `lucid/recipes/feature_visualization.py`**, not the `render_vis` defaults. Ludwig Schubert, Microscope's first author, wrote it in July 2019 ("WIP additional recipes"). Its parameters map one-to-one onto Microscope's blob path keys: `alpha`, `negative`, `objective` (channel/neuron), batched channel indices (`start`/`stop`) and `steps` **(verified code; that Microscope used it is inferred)**. What it does:
  - `param.image(224, 224, fft=True, decorrelate=True, alpha=False, batch=len(channels))`: a native 224 image, **not 128 upsampled**.
  - `transforms = transform.standard_transforms + [transform.crop_or_pad_to(224, 224)]`
  - `with overrides.relu_overrides():` (redirected ReLU gradient)
  - Objective: `sum(sign * objectives.channel(layer, i, batch=b))`, one channel per batch element.
  - Optimizer: `tf.train.AdamOptimizer(0.05)`. The recipe default is 128 steps, but **Microscope used `steps=4096`** (from the blob path).
  - Gotcha: `minimize()` is called without `global_step`, so in this recipe `global_step` stays 0. The redirected-ReLU "≤16 steps" cutoff therefore never triggers **(inferred from code)**.
- **lucent** (`github.com/greentfrapp/lucent`) is a PyTorch port of Lucid's `optvis` plus an InceptionV1 port. The installed 0.1.8 differs from GitHub master (`2ccc434`, 2025-03): 0.1.8 hardcodes `device = cuda:0 if available else cpu`, so there's **no MPS on Mac**. Master threads `device` through `param.image(..., device=)` and removes forward hooks after `render_vis`. The constants are identical in both versions.

### lucent source facts (installed 0.1.8, read directly)
- `render_vis(model, objective_f, param_f=None, optimizer=None, transforms=None, thresholds=(512,), verbose=False, preprocess=True, progress=True, show_image=True, save_image=False, image_name=None, show_inline=False, fixed_image_size=None)`
  - `param_f` default: `param.image(128)`
  - `optimizer` default: `torch.optim.Adam(params, lr=5e-2)`
  - `transforms` default: `transform.standard_transforms` (copied)
  - If `preprocess` is set and `model._get_name()=="InceptionV1"`, it appends `preprocess_inceptionv1()` = `lambda x: x*255 - 117`. Otherwise it appends ImageNet `Normalize`.
  - If the param image is smaller than 224 on either side, it appends `Upsample(224, bilinear, align_corners=True)`, after preprocessing. It never crops a larger image.
  - The loop runs `range(1, max(thresholds)+1)`, i.e. 512 steps. Lucid runs `max(thresholds)+1` = 513.
- `transform.standard_transforms`:
  ```
  pad(12, mode="constant", constant_value=0.5)
  jitter(8)
  random_scale([1 + (i-5)/50. for i in range(11)])   # 0.90,0.92,…,1.10
  random_rotate(list(range(-10, 11)) + 5*[0])         # degrees; 0 has weight 6/26
  jitter(4)
  ```
- `param.image(w, h=None, sd=None, batch=None, decorrelate=True, fft=True, channels=None)`: `sd` defaults to 0.01, and the output goes through `torch.sigmoid`.
- `fft_image(shape, sd=0.01, decay_power=1)`:
  - Spectrum: `scale = 1/max(freq, 1/max(w,h))**decay_power`, then `irfftn(..., norm='ortho')`, then `/4.0` (the "magic" constant).
  - **Equivalent to Lucid**: Lucid multiplies `scale` by `sqrt(w*h)` and uses TF's 1/N-normalized `irfft2d`.
- Color, identical in Lucid and lucent:
  - `color_correlation_svd_sqrt = [[0.26,0.09,0.02],[0.27,0.00,-0.05],[0.27,-0.09,0.03]]`
  - This matrix is divided by `max_norm_svd_sqrt` = max column L2 norm, then applied as `t @ C_norm.T`.
  - `color_mean = [0.48,0.46,0.41]` is only used by Lucid's non-sigmoid path.
- Objectives:
  - `channel`: `-model(layer)[:, n].mean()`
  - `neuron`: center position `(shape//2)`, the same convention as Lucid. For a 224 input, mixed3b (28×28) uses position 14,14.

### Defaults table: Lucid vs lucent (vs what Microscope used)

| Item | Lucid (`render_vis` / modules) | lucent 0.1.8 | Microscope InceptionV1 (2020-07-25 assets) |
|---|---|---|---|
| Param size | `param.image(128)` | `param.image(128)` → upsampled to 224 | 224×224 native, batch of ≤32 channels **(inferred, recipe)** |
| Param type | FFT, decorrelate=True, sd=0.01, sigmoid | same | same **(inferred)** |
| FFT scaling | 1/f, `*sqrt(wh)`, TF irfft2d, /4 | 1/f, `irfftn(norm='ortho')`, /4: **equivalent** | same |
| Optimizer | Adam lr 0.05 | Adam lr 0.05 | Adam 0.05 **(inferred)** |
| Steps | thresholds=(512,) → 513 steps | 512 steps | **4096** (`steps=4096` in path) **(verified)** |
| pad | 12, constant 0.5 | 12, constant 0.5 | same |
| jitter(8), jitter(4) | `tf.random_crop`, which **shrinks** the image by d and uses a random offset | `kornia.translate` by (dx,dy) ∈ [0,d): **size unchanged, shifts only right/down**, zero (black) fill | Lucid semantics |
| random_scale | 0.90–1.10 in 0.02 steps; `resize_bilinear`, no pad | same list; `Upsample(align_corners=True)`, then pads back if <1. Pad order `[pad_y,pad_x]*2` is harmless only for square images | Lucid semantics |
| random_rotate | −10…10° plus five extra 0s; `tf.contrib.image.rotate` (TF1 default interpolation NEAREST); deg→rad uses 3.14 | same list; kornia `warp_affine` bilinear | Lucid semantics |
| Final crop | none in `render_vis`; the recipe adds `crop_or_pad_to(224,224)` (center crop or zero-pad) | **none**: the network sees ~248–273 px | crop_or_pad_to 224 **(inferred)** |
| Input scaling (InceptionV1) | `image_value_range=(-117,138)`: `-117 + x*255` | `x*255 - 117` | same |
| ReLU gradient | `redirected_relu_grad`: only for a batch entry whose whole ReLU gradient is zero; only lets through gradient that pushes the input up; ≤16 global steps | `RedirectedReLU`: **always** passes 0.1× the gradient where x<0, in both directions, every step (`inceptionv1(redirected_ReLU=True)` by default) | Lucid override **(inferred)** |
| Layout | NHWC | NCHW | n/a |

**Implementation consequence:**
- To match Microscope with lucent, use `param.image(224)`, 4096 steps and Adam 0.05. Then either replace jitter with a random crop (Lucid semantics) or append a center crop or pad back to 224 after the standard transforms.
- Consider `inceptionv1(pretrained=True, redirected_ReLU=False)` with your own Lucid-style override.
- The 2020 Microscope channel renders have a **mid-gray periphery and a dark inner "frame"** that the 2019 renders don't have (they fill the frame; see §4).
  - Likely cause **(inferred)**: `crop_or_pad_to(224)` crops the param's outer ~6–20 px when scale >1 (less gradient there) and zero-pads, i.e. black, when scale <1.
  - lucent never crops, so expect lucent renders to fill the frame more.

### Distill Feature Visualization (2017) settings, which are not the same as `standard_transforms`
- 2560 steps, decorrelated Fourier space, Adam 0.05.
- Transforms, in order:
  - pad 16
  - jitter up to 16
  - scale ∈ {1, 0.975, 1.025, 0.95, 1.05}
  - rotate ∈ {−5…5}°
  - jitter up to 8
  - crop the padding

Don't conflate these with `standard_transforms`. The Circuits "Early Vision" article adds: "small amounts of transformation robustness when visualizing the first few layers… increase as we move to higher layers. For low layers, we use L2 regularization to push pixels towards gray". For conv2d0 it shows the weights.

---

## 2. Per-unit views Microscope showed (InceptionV1)

Source: the archived React bundle (`static/js/main.7ef1d588.chunk.js`, 2021-03) and API responses.

| View (backend id) | Level | What was shown | Params |
|---|---|---|---|
| Feature Visualization (`lucid.feature_vis`) | unit | **3 channel-objective renders + 3 neuron-objective renders**. Captions: "Channel optimization objective results in a repeating pattern." / "Neuron optimization objective shows spatial preferences." Thumbnail = channel, repeat 0 | `op_feature_vis(model, op, objective, 4096, repeat)`, repeat ∈ {0,1,2} = independent restarts **(verified)** |
| Dataset Samples (`lucid.dataset_examples`) | unit | "Pieces of images from the training dataset that result in the largest activations". 10×10 grid from **ImageNet**, plus a second grid from **YFCC** (Yahoo Flickr Creative Commons) | `op_dataset_example_crops(model, op, "imagenet"\|"yfcc")` |
| Synthetic Tuning (`lucid.synthetic_tuning`) | unit | "Test patterns that show how strongly a unit reacts to specific, artificial inputs" (families of synthetic stimuli, e.g. curves/lines/angles). Enabled only for conv2d0–2 and mixed3a's 1x1, 3x3_bottleneck, 5x5_bottleneck and pool_reduce branches | no archived data found **(unconfirmed details)** |
| Caricature (`lucid.caricature`) | op | Feature inversion of a fixed set of images (e.g. "dog-drawing", "four-circles") per layer. `cossim_pow` ∈ {0.0, 1.0, 2.0} | `steps=4096`; recipe `lucid/recipes/caricature.py` (`dot_compare` objective) |
| DeepDream (`lucid.deep_dream`) | op | "maximizes the activations of all units in an op" | `steps=1024`; the file is (mis)named `caricature.png` |
| Text Feature Vis (`lucid.text_feature_vis`) | unit | **CLIP models only** (`contrastive_*`): "Text that maximizes dot product with neuron". JSON list of `[score, "<\|startoftext\|>…"]` | `lm_penalty=0.0025`, batches `start`/`end` |

- **Not present:** diversity, negative/min dataset examples, attribution or spatial views. `negative=True` feature vis returns 404 on the mirror, but negatives do exist in the 2019 GCS set (§4).
- **Neuron vs channel:**
  - Neuron = the center spatial position only.
  - Channel = mean over all positions, which gives the tiled pattern.

### How dataset examples were computed (Microscope)
Evidence: spritesheets and metadata from the mirror. Pixel analysis is my own.

- **Ranking:** `channel_{u}.json` holds **exactly 100 activation values, sorted descending** (e.g. 3b:379: 1255.9 → 1135.1). So it's the top-100 by activation **(verified)**. Whether that's 100 distinct images, each using its max over positions, or 100 positions is **(unconfirmed)**; the sheets look like distinct images. ImageNet split (train vs val) is **(unconfirmed)**.
- **Crop geometry (verified by pixel analysis):**
  - Sheets `channel_{u}_{N}.png` with N ∈ {20, 30, 40, 60} are 10×10 grids of **2N-px tiles**: 400, 600, 800 and 1200 px RGBA.
  - `_20` is *exactly* the central 40 px of each `_40` tile: mean absolute difference 0.0, whereas downscaling `_40` gives a difference of 49.7.
  - So **N is a crop half-width in native pixels, not a display scale**. Each example is a square 2N-px crop at the image's native (network-input) resolution.
  - **The crop is centered on one activation cell's input-space center, `stride*i + stride/2`.** Transparent (alpha=0) strips mark where the crop runs past the image border, and their widths quantize to the layer stride:
    - mixed3b (stride 8): widths 2/10/18 (N=30), 4/12/20/28 (N=40), 8…48 (N=60). All imply centers ≡ 4 (mod 8).
    - mixed4b (stride 16): widths 4/16/32. Centers ≡ 8 (mod 16).
  - Out-of-image area is **transparent**, not padded with image content. The UI draws a transparency-checker background (`/transparency-indicator.png`).
- **Not receptive-field sized:**
  - Microscope used the same fixed crops (40/60/80/120 px) for every layer, e.g. 120 px ≫ mixed3b's listed RF of 55×57.
  - The UI uses `spritesheet[0]` (`_20`) for the thumbnail and `spritesheet[2]` (`_40`) for the main view.
  - The Distill Curve Detectors quilts instead say "Receptive-field sized crops of an image". That is Distill's own figure method, not Microscope's.
- Each model's `graph_metadata` gives every op a `receptive_field_size` (table §5). These values are smaller than the theoretical RFs (mixed3b's theoretical 5×5-branch RF is 91 px), so they're probably **effective/empirical RFs** **(inferred; method unconfirmed)**.
- Lucid itself has **no public dataset-example code**. Its `negative_neurons.ipynb` says: "constraints on ImageNet mean we can't provide an easy way for you to reproduce the dataset examples".
- License note from the About page: visualizations are CC-BY 4.0 **except dataset examples** (ImageNet-derived).

---

## 3. Transformation robustness and parameterization (why the tricks)

- **Feature Visualization** (Olah, Mordvintsev, Schubert; Distill 2017):
  - Naive pixel optimization yields high-frequency "optical illusion" noise. Three families of fixes: frequency penalization, **transformation robustness**, and learned priors.
  - Transformation robustness means stochastically jittering, rotating and scaling before each step.
  - **Preconditioning / parameterization**: run gradient descent in a decorrelated, whitened basis.
    - Spatial: the **Fourier basis with frequencies scaled to equal energy (≈1/f)**. This is Lucid's `fft_image` with `1/max(f, 1/max(w,h))`.
    - Color: "explicitly measure the correlation between colors in the training set" and decorrelate. The article says it uses a Cholesky decomposition; the code uses the SVD-sqrt matrix above.
    - The decorrelated color space plus a sigmoid keeps pixels in [0,1].
  - Padding (constant 0.5 gray) avoids edge artifacts.
  - The article also introduces diversity (a batch-diversity term), neuron interaction and negative visualizations, with linked colabs.
- **Building Blocks of Interpretability** (Olah et al., Distill 2018):
  - Composes feature visualization with attribution and the activation "cube" (neuron / spatial position / channel slices).
  - Uses semantic dictionaries, activation grids and spatial or channel attribution, and matrix factorization (NMF neuron groups).
  - Notes "dataset examples as another substrate". It's the source of Lucid's `building-blocks` notebooks.
  - Microscope did not expose these spatial/attribution views.
- **Distill Circuits / Early Vision (2020):** uses Lucid, with transformation robustness scaled by layer and L2 toward gray for low layers.

---

## 4. Site structure, hosting, and surviving assets

**Timeline:**
- Launched 2020-04-14 (OpenAI blog "OpenAI Microscope"; authors Ludwig Schubert, Michael Petrov, Shan Carter, Nick Cammarata, Gabriel Goh, Chris Olah).
- The blog text says both "eight vision model organisms" and "nine frequently studied vision models". The Jan-2021 `models` API lists 8.
- CLIP (`contrastive_*`) models were added around March 2021.
- The last Wayback 200 for the site is 2024-09. From 2025-01 it returns 503. **Today the DNS no longer resolves** (curl exit 6).

**Models** (`EncyclopediaGlobalState.models`, 2021-01): alexnet, alexnet_caffe_places365, inceptionv1, inceptionv1_caffe_places365, inceptionv3_slim, inceptionv4_slim, resnetv2_50_slim, vgg19_caffe. Later CDX captures also show contrastive_4x, contrastive_16x, contrastive_v1, contrastive_v2 and contrastive_rn50 (CLIP).

**Frontend:** a React SPA (create-react-app `static/js/main.*.chunk.js`). Routes:
- `/models` → model list
- `/models/{modelId}` → model page with graph diagram
- `/models/{modelId}/{opId}` → op/layer page
- `/models/{modelId}/{opId}/{unitId}` → unit page
- `/about`

`opId` is the TF tensor name with `:` → `_`. For example, **`https://microscope.openai.com/models/inceptionv1/mixed3b_0/379`** **(verified; About page and Distill links use this form)**. Unit titles were "Unit N". For InceptionV1 the UI flattened the aux-head branches out of the layout.

**API** (JSON, GET): `/api/encyclopedia_endpoints/call:{Class}.{method}?args=<urlencoded JSON [Class, [], method, [args...]]>`, e.g.
- `EncyclopediaGlobalState.models` / `.model(modelId)`: the op list, each op's `techniques` flags, and `graph_metadata`. `graph_metadata` holds, per op: `op_type`, `channels`, `default_size`, `receptive_field_size`, `conv_weight` and `branches` (channel slices of each concat), plus `weights` shapes, `layout` and `groups`.
- `FeatureVisEndpoints.op_feature_vis(model, op, "channel"|"neuron", 4096, repeat)` → `{channels:[{images:[{label, image:{url, type, shape:[224,224,3]}}]}]}`
- `DatasetExamplesEndpoints.op_dataset_example_crops(model, op, "imagenet"|"yfcc")` → `{examples:[{spritesheet:[4 urls], metadata:{url}}]}`
- `CaricatureEndpoints.op_caricature(model, op, image_name, 4096, cossim_pow)`, `DeepDreamEndpoints.op_deepdream(model, op, 1024)`, `TextFeatureVisEndpoints.op_text_feature_vis(model, op)`

**Assets were fully precomputed.** The API only returned URLs into blob storage. The About page says: "Generating the millions of images… requires running lots of distributed jobs".

**Blob hosting:**
- Original: `https://oaiggoh.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/{model}/{technique}/_{technique}/{urlencoded params}/{file}`. It **now returns 403 AuthorizationFailure**.
- **Live mirror**: replace the host with **`openaipublic.blob.core.windows.net`** (also `openaipublic.azureedge.net`). Checked:
  - md5 identical to the Wayback copy for 3b:379.
  - 200 for every one of a few dozen spot-checked InceptionV1 assets (conv2d2, mixed3a, mixed3b, mixed4a, mixed4c, mixed5b; feature vis repeats 0–2, neuron, dataset sheets, metadata JSON) and an AlexNet asset.
  - Container listing is disabled. I didn't check that every asset is present.
- The assets were generated on 2020-07-25, i.e. regenerated after the April launch. Whether the April 2020 images differed is **(unconfirmed)**.

**URL patterns (InceptionV1)**, with `B = https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1`:
- Feature vis (224×224 RGB PNG):
  `B/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3D{channel|neuron}%26op%3D{layer}%253A0%26repeat%3D{0|1|2}%26start%3D{s}%26steps%3D4096%26stop%3D{e}/channel-{u}.png`
  - `s = (u//32)*32` and `e = min(s+32, C)`. Examples: mixed4a (C=508) channel 500 → start 480, stop 508; `stop=512` gives a 404.
  - Small layers use one batch, e.g. mixed3a_5x5_bottleneck (16 ch): `start=0, stop=16`.
  - Note the **double-encoded** `%253A` (`:`) in `op`.
- Dataset examples: `B/lucid.dataset_examples/_dataset_examples/dataset%3D{imagenet|yfcc}%26op%3D{layer}%253A0/channel_{u}_{20|30|40|60}.png`, plus `channel_{u}.json`, which holds the 100 sorted activations.
- Caricature: `B/lucid.caricature/_caricature/cossim_pow%3D0.0%26image_name%3D{img}%26op%3D{layer}%253A0%26steps%3D4096/caricature.jpg` (and `{img}.jpg` = input)
- DeepDream: `B/lucid.deep_dream/_deep_dream/op%3D{layer}%253A0%26steps%3D1024/caricature.png`

**Working comparison URLs (verified HTTP 200 on 2026-10-04):**
- 3b:379 channel, repeat 0:
  https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dchannel%26op%3Dmixed3b%253A0%26repeat%3D0%26start%3D352%26steps%3D4096%26stop%3D384/channel-379.png
- 3b:379 channel, repeat 1:
  https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dchannel%26op%3Dmixed3b%253A0%26repeat%3D1%26start%3D352%26steps%3D4096%26stop%3D384/channel-379.png
- 3b:379 neuron, repeat 0:
  https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dneuron%26op%3Dmixed3b%253A0%26repeat%3D0%26start%3D352%26steps%3D4096%26stop%3D384/channel-379.png
- 4a:0 channel, repeat 0:
  https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dchannel%26op%3Dmixed4a%253A0%26repeat%3D0%26start%3D0%26steps%3D4096%26stop%3D32/channel-0.png
- 4a:97 channel, repeat 0 (repeat 2 also 200):
  https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dchannel%26op%3Dmixed4a%253A0%26repeat%3D0%26start%3D96%26steps%3D4096%26stop%3D128/channel-97.png
- 4a:500 channel, repeat 0 (last batch, `stop=508`):
  https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dchannel%26op%3Dmixed4a%253A0%26repeat%3D0%26start%3D480%26steps%3D4096%26stop%3D508/channel-500.png
- 4c:447 (car) channel, repeat 0:
  https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dchannel%26op%3Dmixed4c%253A0%26repeat%3D0%26start%3D416%26steps%3D4096%26stop%3D448/channel-447.png
- All 10 mixed3b curve detectors (379, 406, 385, 343, 342, 388, 340, 330, 349, 324) return 200 with `start=(u//32)*32`, `stop=start+32`.
- Dataset examples for 3b:379 (sheet + activations):
  https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.dataset_examples/_dataset_examples/dataset%3Dimagenet%26op%3Dmixed3b%253A0/channel_379_40.png
  https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.dataset_examples/_dataset_examples/dataset%3Dimagenet%26op%3Dmixed3b%253A0/channel_379.json
- Dataset examples for mixed4a: `channel_{0,26,97,476}_40.png` (same pattern with `op%3Dmixed4a%253A0`), and `dataset%3Dyfcc` for 4a:0.
- The Wayback Machine also holds copies of 476/480 mixed3b, 512 mixed4b and ~182 mixed4c channel images (repeat 0) under the `oaiggoh` URLs. Prefix them with `https://web.archive.org/web/<ts>id_/`. It has **no mixed4a**.

**Secondary reference set: Distill/OpenAI "model-visualizer", May 2019** (publicly listable GCS):
- `https://storage.googleapis.com/storage/v1/b/distill-circuits/o?prefix=model-visualizer/1556758232/InceptionV1/feature_visualization/` (1556758232 = 2019-05-02)
- Every InceptionV1 layer (conv2d0…mixed5b, head0/1_bottleneck) × {channel, neuron} × negative {False, True}, plus alpha=True neuron.
- Files: `channel_index={u}.png` (224×224), `manifest.json` (per-unit loss curve, **768 steps**) and `spritesheet.{png,jpg}`.
- URL form: `https://distill-circuits.storage.googleapis.com/model-visualizer%2F1556758232%2FInceptionV1%2Ffeature_visualization%2Falpha%3DFalse%26layer_name%3Dmixed4a%26negative%3DFalse%26objective_name%3Dchannel%2Fchannel_index%3D97.png`
- These are **not** the Microscope renders: fewer steps, and they fill the frame without the gray border.

**Distill Circuits assets:**
- `https://distill.pub/2020/circuits/early-vision/images/neuron/{layer}_{u}.jpg` (`.png` for conv2d*). These are **60×60 center crops of the 2019 neuron-objective renders**. Per `distill-circuits/inceptionv1-weight-explorer/download_and_cropy_imgs.py`: crop width W = 60 for mixed3a/3b (offset +5), 100 for mixed4a (offset +10) and 110/120/130 for 4b/4c/4d. They're too small for pixel comparison.
- Early Vision has all 480 mixed3b and 256 mixed3a units. Curve Detectors also has 14 mixed4a units: 26, 50, 58, 62, 95, 168, 403, 406, 407, 415, 424, 428, 436 and 437.
- Weight-explorer pages: `https://storage.googleapis.com/distill-circuits/inceptionv1-weight-explorer/{layer}_{u}.html`.

**Curve detectors** (Distill "Curve Detectors" and "Early Vision"):
- **mixed3b: 379, 406, 385, 343, 342, 388, 340, 330, 349, 324**
- mixed3a: 81, 104, 92, 145, 95, 163, 171, 71, 147, 189, 137
- conv2d2: 32, 171, 111, 80, 140, 180, 39, 130, 63, 182, 117
- Other units the About page names: dog-head detectors 4b:409, 4b:418 and 4c:447. The blog names a car (4c:447), a wheel (4b:373) and a window (4b:237).

---

## 5. Modelzoo naming, channel counts, lucent weights

Microscope `graph_metadata` (224 input). Lucid `InceptionV1.layers` agrees on depths.

| Layer (Microscope op) | Channels | Spatial | Microscope `receptive_field_size` | Concat branch slices |
|---|---|---|---|---|
| conv2d0 | 64 | 112 | 6×6 | |
| conv2d1 | 64 | 56 | 10×10 | |
| conv2d2 | 192 | 56 | 18×18 | |
| mixed3a | 256 | 28 | 49×49 | 1x1 0–64, 3x3 64–192, 5x5 192–224, pool 224–256 |
| mixed3b | 480 | 28 | 55×57 | 1x1 0–128, 3x3 128–320, 5x5 320–416, pool 416–480 |
| mixed4a | **508** | 14 | 81×77 | 1x1 0–192, 3x3 192–396, 5x5 396–444, pool 444–508 |
| mixed4b | 512 | 14 | 103×107 | 0–160, 160–384, 384–448, 448–512 |
| mixed4c | 512 | 14 | 117×127 | 0–128, 128–384, 384–448, 448–512 |
| mixed4d | 528 | 14 | 162×159 | 0–112, 112–400, 400–464, 464–528 |
| mixed4e | 832 | 14 | 189×169 | 0–256, 256–576, 576–704, 704–832 |
| mixed5a | 832 | 7 | 217×214 | |
| mixed5b | 1024 | 7 | 215×219 | |
| head0_bottleneck / head1_bottleneck | 128 | 4 | | |
| softmax0/1/2 | 1008 (ImageNet_alternate labels) | | | |

- Microscope also exposed every branch op (`mixed4a_1x1`, `_3x3_bottleneck`, `_3x3`, `_5x5_bottleneck`, `_5x5`, `_pool_reduce`), `add:0` (the input after the value-range transform, **(inferred)**), and `nn0`/`nn1`.
- `mixedXX:0` is the **post-ReLU Concat**. Lucid also synthesizes `mixedXX_pre_relu`.
- Lucent's hook names match: `mixed4a` is the `CatLayer` of the post-ReLU branches, and the branches are `mixed4a_1x1`, `mixed4a_3x3_pre_relu_conv`, etc.
- Lucent has **no concatenated pre-ReLU layer**.
- **Weights:**
  - Lucid's `InceptionV1.pb` docstring: "The weights were trained at Google and released in an early TensorFlow tutorial" (the DeepDream tutorial's `inception5h.zip`).
  - lucent's `inceptionv1(pretrained=True)` loads `https://github.com/ProGamerGov/pytorch-old-tensorflow-models/raw/master/inception5h.pth`. That repo lists its source as `storage.googleapis.com/download.tensorflow.org/models/inception5h.zip`, and lucent's model code is ProGamerGov's `neural-dream` `inception5h.py`.
  - So it's the **same network and weights as Lucid's InceptionV1** **(inferred; conversion fidelity unconfirmed)**.
  - Input: RGB, `x*255-117`. Output: `softmax2` with 1008 classes.
  - The Circuits authors warn that "five versions of InceptionV1" exist, and this is the canonical one. It is **not** torchvision GoogLeNet or TF-Slim InceptionV1.
- **LRN bug in lucent's port (confirmed by measurement, 2026-10-04):**
  - `InceptionV1.py` calls `F.local_response_norm(x, size=9, alpha=1e-4, beta=0.5, k=1)` with TF's alpha. PyTorch computes `alpha/size * sum(x²)`; TF's `lrn` computes `alpha * sum(x²)`. So the LRN is 9× too weak.
  - Effect: on the top val image for mixed4a units 0, 3, 97, 120, 150, 200 and 500, lucent's max activation is 2.5–3.35× Microscope's `channel_<u>.json` value.
  - With alpha × 9 the ratio is 0.83–1.12.
  - ImageNet accuracy is the same either way: 64.8% vs 64.6% top-1 and 86.8% top-5 on 2,000 val images, measured with the 1008-way head and `ImageNet_alternate` labels.
  - `microscope/model.py` patches this by default (`load_model(fix_lrn=True)`).

---

## Unconfirmed
- That Microscope ran `lucid/recipes/feature_visualization.py` exactly: the transform list, `crop_or_pad_to`, relu override and lr are inferred from parameter-name matching. Only `steps=4096`, `alpha=False`, `negative=False`, the objective names, 3 repeats and batches of 32 are confirmed by paths.
- The cause of the gray border / dark frame in the 2020 renders.
- Dataset examples:
  - top-100 distinct images vs positions
  - ImageNet split
  - image resizing before the forward pass (224? aspect handling?)
  - whether YFCC was a 100M subset
  - how ties and duplicates were handled
- How `receptive_field_size` was computed.
- Synthetic Tuning stimuli and data format: no archived InceptionV1 response found.
- Whether the April 2020 launch assets equal the 2020-07-25 assets.
- Completeness of the `openaipublic` mirror. Only spot-checked, but no misses so far.
- TF1 `tf.contrib.image.rotate` default interpolation (NEAREST) is from memory of the TF1 API, not re-checked.
- Fidelity of the ProGamerGov inception5h conversion for padding and pooling. **LRN is resolved; see the note at the end of §5.**

## Sources
Useful:
- Lucent installed source: `.venv/lib/python3.12/site-packages/lucent/` (optvis/render.py, transform.py, param/{images,spatial,color}.py, objectives.py, modelzoo/inceptionv1/*). Useful.
- https://github.com/greentfrapp/lucent (cloned, master 2ccc434). Useful (diffed against 0.1.8).
- https://github.com/tensorflow/lucid (cloned): optvis/render.py, transform.py, param/{spatial,color,images}.py, objectives.py, objectives_util.py, overrides/redirected_relu_grad.py, misc/redirected_relu_grad.py, modelzoo/other_models/InceptionV1.py, modelzoo/vision_base.py, **recipes/feature_visualization.py**, recipes/caricature.py, notebooks/feature-visualization/negative_neurons.ipynb. Useful.
- GitHub API commits for lucid/recipes/feature_visualization.py. Useful (author/date).
- Wayback CDX `http(s)://web.archive.org/cdx/search/cdx?url=microscope.openai.com*`: 6,274 URLs. Useful. (The plain-http CDX endpoint later started failing; https worked.)
- Wayback snapshots: `microscope.openai.com/static/js/main.7ef1d588.chunk.js` (2021-03-06) and API JSON (`EncyclopediaGlobalState.models/model`, `FeatureVisEndpoints.op_feature_vis`, `DatasetExamplesEndpoints.op_dataset_example_crops`, `CaricatureEndpoints`, `DeepDreamEndpoints`). Useful.
- Wayback CDX `oaiggoh.blob.core.windows.net/microscopeprod/…` (feature_vis 1,258 rows; dataset_examples 27 rows; caricature; text JSON). Useful.
- https://oaiggoh.blob.core.windows.net/microscopeprod/… Dead (403).
- **https://openaipublic.blob.core.windows.net/microscopeprod/…** and https://openaipublic.azureedge.net/microscopeprod/… Live, very useful.
- https://web.archive.org/web/20200415011535/https://openai.com/blog/microscope/. Useful (blog text).
- https://openai.com/index/microscope/. 403 to fetcher (not useful).
- https://distill.pub/2017/feature-visualization/ and /appendix/. Useful.
- https://distill.pub/2018/building-blocks/. Useful (context).
- https://distill.pub/2020/circuits/zoom-in/. Useful (context).
- https://distill.pub/2020/circuits/early-vision/. Useful (unit lists, method notes, thumbnails).
- https://distill.pub/2020/circuits/curve-detectors/. Useful (curve units, "receptive-field sized crops").
- https://storage.googleapis.com/storage/v1/b/distill-circuits/o (public listing), `model-visualizer/1556758232/InceptionV1/...` and `inceptionv1-weight-explorer/{download_and_cropy_imgs.py,generate_weight_pages.py,*.html}`. Useful.
- https://github.com/ProGamerGov/pytorch-old-tensorflow-models. Useful (inception5h source).

Dead / not useful:
- https://microscope.openai.com/ (DNS fails; 503 in Wayback since 2025-01)
- https://storage.googleapis.com/clarity-public/colah/experiments/aprox_weights_1/… (404)
- https://openai-clarity.storage.googleapis.com/model-visualizer/… (404)
- https://openaipublic.blob.core.windows.net/microscopeprod?restype=container&comp=list (listing disabled)
- https://distill.pub/2020/circuits/early-vision/images/neuron/*.png for mixed layers (404; they're .jpg)
