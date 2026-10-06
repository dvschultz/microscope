#!/usr/bin/env bash
# Run precompute on a remote CUDA box and pull the assets back.
#
#   AI_HOST=user@gpu-host ./remote.sh setup          # copy code + dataset, build venv
#   AI_HOST=user@gpu-host ./remote.sh run mixed4a    # start a layer in the background (nohup)
#   AI_HOST=user@gpu-host ./remote.sh status mixed4a # tail the log
#   AI_HOST=user@gpu-host ./remote.sh pull           # rsync site/assets + cache back
#   AI_HOST=... ./remote.sh push                         # code only (*.py), no data or assets
#   AI_HOST=... MODEL=vgg19 ./remote.sh queue conv5_4 conv5_3   # MODEL picks a microscope/models.py entry
#   AI_HOST=... ./remote.sh pull-model vgg19             # only site/assets/vgg19 + cache/vgg19 back
#
# CUDA_VISIBLE_DEVICES picks the GPU (default 0). The run survives SSH disconnects.
set -euo pipefail
: "${AI_HOST:?set AI_HOST=user@host}"
DIR="${REMOTE_DIR:-microscope}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-inceptionv1}"
LOGDIR=cache; [ "$MODEL" = inceptionv1 ] || LOGDIR="cache/$MODEL"
cd "$(dirname "$0")"

case "${1:-}" in
  setup)
    # Assets go too, so channels already rendered locally are skipped remotely.
    rsync -az --exclude .venv --exclude .git --exclude '__pycache__' ./ "$AI_HOST:$DIR/"
    # BASE_PYTHON: an existing python with CUDA torch + torchvision. The venv inherits it
    # (--system-site-packages) instead of downloading ~4 GB of wheels; nothing is added to the base env.
    BASE="${BASE_PYTHON:-\$HOME/anaconda3/envs/comfy/bin/python}"
    ssh "$AI_HOST" "cd $DIR && [ -x .venv/bin/python ] || $BASE -m venv --system-site-packages .venv; \
      .venv/bin/pip install -q --no-deps torch-lucent kornia==0.4.1 && .venv/bin/pip install -q pyarrow tqdm && \
      .venv/bin/python -c 'import torch, torchvision, lucent; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'"
    ;;
  run)
    L="${2:?layer}"; shift 2
    ssh -f -n "$AI_HOST" "cd $DIR && mkdir -p $LOGDIR && CUDA_VISIBLE_DEVICES=$GPU setsid nohup .venv/bin/python -u precompute.py --model $MODEL --layer $L $* \
      > $LOGDIR/run_$L.log 2>&1 < /dev/null & echo started $MODEL $L"
    ;;
  queue)
    # several layers back to back on one GPU: CUDA_VISIBLE_DEVICES=1 ./remote.sh queue mixed5a mixed4d
    shift; LAYERS="$*"
    ssh -f -n "$AI_HOST" "cd $DIR && mkdir -p $LOGDIR && CUDA_VISIBLE_DEVICES=$GPU setsid nohup bash -c \
      'for L in $LAYERS; do .venv/bin/python -u precompute.py --model $MODEL --layer \$L --batch ${BATCH:-64} > $LOGDIR/run_\$L.log 2>&1; done' \
      > /dev/null 2>&1 < /dev/null & echo queued $MODEL on GPU $GPU: $LAYERS"
    ;;
  status)
    ssh "$AI_HOST" "tail -c 600 $DIR/$LOGDIR/run_${2:?layer}.log | tr '\r' '\n' | tail -4; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
    ;;
  pull)
    rsync -az "$AI_HOST:$DIR/site/assets/" site/assets/
    rsync -az "$AI_HOST:$DIR/cache/timings.json" "$AI_HOST:$DIR/cache/run_*.log" cache/
    .venv/bin/python precompute.py --layer manifest --device cpu
    ;;
  push)
    rsync -az --relative ./*.py ./microscope/*.py ./remote.sh "$AI_HOST:$DIR/"
    ;;
  pull-model)
    M="${2:?model}"
    [ "$M" != inceptionv1 ] || { echo "use pull for inceptionv1"; exit 1; }
    rsync -az "$AI_HOST:$DIR/site/assets/$M/" "site/assets/$M/"
    rsync -az --exclude 'topk_*' --exclude 'attr_*' --exclude 'text_*.npy' "$AI_HOST:$DIR/cache/$M/" "cache/$M/"
    .venv/bin/python precompute.py --layer manifest --device cpu
    ;;
  *) sed -n 2,13p "$0"; exit 1 ;;
esac
