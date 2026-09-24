# SAM3 Segmentation Service

Runs Meta's [SAM 3](https://huggingface.co/facebook/sam3) on images and exposes it both as a **CLI**
and as a **FastAPI HTTP service**. Both prompt styles are supported:

- **Text / box prompts** — *Promptable Concept Segmentation* (`Sam3Model`, 840.4 M params): find **all**
  instances matching a concept, or segment what a box points at. Endpoint: `POST /segment`.
- **Point prompts** — *Promptable Visual Segmentation* (`Sam3TrackerModel`, 458.3 M params): click a
  point (positive/negative) to segment **that specific** object. Endpoint: `POST /segment/point`.

Device is auto-detected: **CUDA → MPS → CPU**. `dtype` defaults to `bfloat16` on GPU (falls back to
`float16` on pre-Ampere cards) and `float32` on CPU/MPS.

The tracker is loaded **lazily** on the first point request, so a text-only deployment never pays its
memory cost.

## Layout

```
sam3/           downloaded model weights (gitignored)
sam3_api/
  engine.py     Sam3Engine: device/dtype resolution + inference
  server.py     FastAPI app (endpoints)
  schemas.py    response models
predict.py      one-shot CLI
serve.py        API server entry point
```

## Setup

### 1. Create the env and install PyTorch for your hardware

```bash
uv venv

# CPU only
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (CUDA 12.4) — use the wheel that matches your driver
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install the remaining dependencies

```bash
uv pip install -e .     # torch is already satisfied, so it will not be reinstalled
```

### 3. Download the model

```bash
modelscope download --model facebook/sam3 --local_dir ./sam3 \
  config.json model.safetensors processor_config.json tokenizer.json \
  tokenizer_config.json special_tokens_map.json vocab.json merges.txt configuration.json
```

`sam3.pt` in the repo is the same weights in a different format — skip it to save ~3.4 GB.

## CLI

```bash
python predict.py data/turn00_front.jpg --prompt "water bottle" --save out.png
python predict.py data/turn00_front.jpg --boxes "[[120,40,240,245]]"
python predict.py data/turn00_front.jpg --prompt "box" --device cuda --dtype bfloat16 --runs 3
```

Overrides: `--device {auto,cuda,mps,cpu}`, `--dtype {auto,bfloat16,float16,float32}`,
`--model-dir`, `--runs`, `--save`. Env equivalents: `SAM3_DEVICE`, `SAM3_DTYPE`, `SAM3_MODEL_DIR`.

## API server

```bash
python serve.py                 # http://0.0.0.0:8000
```

Env vars: `SAM3_HOST`, `SAM3_PORT` (default 8000), `SAM3_MODEL_DIR` (default `./sam3`),
`SAM3_DEVICE`, `SAM3_DTYPE`, and optional `SAM3_API_KEY` (when set, requests must send
`Authorization: Bearer <key>`).

Interactive docs: `http://localhost:8000/docs`.

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"device":"cuda","dtype":"bfloat16","params_m":840.4,"model_dir":"./sam3"}
```

### `POST /segment`

`multipart/form-data`:

| field            | type   | default | description                                            |
| ---------------- | ------ | ------- | ------------------------------------------------------ |
| `image`          | file   | —       | image to segment (required)                            |
| `prompt`         | text   | —       | text concept, e.g. `water bottle`                      |
| `boxes`          | JSON   | —       | `[[x1,y1,x2,y2], ...]` (positive/negative via labels)  |
| `box_labels`     | JSON   | all `1` | `[1,0,...]`, `1` = positive, `0` = negative            |
| `threshold`      | float  | `0.5`   | detection score threshold                              |
| `mask_threshold` | float  | `0.5`   | mask binarization threshold                            |
| `return_masks`   | bool   | `true`  | include per-object mask as base64 PNG                  |
| `return_overlay` | bool   | `true`  | include a combined colored overlay as base64 PNG       |

At least one of `prompt` / `boxes` is required.

```bash
# text prompt
curl -X POST http://localhost:8000/segment \
  -F "image=@data/turn00_front.jpg" \
  -F "prompt=water bottle" -o seg.json

# box prompt, boxes only (no masks/overlay)
curl -X POST http://localhost:8000/segment \
  -F "image=@data/turn00_front.jpg" \
  -F "boxes=[[120,40,240,245]]" \
  -F "return_masks=false" -F "return_overlay=false"
```

Response:

```json
{
  "device": "cuda", "dtype": "bfloat16", "params_m": 840.4,
  "width": 640, "height": 480, "inference_ms": 148.2, "count": 1,
  "objects": [{ "score": 0.9667, "box": [133.6, 49.3, 228.7, 235.3], "mask_png_b64": "..." }],
  "overlay_png_b64": "..."
}
```

Python client:

```python
import requests, base64

with open("data/turn00_front.jpg", "rb") as fh:
    r = requests.post(
        "http://localhost:8000/segment",
        files={"image": fh},
        data={"prompt": "water bottle"},
    )
r.raise_for_status()
out = r.json()
print(out["count"], out["objects"][0]["box"])
open("overlay.png", "wb").write(base64.b64decode(out["overlay_png_b64"]))
```

### `POST /segment/point`

Point-prompt segmentation via `Sam3TrackerModel`. `multipart/form-data`:

| field            | type   | default | description                                                       |
| ---------------- | ------ | ------- | ----------------------------------------------------------------- |
| `image`          | file   | —       | image to segment (required)                                       |
| `points`         | JSON   | —       | one entry per object: `[[[x,y],[x2,y2]], [[x,y]]]` (required)     |
| `labels`         | JSON   | all `1` | `1` = positive point, `0` = negative; must match `points` shape    |
| `return_masks`   | bool   | `true`  | include per-object mask as base64 PNG                             |
| `return_overlay` | bool   | `true`  | include a combined colored overlay as base64 PNG                  |

The tracker is loaded on the first call to this endpoint.

```bash
# single object, single click
curl -X POST http://localhost:8000/segment/point \
  -F "image=@data/turn00_front.jpg" \
  -F "points=[[[180,140]]]" -o point.json

# two objects at once
curl -X POST http://localhost:8000/segment/point \
  -F "image=@data/turn00_front.jpg" \
  -F "points=[[[180,140]],[[320,200]]]" -o point.json
```

Response (no confidence score — the tracker segments the pointed instance):

```json
{
  "device": "cuda", "dtype": "bfloat16", "params_m": 458.3,
  "width": 640, "height": 480, "inference_ms": 80.1, "count": 1,
  "objects": [{ "box": [134.0, 49.0, 227.0, 234.0], "mask_png_b64": "..." }],
  "overlay_png_b64": "..."
}
```

## Performance

Measured on this machine (Intel i5-1345U, 10 threads, **CPU only, float32**):

| model             | params  | endpoint          | per-image |
| ----------------- | ------- | ----------------- | --------- |
| `Sam3Model`       | 840.4 M | `/segment`        | ~25-31 s  |
| `Sam3TrackerModel`| 458.3 M | `/segment/point`  | ~22 s     |

Model load ~1-1.5 s each. On a CUDA GPU with `bfloat16` this drops to the sub-second range; run
`predict.py --runs 3` on the target box to get the real number.

Memory: both models resident cost ~5.2 GB in `float32` (CPU) or ~2.6 GB in `bfloat16` (GPU). The
tracker is only loaded once a point request arrives.

Requests are serialized behind a single model instance and a lock. For higher throughput, run
multiple server processes behind a reverse proxy, or batch on the client.

## Notes

- `predict.py` covers text and box prompts; point prompts are only exposed over the API
  (`POST /segment/point`).
- Masks are returned as base64 PNG (one per object). Set `return_masks=false` when only boxes and
  scores are needed to cut payload size.
- Box prompts are limited to positive/negative labels; point prompts do not return confidence scores
  since the tracker segments the indicated instance rather than detecting a concept.
