import io
import json
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool

from .engine import Sam3Engine
from .schemas import HealthResponse, PointResponse, SegmentResponse

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("SAM3_MODEL_DIR", "./models/sam3")
    STATE["engine"] = Sam3Engine(model_dir)
    info = STATE["engine"].info()
    print(f"[sam3-api] model loaded: device={info['device']} dtype={info['dtype']} params={info['params_m']}M")
    yield
    STATE.clear()


app = FastAPI(title="SAM3 Segmentation API", version="0.1.0", lifespan=lifespan)


def require_key(authorization: str | None = Header(None)) -> None:
    expected = os.getenv("SAM3_API_KEY")
    if not expected:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine: Sam3Engine | None = STATE.get("engine")
    if engine is None:
        return HealthResponse(status="loading", model_loaded=False)
    info = engine.info()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=info["device"],
        dtype=info["dtype"],
        params_m=info["params_m"],
        model_dir=info["model_dir"],
        tracker_loaded=info["tracker_loaded"],
    )


def _parse_boxes(boxes: str | None, box_labels: str | None) -> tuple[list[list[float]] | None, list[int] | None]:
    if boxes is None:
        return None, None
    try:
        parsed = json.loads(boxes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"`boxes` must be JSON: {exc}") from exc
    if not isinstance(parsed, list) or not parsed or not all(len(b) == 4 for b in parsed):
        raise HTTPException(status_code=422, detail="`boxes` must be a non-empty list of [x1,y1,x2,y2]")
    labels = json.loads(box_labels) if box_labels else [1] * len(parsed)
    if len(labels) != len(parsed):
        raise HTTPException(status_code=422, detail="`box_labels` length must match `boxes`")
    return parsed, labels


@app.post("/segment", response_model=SegmentResponse, dependencies=[Depends(require_key)])
async def segment(
    image: UploadFile = File(..., description="image file (jpg/png)"),
    prompt: str | None = Form(None, description="text concept, e.g. 'water bottle'"),
    boxes: str | None = Form(None, description='JSON list of boxes, e.g. "[[10,20,100,200]]"'),
    box_labels: str | None = Form(None, description='JSON list of 1/0, e.g. "[1]"'),
    threshold: float = Form(0.5),
    mask_threshold: float = Form(0.5),
    return_masks: bool = Form(True),
    return_overlay: bool = Form(True),
) -> SegmentResponse:
    engine: Sam3Engine | None = STATE.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="model still loading")

    parsed_boxes, labels = _parse_boxes(boxes, box_labels)
    if prompt is None and parsed_boxes is None:
        raise HTTPException(status_code=422, detail="provide `prompt` and/or `boxes`")

    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not read image: {exc}") from exc

    result = await run_in_threadpool(
        engine.segment,
        pil_image,
        text=prompt,
        boxes=parsed_boxes,
        box_labels=labels,
        threshold=threshold,
        mask_threshold=mask_threshold,
        return_masks=return_masks,
        return_overlay=return_overlay,
    )
    return SegmentResponse(**result)


def _parse_points(points: str, labels: str | None) -> tuple[list[list[list[float]]], list[list[int]] | None]:
    try:
        parsed = json.loads(points)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"`points` must be JSON: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise HTTPException(status_code=422, detail="`points` must be a non-empty list of objects")
    for obj in parsed:
        if (
            not isinstance(obj, list)
            or not obj
            or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in obj)
        ):
            raise HTTPException(status_code=422, detail="each object must be a non-empty list of [x, y] points")

    if labels is None:
        return parsed, None
    try:
        parsed_labels = json.loads(labels)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"`labels` must be JSON: {exc}") from exc
    if len(parsed_labels) != len(parsed) or any(len(a) != len(b) for a, b in zip(parsed_labels, parsed)):
        raise HTTPException(status_code=422, detail="`labels` must match the shape of `points`")
    return parsed, parsed_labels


@app.post("/segment/point", response_model=PointResponse, dependencies=[Depends(require_key)])
async def segment_point(
    image: UploadFile = File(..., description="image file (jpg/png)"),
    points: str = Form(..., description='JSON objects of points, e.g. "[[[180,140]]]" = 1 object with 1 point'),
    labels: str | None = Form(None, description='JSON 1/0 per point, e.g. "[[1]]" (1 = positive, 0 = negative)'),
    return_masks: bool = Form(True),
    return_overlay: bool = Form(True),
) -> PointResponse:
    engine: Sam3Engine | None = STATE.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="model still loading")

    parsed_points, parsed_labels = _parse_points(points, labels)

    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not read image: {exc}") from exc

    result = await run_in_threadpool(
        engine.segment_points,
        pil_image,
        points=parsed_points,
        labels=parsed_labels,
        return_masks=return_masks,
        return_overlay=return_overlay,
    )
    return PointResponse(**result)
