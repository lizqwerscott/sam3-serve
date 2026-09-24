from pydantic import BaseModel


class ObjectOut(BaseModel):
    score: float
    box: list[float]
    mask_png_b64: str | None = None


class SegmentResponse(BaseModel):
    device: str
    dtype: str
    params_m: float
    width: int
    height: int
    inference_ms: float
    count: int
    objects: list[ObjectOut]
    overlay_png_b64: str | None = None


class PointObjectOut(BaseModel):
    box: list[float]
    mask_png_b64: str | None = None


class PointResponse(BaseModel):
    device: str
    dtype: str
    params_m: float
    width: int
    height: int
    inference_ms: float
    count: int
    objects: list[PointObjectOut]
    overlay_png_b64: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
    dtype: str | None = None
    params_m: float | None = None
    model_dir: str | None = None
    tracker_loaded: bool | None = None
