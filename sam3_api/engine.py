import base64
import io
import os
import threading
import time

import numpy as np
import torch
from PIL import Image
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

PALETTE = [
    (255, 0, 0),
    (0, 160, 0),
    (0, 90, 255),
    (255, 160, 0),
    (170, 0, 255),
    (0, 200, 200),
]


def resolve_device(preferred: str | None = None) -> str:
    preferred = (preferred or os.getenv("SAM3_DEVICE") or "auto").lower()
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, preferred: str | None = None) -> torch.dtype:
    preferred = (preferred or os.getenv("SAM3_DTYPE") or "auto").lower()
    if preferred != "auto":
        return getattr(torch, preferred)
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _mask_to_png_b64(mask: np.ndarray) -> str:
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _overlay(image: Image.Image, masks: list[np.ndarray]) -> Image.Image:
    base = image.convert("RGBA")
    for i, mask in enumerate(masks):
        layer = Image.new("RGBA", base.size, PALETTE[i % len(PALETTE)] + (0,))
        layer.putalpha(Image.fromarray((mask > 0).astype(np.uint8) * 128, mode="L"))
        base = Image.alpha_composite(base, layer)
    return base.convert("RGB")


class Sam3Engine:
    """Device-agnostic SAM3 wrapper. Runs on CUDA, MPS or CPU."""

    def __init__(self, model_dir: str = "./sam3", device: str | None = None, dtype: str | None = None):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        self.model_dir = model_dir
        self._lock = threading.Lock()
        self._tracker_init_lock = threading.Lock()
        self._tracker = None
        self._tracker_processor = None
        self._tracker_params = None

        self.model = Sam3Model.from_pretrained(
            model_dir, dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_dir)
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def info(self) -> dict:
        return {
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "params_m": round(self.n_params / 1e6, 1),
            "model_dir": self.model_dir,
            "tracker_loaded": self._tracker is not None,
        }

    @torch.inference_mode()
    def segment(
        self,
        image: Image.Image,
        text: str | None = None,
        boxes: list[list[float]] | None = None,
        box_labels: list[int] | None = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        return_masks: bool = True,
        return_overlay: bool = True,
    ) -> dict:
        if text is None and boxes is None:
            raise ValueError("at least one of `text` or `boxes` is required")

        call_kwargs: dict = {"images": image, "return_tensors": "pt"}
        if text is not None:
            call_kwargs["text"] = text
        if boxes is not None:
            call_kwargs["input_boxes"] = [boxes]
            call_kwargs["input_boxes_labels"] = [box_labels or [1] * len(boxes)]

        inputs = self.processor(**call_kwargs).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with self._lock:
            t0 = time.perf_counter()
            with torch.autocast(
                device_type="cuda", enabled=self.device.startswith("cuda") and self.dtype != torch.float32
            ):
                outputs = self.model(**inputs)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - t0) * 1000

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = [
            (m.cpu().numpy() > 0).astype(np.uint8)
            for m in results["masks"]
        ]
        objects = []
        for i, (score, box) in enumerate(zip(results["scores"].tolist(), results["boxes"].tolist())):
            obj = {
                "score": round(float(score), 4),
                "box": [round(float(v), 2) for v in box],
            }
            if return_masks:
                obj["mask_png_b64"] = _mask_to_png_b64(masks[i])
            objects.append(obj)

        out = {
            **self.info(),
            "width": image.size[0],
            "height": image.size[1],
            "inference_ms": round(inference_ms, 1),
            "count": len(objects),
            "objects": objects,
        }
        if return_overlay:
            out["overlay_png_b64"] = _encode_image_b64(_overlay(image, masks))
        return out

    def _ensure_tracker(self):
        with self._tracker_init_lock:
            if self._tracker is None:
                self._tracker = (
                    Sam3TrackerModel.from_pretrained(
                        self.model_dir, dtype=self.dtype, low_cpu_mem_usage=True
                    )
                    .to(self.device)
                    .eval()
                )
                self._tracker_processor = Sam3TrackerProcessor.from_pretrained(self.model_dir)
                self._tracker_params = sum(p.numel() for p in self._tracker.parameters()) / 1e6
        return self._tracker, self._tracker_processor

    @torch.inference_mode()
    def segment_points(
        self,
        image: Image.Image,
        points: list[list[list[float]]],
        labels: list[list[int]] | None = None,
        return_masks: bool = True,
        return_overlay: bool = True,
    ) -> dict:
        if not points:
            raise ValueError("`points` must be a non-empty list of objects, each a list of [x, y]")

        model, processor = self._ensure_tracker()
        labels = labels or [[1] * len(obj) for obj in points]

        inputs = processor(
            images=image,
            input_points=[points],
            input_labels=[labels],
            return_tensors="pt",
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        with self._lock:
            t0 = time.perf_counter()
            with torch.autocast(
                device_type="cuda", enabled=self.device.startswith("cuda") and self.dtype != torch.float32
            ):
                outputs = model(**inputs, multimask_output=False)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - t0) * 1000

        masks_t = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = [(m[0].numpy() > 0).astype(np.uint8) for m in masks_t]

        objects = []
        for mask in masks:
            obj = {"box": _bbox_from_mask(mask)}
            if return_masks:
                obj["mask_png_b64"] = _mask_to_png_b64(mask)
            objects.append(obj)

        out = {
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "params_m": round(self._tracker_params, 1),
            "width": image.size[0],
            "height": image.size[1],
            "inference_ms": round(inference_ms, 1),
            "count": len(objects),
            "objects": objects,
        }
        if return_overlay:
            out["overlay_png_b64"] = _encode_image_b64(_overlay(image, masks))
        return out


def _encode_image_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
