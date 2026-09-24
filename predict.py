import argparse
import base64
import json
import time

from PIL import Image

from sam3_api.engine import Sam3Engine


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot SAM3 segmentation (GPU/CPU).")
    parser.add_argument("image", nargs="?", default="data/turn00_front.jpg")
    parser.add_argument("--prompt", default="water bottle", help="text concept")
    parser.add_argument("--boxes", default=None, help='JSON boxes, e.g. "[[10,20,100,200]]"')
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--dtype", default="auto", help="auto | bfloat16 | float16 | float32")
    parser.add_argument("--model-dir", default="./sam3")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--save", default=None, help="save the overlay image here")
    args = parser.parse_args()

    t0 = time.perf_counter()
    engine = Sam3Engine(args.model_dir, device=args.device, dtype=args.dtype)
    info = engine.info()
    print(f"[load] {time.perf_counter() - t0:.2f}s device={info['device']} dtype={info['dtype']} params={info['params_m']}M")

    image = Image.open(args.image).convert("RGB")
    boxes = json.loads(args.boxes) if args.boxes else None
    print(f"[input] {args.image} {image.size[0]}x{image.size[1]}")

    result = None
    for i in range(args.runs):
        result = engine.segment(image, text=args.prompt, boxes=boxes)
        print(f"[run {i + 1}/{args.runs}] {result['inference_ms'] / 1000:.3f} s")

    print(f"\n[done] count={result['count']}")
    for obj in result["objects"]:
        b = obj["box"]
        print(f"  score={obj['score']:.3f} box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")

    if args.save and result.get("overlay_png_b64"):
        with open(args.save, "wb") as fh:
            fh.write(base64.b64decode(result["overlay_png_b64"]))
        print(f"[save] overlay written to {args.save}")


if __name__ == "__main__":
    main()
