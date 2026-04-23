from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import COGVIEW_DIFFUSERS_SRC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: generate a 2D image with CogView3.")
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--model-path", default="THUDM/CogView3-Plus-3B", type=str)
    parser.add_argument("--guidance-scale", default=7.0, type=float)
    parser.add_argument("--num-images-per-prompt", default=1, type=int)
    parser.add_argument("--num-inference-steps", default=50, type=int)
    parser.add_argument("--width", default=1024, type=int)
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # CogView3 expects a newer diffusers implementation than the shared environment.
    sys.path.insert(0, str(COGVIEW_DIFFUSERS_SRC))

    import torch
    from diffusers import CogView3PlusPipeline

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    pipe = CogView3PlusPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    image = pipe(
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
    ).images[0]
    image.save(args.output_path)
    print(f"Saved CogView image to {args.output_path}")


if __name__ == "__main__":
    main()
