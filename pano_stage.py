from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from common import SD360_ROOT, WORKSPACE_ROOT


def create_white_mask(reference_image_path: Path, output_path: Path) -> Path:
    reference = Image.open(reference_image_path)
    white = Image.new("RGB", reference.size, "white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    white.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: convert a 2D image into a panorama.")
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--model-dir", default=WORKSPACE_ROOT / "SD-T2I-360PanoImage-main" / "models", type=Path)
    parser.add_argument("--mask-path", default=None, type=Path)
    parser.add_argument("--generated-mask-path", default=None, type=Path)
    parser.add_argument("--num-inference-steps", default=20, type=int)
    parser.add_argument("--guidance-scale", default=7.0, type=float)
    parser.add_argument("--controlnet-conditioning-scale", default=4.0, type=float)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--upscale", action="store_true", default=False)
    parser.add_argument("--refinement", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mask_path is None:
        generated_mask = args.generated_mask_path or args.output_path.parent / "white_mask.png"
        args.mask_path = create_white_mask(args.image_path, generated_mask)
        print(f"Created default white mask at {args.mask_path}")

    sys.path.insert(0, str(SD360_ROOT))

    import torch
    from diffusers.utils import load_image
    from img2panoimg import Image2360PanoramaImagePipeline

    image = load_image(str(args.image_path)).resize((512, 512))
    mask = load_image(str(args.mask_path))

    input_data = {
        "prompt": args.prompt,
        "image": image,
        "mask": mask,
        "upscale": args.upscale,
        "refinement": args.refinement,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
        "seed": args.seed,
    }

    pipeline = Image2360PanoramaImagePipeline(
        str(args.model_dir),
        torch_dtype=torch.float16,
    )
    output = pipeline(input_data)
    output.save(args.output_path)
    print(f"Saved panorama image to {args.output_path}")


if __name__ == "__main__":
    main()
