from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import COGVIEW_DIFFUSERS_SRC, DREAMSCENE_ROOT, POEM_ROOT, SD360_ROOT, default_run_name, ensure_dir, python_cmd, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full poem image -> panorama -> 3D pipeline.")
    parser.add_argument("--prompt", required=True, type=str, help="Base prompt for the whole pipeline.")
    parser.add_argument(
        "--panorama-prompt",
        default=None,
        type=str,
        help="Optional prompt override for the panorama stage. Defaults to --prompt.",
    )
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--runs-dir", default=POEM_ROOT / "runs", type=Path)

    parser.add_argument("--cogview-model-path", default="THUDM/CogView3-Plus-3B", type=str)
    parser.add_argument("--cogview-guidance-scale", default=7.0, type=float)
    parser.add_argument("--cogview-steps", default=50, type=int)
    parser.add_argument("--cogview-width", default=1024, type=int)
    parser.add_argument("--cogview-height", default=1024, type=int)
    parser.add_argument("--cogview-dtype", default="bfloat16", choices=["float16", "bfloat16"])

    parser.add_argument("--pano-model-dir", default=None, type=Path)
    parser.add_argument("--pano-mask-path", default=None, type=Path)
    parser.add_argument("--pano-steps", default=20, type=int)
    parser.add_argument("--pano-guidance-scale", default=7.0, type=float)
    parser.add_argument("--pano-controlnet-conditioning-scale", default=4.0, type=float)
    parser.add_argument("--pano-seed", default=-1, type=int)
    parser.add_argument("--pano-upscale", action="store_true", default=False)
    parser.add_argument("--pano-refinement", action="store_true", default=False)

    parser.add_argument("--dreamscene-iterations", default=10000, type=int)
    parser.add_argument("--dreamscene-port", default=8088, type=int)
    parser.add_argument("--dreamscene-ip", default="127.0.0.6", type=str)
    parser.add_argument("--dreamscene-quiet", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False, help="Check referenced files/classes and print planned commands without executing heavy model inference.")
    return parser.parse_args()


def _assert_exists(path: Path, label: str) -> list[str]:
    if path.exists():
        return [f"[ok] {label}: {path}"]
    return [f"[missing] {label}: {path}"]


def _assert_file_contains(path: Path, pattern: str, label: str) -> list[str]:
    if not path.exists():
        return [f"[missing] {label}: {path}"]
    text = path.read_text(encoding="utf-8", errors="ignore")
    if pattern in text:
        return [f"[ok] {label}: found `{pattern}` in {path}"]
    return [f"[missing] {label}: `{pattern}` not found in {path}"]


def validate_pipeline_references() -> list[str]:
    messages: list[str] = []

    messages.extend(_assert_exists(POEM_ROOT / "cogview_stage.py", "CogView stage entry"))
    messages.extend(_assert_exists(POEM_ROOT / "pano_stage.py", "Panorama stage entry"))
    messages.extend(_assert_exists(POEM_ROOT / "dreamscene_stage.py", "DreamScene stage entry"))

    messages.extend(
        _assert_file_contains(
            COGVIEW_DIFFUSERS_SRC / "diffusers" / "pipelines" / "cogview3" / "pipeline_cogview3plus.py",
            "class CogView3PlusPipeline",
            "CogView pipeline class",
        )
    )
    messages.extend(
        _assert_file_contains(
            SD360_ROOT / "img2panoimg" / "image_to_360panorama_image_pipeline.py",
            "class Image2360PanoramaImagePipeline",
            "Panorama pipeline class",
        )
    )
    messages.extend(
        _assert_file_contains(
            POEM_ROOT / "pano_stage.py",
            "def create_white_mask",
            "Default mask helper",
        )
    )
    messages.extend(
        _assert_file_contains(
            DREAMSCENE_ROOT / "scene" / "__init__.py",
            "class Scene",
            "DreamScene scene class",
        )
    )
    messages.extend(
        _assert_file_contains(
            DREAMSCENE_ROOT / "scene" / "gaussian_model.py",
            "class GaussianModel",
            "DreamScene Gaussian model",
        )
    )
    messages.extend(
        _assert_file_contains(
            DREAMSCENE_ROOT / "train.py",
            "def training(",
            "DreamScene training entry function",
        )
    )
    return messages


def main() -> None:
    args = parse_args()
    run_name = args.run_name or default_run_name(args.prompt)
    run_dir = ensure_dir(args.runs_dir / run_name)

    stage1_dir = ensure_dir(run_dir / "01_cogview")
    stage2_dir = ensure_dir(run_dir / "02_panorama")
    stage3_dir = ensure_dir(run_dir / "03_dreamscene")

    image_path = stage1_dir / "image.png"
    panorama_path = stage2_dir / "panorama.png"
    generated_mask_path = stage2_dir / "white_mask.png"
    dreamscene_source_dir = stage3_dir / "source"
    dreamscene_model_dir = stage3_dir / "output"

    panorama_prompt = args.panorama_prompt or args.prompt

    metadata = {
        "prompt": args.prompt,
        "panorama_prompt": panorama_prompt,
        "run_name": run_name,
        "dry_run": args.dry_run,
        "paths": {
            "image_path": str(image_path),
            "panorama_path": str(panorama_path),
            "dreamscene_source_dir": str(dreamscene_source_dir),
            "dreamscene_model_dir": str(dreamscene_model_dir),
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    cogview_cmd = python_cmd(
        POEM_ROOT / "cogview_stage.py",
        "--prompt",
        args.prompt,
        "--output-path",
        str(image_path),
        "--model-path",
        args.cogview_model_path,
        "--guidance-scale",
        str(args.cogview_guidance_scale),
        "--num-inference-steps",
        str(args.cogview_steps),
        "--width",
        str(args.cogview_width),
        "--height",
        str(args.cogview_height),
        "--dtype",
        args.cogview_dtype,
    )

    pano_cmd = python_cmd(
        POEM_ROOT / "pano_stage.py",
        "--image-path",
        str(image_path),
        "--prompt",
        panorama_prompt,
        "--output-path",
        str(panorama_path),
        "--generated-mask-path",
        str(generated_mask_path),
        "--num-inference-steps",
        str(args.pano_steps),
        "--guidance-scale",
        str(args.pano_guidance_scale),
        "--controlnet-conditioning-scale",
        str(args.pano_controlnet_conditioning_scale),
        "--seed",
        str(args.pano_seed),
    )
    if args.pano_model_dir is not None:
        pano_cmd.extend(["--model-dir", str(args.pano_model_dir)])
    if args.pano_mask_path is not None:
        pano_cmd.extend(["--mask-path", str(args.pano_mask_path)])
    if args.pano_upscale:
        pano_cmd.append("--upscale")
    if args.pano_refinement:
        pano_cmd.append("--refinement")

    dreamscene_cmd = python_cmd(
        POEM_ROOT / "dreamscene_stage.py",
        "--panorama-path",
        str(panorama_path),
        "--source-dir",
        str(dreamscene_source_dir),
        "--model-dir",
        str(dreamscene_model_dir),
        "--iterations",
        str(args.dreamscene_iterations),
        "--port",
        str(args.dreamscene_port),
        "--ip",
        args.dreamscene_ip,
    )
    if args.dreamscene_quiet:
        dreamscene_cmd.append("--quiet")

    if args.dry_run:
        print("Dry run: validating referenced files/classes from infer_pipeline.py")
        for line in validate_pipeline_references():
            print(line)
        print("\nDry run: planned commands")
        print(" ".join(cogview_cmd))
        print(" ".join(pano_cmd))
        print(" ".join(dreamscene_cmd))
        print(f"\nRun directory: {run_dir}")
        print(f"2D image target: {image_path}")
        print(f"Panorama target: {panorama_path}")
        print(f"DreamScene target: {dreamscene_model_dir}")
        return

    run_command(cogview_cmd, cwd=POEM_ROOT)
    run_command(pano_cmd, cwd=POEM_ROOT)
    run_command(dreamscene_cmd, cwd=POEM_ROOT)

    print("\nPipeline finished.")
    print(f"Run directory: {run_dir}")
    print(f"2D image: {image_path}")
    print(f"Panorama image: {panorama_path}")
    print(f"DreamScene output: {dreamscene_model_dir}")


if __name__ == "__main__":
    main()
