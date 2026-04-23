from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from common import DREAMSCENE_ORIG_ROOT, DREAMSCENE_ROOT, ensure_dir, python_cmd, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: lift a panorama image into a 3D DreamScene360 scene.")
    parser.add_argument("--panorama-path", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--iterations", default=10000, type=int)
    parser.add_argument("--port", default=8088, type=int)
    parser.add_argument("--ip", default="127.0.0.6", type=str)
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = ensure_dir(args.source_dir)
    model_dir = ensure_dir(args.model_dir)

    panorama_target = source_dir / "panorama.png"
    shutil.copy2(args.panorama_path, panorama_target)
    print(f"Copied panorama to {panorama_target}")

    cmd = python_cmd(
        DREAMSCENE_ROOT / "train.py",
        "-s",
        str(source_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(args.iterations),
        "--port",
        str(args.port),
        "--ip",
        args.ip,
    )
    if args.quiet:
        cmd.append("--quiet")

    env = {}
    omnidata_ckpt = DREAMSCENE_ORIG_ROOT / "pre_checkpoints" / "omnidata_dpt_depth_v2.ckpt"
    if omnidata_ckpt.exists():
        env["DREAMSCENE_OMNIDATA_CKPT"] = str(omnidata_ckpt)

    # Prefer a caller-provided DINO weight path. If absent, keep the runtime fallback logic.
    dino_env = os.environ.get("DREAMSCENE_DINOV2_WEIGHTS")
    if dino_env:
        env["DREAMSCENE_DINOV2_WEIGHTS"] = dino_env

    run_command(cmd, cwd=DREAMSCENE_ROOT, env=env)


if __name__ == "__main__":
    main()
