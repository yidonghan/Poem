# Poem Pipeline

This directory packages a three-stage generation workflow:

1. Text -> image
2. Image -> 360 scene image
3. 360 scene image -> 3D scene

The main entry is:

```bash
python3 infer_pipeline.py --prompt "your prompt here"
```

## Directory Overview

- `infer_pipeline.py`
  Runs the full workflow from a single prompt.
- `requirements.txt`
  Consolidated Python dependencies for this directory.
- `vendor/`
  Local copies of the code used by the three stages.
- `runs/`
  Default output location for each run.

## What This Pipeline Does

Given one text prompt, the workflow will:

1. Generate a single 2D image
2. Expand that image into a 360 image
3. Convert the 360 image into a 3D scene representation

Each run creates its own output folder so intermediate results are preserved.

## Installation

Create and activate your environment first, then install:

```bash
pip install -r requirements.txt
```

Some dependencies in this project include compiled CUDA extensions. If your machine has not built them before, installation may take longer than a standard Python-only environment.

## Required Assets

Before running the full workflow, make sure the following resources are already available in the workspace:

- Text-to-image stage weights
- 360-image stage weights
- Depth estimation checkpoint
- Feature encoder weights

This directory contains the pipeline code, but large pretrained assets are still expected to exist on disk.

## Quick Start

Run a lightweight preflight check first:

```bash
python3 infer_pipeline.py --prompt "a quiet mountain temple at dawn" --dry-run
```

This mode will:

- Verify that the referenced stage files and key classes can be found
- Print the commands that would be executed
- Show the output paths for the current run

It will not load models or generate images.

## Full Run

Example:

```bash
python3 infer_pipeline.py --prompt "a quiet mountain temple at dawn"
```

If you want to control the run folder name:

```bash
python3 infer_pipeline.py --prompt "a quiet mountain temple at dawn" --run-name temple_test
```

If you want to customize stage-specific options, run:

```bash
python3 infer_pipeline.py --help
```

## Outputs

By default, results are written under:

```text
runs/<run_name>/
```

Inside each run folder you will typically see three stage folders. A typical layout is:

```text
stage_1/
  image.png

stage_2/
  scene_360.png
  white_mask.png

stage_3/
  source/
  output/
```

Even though the folder names preserve the internal code structure, you can think of them simply as:

- Stage 1 output
- Stage 2 output
- Stage 3 output

## Mask Behavior

The second stage expects an image mask.

If you do not provide one, the pipeline will automatically create a white mask based on the size of the first-stage image. This is useful for fast end-to-end testing.

## Notes

- The final stage is not an instant forward pass. It performs scene-level optimization, so it can take a noticeable amount of time.
- A successful dry run only confirms that the call chain is wired correctly. It does not guarantee that all checkpoints, GPU libraries, or compiled extensions are ready.
- If a real run fails, check local weights, CUDA availability, and extension installation first.

## Debugging Tips

Start from these checks:

1. Run `--dry-run` and confirm the printed commands and output paths look correct.
2. Confirm the required weights and checkpoints exist locally.
3. Confirm the Python environment can import the installed packages.
4. Confirm compiled CUDA extensions were installed successfully.

## Recommended Workflow

For the first successful test:

1. Run a dry run
2. Run one short prompt end to end
3. Check that the 2D image looks reasonable
4. Check that the 360 image is produced
5. Check that the final output directory is created

Once that path is stable, you can start tuning prompt wording and stage-specific options.
