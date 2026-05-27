# Scripts Notes

The `scripts` directory contains thin execution entry points that call the reusable code under `src`.

## Files

- `train_pipeline.py` runs the full training workflow and writes processed data, figures, and models
- `generate_notebooks.py` creates the documented notebooks after artifacts have been generated

## Why keep scripts thin

The scripts are intentionally small so the important logic remains testable and importable from `src`. This also makes Docker, local runs, and future automation easier to maintain.

