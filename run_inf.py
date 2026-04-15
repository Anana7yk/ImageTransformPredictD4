#!/usr/bin/env python3
"""
Thin entrypoint for probing D4 rotations with a single-image EfficientNet checkpoint.

This intentionally reuses the existing `run_probe_rotations.py` logic so that:
- the model is loaded only from checkpoint weights;
- the probe runs on one source image;
- synthetic targets e / r / r2 / r3 are generated from that image;
- the script prints what the current checkpoint predicts for each pair.

Use `train_config_d4.yaml` with EfficientNet checkpoints and
`train_config_d4_vit.yaml` with ViT checkpoints.
"""

from run_probe_rotations import main


if __name__ == "__main__":
    main()
