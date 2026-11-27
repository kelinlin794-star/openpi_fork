"""Check real one- or three-view S9/A7 pick samples without training a model."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("MUJOCO_PANDA_DATA_ROOT"))
    parser.add_argument("--config-name", choices=("pi05_mujoco_panda_pick", "pi05_mujoco_panda_pick_3view"),
                        default="pi05_mujoco_panda_pick")
    args = parser.parse_args()
    if args.dataset_root is None:
        parser.error("Set MUJOCO_PANDA_DATA_ROOT or pass --dataset-root pointing at converted v2.1 data.")
    root = args.dataset_root.resolve()
    # The named config reads this variable at import time.
    os.environ["MUJOCO_PANDA_DATA_ROOT"] = str(root)

    from openpi import transforms  # noqa: PLC0415
    from openpi.policies import mujoco_panda_policy as panda  # noqa: PLC0415
    from openpi.training import config as configs  # noqa: PLC0415
    from openpi.training import data_loader  # noqa: PLC0415

    config = configs.get_config(args.config_name)
    mapping = panda.IMAGE_KEY_MAPPING if config.data.three_views else panda.IMAGE_KEY_MAPPING[:1]
    expected_views = [model_key for _, _, model_key in mapping]
    missing_views = set(panda.MODEL_IMAGE_KEYS) - set(expected_views)
    data_config = config.data.create(config.assets_dirs, config.model)
    # Shares the exact reader and action horizon used by training and norm stats.
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if len(dataset) == 0:
        raise ValueError("Dataset has no samples.")
    prepare = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    checks = []
    for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        raw = dataset[index]
        sample = prepare(raw)
        valid_views = [key for key, valid in sample["image_mask"].items() if bool(valid)]
        if valid_views != expected_views:
            raise ValueError(f"Sample {index}: expected valid views {expected_views}, got {valid_views}.")
        if set(sample["image"]) != set(panda.MODEL_IMAGE_KEYS):
            raise ValueError(f"Sample {index}: unexpected model camera slots.")
        for key in missing_views:
            if np.any(sample["image"][key]):
                raise ValueError(f"Sample {index}: missing wrist camera is not zero-padded.")
        image_shapes = {}
        for source_key, _, model_key in mapping:
            image = sample["image"][model_key]
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(f"Sample {index}: {source_key} should be uint8 RGB (480, 640, 3).")
            if not np.array_equal(image, panda._parse_image(raw[source_key])):
                raise ValueError(f"Sample {index}: camera was replaced or mapped incorrectly: {source_key}.")
            image_shapes[model_key] = list(image.shape)
        if sample["state"].shape != (9,) or not np.isfinite(sample["state"]).all():
            raise ValueError(f"Sample {index}: expected finite single-arm S9 state.")
        if sample["actions"].shape != (config.model.action_horizon, 7):
            raise ValueError(f"Sample {index}: incorrect single-arm action chunk shape.")
        if not np.isfinite(sample["actions"]).all():
            raise ValueError(f"Sample {index}: actions contain non-finite values.")
        if not np.array_equal(sample["state"], np.asarray(raw["observation.state"], dtype=np.float32)):
            raise ValueError(f"Sample {index}: recorded state was changed by the input adapter.")
        expected = np.asarray(raw["action"], dtype=np.float32).copy()
        expected[..., 6] *= panda.GRIPPER_MAX_WIDTH_M / panda.GRIPPER_RAW_MAX
        if not np.allclose(sample["actions"], expected, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Sample {index}: gripper conversion or action labels differ.")
        if not isinstance(sample.get("prompt"), str) or not sample["prompt"].strip():
            raise ValueError(f"Sample {index}: missing task instruction.")
        checks.append({"index": index, "valid_views": valid_views, "state_shape": list(sample["state"].shape),
                       "action_shape": list(sample["actions"].shape), "image_shapes": image_shapes})
    print(json.dumps({"config_name": args.config_name, "dataset_root": str(root), "dataset_frames": len(dataset),
                      "source_cameras": [key for key, _, _ in mapping], "samples_checked": checks}, indent=2))
    print("Camera and single-arm sample checks passed. No model or training run was started.")


if __name__ == "__main__":
    main()
