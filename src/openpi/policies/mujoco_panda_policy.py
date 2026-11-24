import dataclasses

import einops
import numpy as np

from openpi import transforms


GRIPPER_RAW_MAX = 255.0
GRIPPER_MAX_WIDTH_M = 0.10
STATE_DIM = 9
ACTION_DIM = 7


def _parse_image(image) -> np.ndarray:
    """Convert a LeRobot image to uint8 HWC format."""

    image = np.asarray(image)

    # LeRobot可能返回0～1的浮点图像。
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # CHW → HWC。
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"Expected image shape [H, W, 3] or [3, H, W], got {image.shape}"
        )

    return image


def _convert_actions_to_metric(actions) -> np.ndarray:
    """Convert raw A7 gripper values from 0-255 to width in metres."""

    metric_actions = np.asarray(actions, dtype=np.float32).copy()

    if metric_actions.ndim == 0 or metric_actions.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"Expected action shape [..., {ACTION_DIM}], "
            f"got {metric_actions.shape}"
        )

    metric_actions[..., 6] = (
        metric_actions[..., 6]
        * GRIPPER_MAX_WIDTH_M
        / GRIPPER_RAW_MAX
    )

    return metric_actions


@dataclasses.dataclass(frozen=True)
class MujocoPandaInputs(transforms.DataTransformFn):
    """Convert the single-arm MuJoCo Panda data to the π0.5 format."""

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/overview"])

        state = np.asarray(
            data["observation/state"],
            dtype=np.float32,
        )

        if state.shape[-1] != STATE_DIM:
            raise ValueError(
                f"Expected state shape [..., {STATE_DIM}], got {state.shape}"
            )

        blank_image = np.zeros_like(base_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": blank_image,
                "right_wrist_0_rgb": blank_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # 训练时有actions，推理时没有。
        if "actions" in data:
            inputs["actions"] = _convert_actions_to_metric(
                data["actions"]
            )

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class MujocoPandaOutputs(transforms.DataTransformFn):
    """Convert π0.5 outputs back to the single-arm A7 action."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)

        if actions.ndim == 0 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(
                f"Expected action shape [..., >= {ACTION_DIM}], "
                f"got {actions.shape}"
            )

        # π0.5内部是32维，仿真环境只需要真实机械臂的前7维。
        raw_actions = actions[..., :ACTION_DIM].copy()

        # 训练时把夹爪从0～255转成了米；输出给仿真时做相反转换。
        raw_actions[..., 6] = np.clip(
            raw_actions[..., 6]
            * GRIPPER_RAW_MAX
            / GRIPPER_MAX_WIDTH_M,
            0.0,
            GRIPPER_RAW_MAX,
        )

        return {
            "actions": raw_actions,
        }

"""
完成的转换是
overview
  → base_0_rgb

没有腕部相机
  → 两张黑图 + mask=False

S9 state
  → state
  → 后续由OpenPI自动补到32维

原始A7 action
  → 第7维从0～255转换成0～0.10m
  → 后续由OpenPI自动补到32维

π0.5输出32维
  → 取前7维"""
