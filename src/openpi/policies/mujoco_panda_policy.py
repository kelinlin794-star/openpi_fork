import dataclasses

import einops
import numpy as np

from openpi import transforms


GRIPPER_RAW_MAX = 255.0
GRIPPER_MAX_WIDTH_M = 0.10
STATE_DIM = 9
ACTION_DIM = 7
# Keep the original single-view experiment; the three-view experiment opts in.
DATASET_IMAGE_KEY = "observation.images.overview"
INPUT_IMAGE_KEY = "observation/overview"
MODEL_IMAGE_KEY = "base_0_rgb"
PADDING_IMAGE_KEYS = ("left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_KEY_MAPPING = (
    (DATASET_IMAGE_KEY, INPUT_IMAGE_KEY, MODEL_IMAGE_KEY),
    ("observation.images.camera_2", "observation/camera_2", "left_wrist_0_rgb"),
    ("observation.images.camera_3", "observation/camera_3", "right_wrist_0_rgb"),
)
MODEL_IMAGE_KEYS = tuple(model_key for _, _, model_key in IMAGE_KEY_MAPPING)


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
    """Single-arm S9/A7 input with one or three real external camera views.

    By default only overview is valid. With three_views=True, camera_2 and
    camera_3 fill the other official slots and all three masks are True.
    The slot names do not describe the physical positions of these cameras.
    Gesture labels are not model inputs.
    """

    three_views: bool = False

    def __call__(self, data: dict) -> dict:
        mapping = IMAGE_KEY_MAPPING if self.three_views else IMAGE_KEY_MAPPING[:1]
        missing = [input_key for _, input_key, _ in mapping if input_key not in data]
        if missing:
            raise ValueError(f"Missing required camera inputs: {missing}. Check the dataset and camera mapping.")
        images = {model_key: _parse_image(data[input_key]) for _, input_key, model_key in mapping}
        base_image = images[MODEL_IMAGE_KEY]

        state = np.asarray(
            data["observation/state"],
            dtype=np.float32,
        )

        if state.shape != (STATE_DIM,):
            raise ValueError(
                f"Expected one single-arm state with shape ({STATE_DIM},), got {state.shape}"
            )

        blank_image = np.zeros_like(base_image)

        inputs = {
            "state": state,
            "image": {key: images.get(key, blank_image) for key in MODEL_IMAGE_KEYS},
            "image_mask": {key: np.bool_(key in images) for key in MODEL_IMAGE_KEYS},
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
overview → base_0_rgb
三视角模式：camera_2 → left_wrist_0_rgb，camera_3 → right_wrist_0_rgb
三路都是外部相机，三路 mask=True；字段名沿用官方接口。

单视角模式下未使用的两个图像槽位
  → 两张黑图 + mask=False

S9 state
  → state
  → 后续由OpenPI自动补到32维

原始A7 action
  → 第7维从0～255转换成0～0.10m
  → 后续由OpenPI自动补到32维

π0.5输出32维
  → 取前7维"""
