"""Inputs/outputs transforms for the user-collected `three_tasks_nosvo` xArm7 dataset.

State / action layout (8-D each):
    [joint_0, joint_1, ..., joint_6, gripper]

The dataset has a single (LEFT) ZED camera; we map it to the model's `base_0_rgb`
slot and zero-pad the two wrist views.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_three_tasks_example() -> dict:
    """Random input example used by tests / norm-stats fake batches."""
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "stack the cubes",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ThreeTasksInputs(transforms.DataTransformFn):
    """Pack our dataset's keys into pi0/pi0.5's expected input dict."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        # Single camera setup -> base view only; pad both wrist slots with zeros.
        zero_img = np.zeros_like(base_image)
        # pi0-FAST does not mask zero-padded images; pi0/pi05 should mask them out.
        is_fast = self.model_type == _model.ModelType.PI0_FAST

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": zero_img,
                "right_wrist_0_rgb": zero_img,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if is_fast else np.False_,
                "right_wrist_0_rgb": np.True_ if is_fast else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class ThreeTasksOutputs(transforms.DataTransformFn):
    """Strip the model's action padding and return the first 8 action dims."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
