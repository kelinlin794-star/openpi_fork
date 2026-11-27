"""Camera/arm contract tests; no model weights or GPU computation required."""

import unittest

import numpy as np

from openpi.policies import mujoco_panda_policy as panda


class SingleViewPickTest(unittest.TestCase):
    def example(self):
        actions = np.zeros((4, 7), dtype=np.float32)
        actions[:, 6] = [0, 255, 127.5, 255]
        state = np.arange(9, dtype=np.float32)
        state[-1] = 0.05
        return {panda.INPUT_IMAGE_KEY: np.full((12, 16, 3), 100, dtype=np.uint8),
                "observation/state": state, "actions": actions, "prompt": b"pick and place"}

    def test_only_overview_is_valid_even_with_other_source_views(self):
        data = self.example()
        data.update({"observation.images.camera_2": object(), "observation.images.camera_3": object(),
                     "observation.gesture": 5})
        result = panda.MujocoPandaInputs()(data)
        self.assertEqual([key for key, valid in result["image_mask"].items() if valid], ["base_0_rgb"])
        self.assertEqual(set(result["image"]), {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"})
        np.testing.assert_array_equal(result["image"]["base_0_rgb"], data[panda.INPUT_IMAGE_KEY])
        for key in panda.PADDING_IMAGE_KEYS:
            self.assertFalse(np.any(result["image"][key]))
        self.assertNotIn("observation.gesture", result)

    def test_raw_gripper_converted_once_and_state_unchanged(self):
        data = self.example()
        result = panda.MujocoPandaInputs()(data)
        np.testing.assert_allclose(result["actions"][:, 6], [0, 0.1, 0.05, 0.1])
        np.testing.assert_array_equal(result["state"], data["observation/state"])
        self.assertEqual(result["prompt"], "pick and place")
        np.testing.assert_array_equal(data["actions"][:, 6], [0, 255, 127.5, 255])

    def test_chw_float_camera(self):
        data = self.example()
        data[panda.INPUT_IMAGE_KEY] = np.ones((3, 12, 16), dtype=np.float32)
        image = panda.MujocoPandaInputs()(data)["image"][panda.MODEL_IMAGE_KEY]
        self.assertEqual(image.shape, (12, 16, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(np.all(image == 255))

    def test_rejects_dual_arm_or_scalar_state(self):
        for state in (np.zeros(16), np.array(1), np.zeros((1, 9))):
            with self.subTest(shape=state.shape):
                data = self.example()
                data["observation/state"] = state
                with self.assertRaisesRegex(ValueError, "single-arm"):
                    panda.MujocoPandaInputs()(data)

    def test_rejects_dual_arm_action(self):
        data = self.example()
        data["actions"] = np.zeros((4, 14))
        with self.assertRaises(ValueError):
            panda.MujocoPandaInputs()(data)

    def test_inference_needs_only_one_real_camera_and_state(self):
        data = self.example()
        del data["actions"]
        result = panda.MujocoPandaInputs()(data)
        self.assertNotIn("actions", result)
        self.assertEqual(sum(bool(valid) for valid in result["image_mask"].values()), 1)

    def test_three_external_views_keep_their_pixels_and_all_masks_true(self):
        data = self.example()
        for value, (_, input_key, _) in enumerate(panda.IMAGE_KEY_MAPPING, start=1):
            data[input_key] = np.full((12, 16, 3), value * 30, dtype=np.uint8)
        result = panda.MujocoPandaInputs(three_views=True)(data)
        self.assertTrue(all(result["image_mask"].values()))
        for _, input_key, model_key in panda.IMAGE_KEY_MAPPING:
            np.testing.assert_array_equal(result["image"][model_key], data[input_key])
        np.testing.assert_array_equal(result["state"], data["observation/state"])
        np.testing.assert_allclose(result["actions"][:, 6], [0, 0.1, 0.05, 0.1])

    def test_three_views_require_both_additional_cameras(self):
        for missing in ("observation/camera_2", "observation/camera_3"):
            data = self.example()
            data.update({"observation/camera_2": data[panda.INPUT_IMAGE_KEY],
                         "observation/camera_3": data[panda.INPUT_IMAGE_KEY]})
            del data[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, missing):
                panda.MujocoPandaInputs(three_views=True)(data)

    def test_three_view_inference_has_no_training_action_requirement(self):
        data = self.example()
        del data["actions"]
        data["observation/camera_2"] = np.ones((3, 12, 16), dtype=np.float32)
        data["observation/camera_3"] = np.zeros((12, 16, 3), dtype=np.uint8)
        result = panda.MujocoPandaInputs(three_views=True)(data)
        self.assertNotIn("actions", result)
        self.assertTrue(all(result["image_mask"].values()))
        self.assertTrue(np.all(result["image"]["left_wrist_0_rgb"] == 255))


if __name__ == "__main__":
    unittest.main()
