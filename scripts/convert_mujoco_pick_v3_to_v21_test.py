"""CPU checks for episode boundaries, video offsets and unchanged raw gripper units.

Run: python -m unittest discover -s scripts -p convert_mujoco_pick_v3_to_v21_test.py
Requires only numpy, pyarrow and av; the production writer is checked on the server.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_mujoco_pick_v3_to_v21 import CAMERA, THREE_CAMERAS, THREE_VIEW_REPO_ID, PickV3Source, convert, decode_frames


def video(path, colors):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "bgr0"
        for color in colors:
            image = np.full((480, 640, 3), color, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class MemoryWriter:
    @classmethod
    def create(cls, **kwargs):
        cls.last = cls()
        cls.last.options = kwargs
        cls.last.episodes = []
        cls.last.buffer = []
        kwargs["root"].mkdir()
        return cls.last

    def add_frame(self, frame):
        for key, feature in self.options["features"].items():
            value = frame[key]
            if value.shape != feature["shape"]:
                raise ValueError(f"Unexpected feature shape: {key}")
            if feature["dtype"] != "image" and value.dtype != np.dtype(feature["dtype"]):
                raise ValueError(f"Unexpected feature dtype: {key}")
        frame = copy.deepcopy(frame)
        frame["timestamp"] = len(self.buffer) / self.options["fps"]
        self.buffer.append(frame)

    def save_episode(self):
        self.episodes.append(self.buffer)
        self.buffer = []


class ConversionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        for relative in ("meta/episodes/chunk-000", "data/chunk-000", "videos"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        self.info = {
            "codebase_version": "v3.0", "fps": 30, "total_episodes": 2, "total_frames": 3,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [9], "names": {"axes": list("abcdefghi")}},
                "action": {"dtype": "float32", "shape": [7], "names": {"axes": [*list("abcdef"), "gripper_target_0_255"]}},
                CAMERA: {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]},
                "observation.hand_visible": {"dtype": "int64", "shape": [1], "names": ["visible"]},
            },
            "video_path": "videos/file-{file_index:03d}.mkv",
        }
        self.write_info()
        self.rows = []
        for index, (ep, frame_index, gripper) in enumerate(((0, 0, 255), (0, 1, 0), (1, 0, 255))):
            self.rows.append({
                "observation.state": [float(index)] * 9, "action": [float(index)] * 6 + [float(gripper)],
                "observation.hand_visible": 1, "episode_index": ep, "frame_index": frame_index,
                "timestamp": frame_index / 30, "index": index, "task_index": 0,
            })
        self.episodes = []
        for ep, start, stop in ((0, 0, 2), (1, 2, 3)):
            self.episodes.append({
                "episode_index": ep, "length": stop - start, "dataset_from_index": start,
                "dataset_to_index": stop, f"videos/{CAMERA}/chunk_index": 0,
                f"videos/{CAMERA}/file_index": 0, f"videos/{CAMERA}/from_timestamp": start / 30,
                f"videos/{CAMERA}/to_timestamp": stop / 30,
            })
        self.write_tables()
        video(self.root / "videos/file-000.mkv", [10, 80, 160])

    def write_info(self):
        (self.root / "meta/info.json").write_text(json.dumps(self.info), encoding="utf-8")

    def write_tables(self):
        pq.write_table(pa.Table.from_pylist(self.rows), self.root / "data/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist(self.episodes), self.root / "meta/episodes/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": "pick red cube"}]),
                       self.root / "meta/tasks.parquet")

    def add_three_cameras(self):
        self.info["video_path"] = "videos/{video_key}/file-{file_index:03d}.mkv"
        for camera_index, camera in enumerate(THREE_CAMERAS):
            self.info["features"][camera] = copy.deepcopy(self.info["features"][CAMERA])
            folder = self.root / "videos" / camera
            folder.mkdir(exist_ok=True)
            # The third camera has its own video offset; frame-number-only
            # synchronization would silently select the wrong observations.
            padding = 2 if camera_index == 2 else 0
            video(folder / "file-000.mkv", [0] * padding + [c + camera_index * 20 for c in (10, 80, 160)])
            for episode in self.episodes:
                prefix = f"videos/{camera}"
                episode.update({
                    f"{prefix}/chunk_index": 0, f"{prefix}/file_index": 0,
                    f"{prefix}/from_timestamp": (episode["dataset_from_index"] + padding) / 30,
                    f"{prefix}/to_timestamp": (episode["dataset_to_index"] + padding) / 30,
                })
        self.write_info()
        self.write_tables()

    def test_conversion_preserves_episode_rows_pixels_and_units(self):
        output = Path(self.temp.name) / "converted"
        report = convert(PickV3Source(self.root), output, dataset_class=MemoryWriter)
        writer = MemoryWriter.last
        self.assertEqual([len(ep) for ep in writer.episodes], [2, 1])
        self.assertEqual([float(f["action"][6]) for ep in writer.episodes for f in ep], [255, 0, 255])
        self.assertEqual([int(f[CAMERA][0, 0, 0]) for ep in writer.episodes for f in ep], [10, 80, 160])
        self.assertEqual(writer.episodes[1][0]["timestamp"], 0)
        self.assertEqual(writer.options["features"][CAMERA]["dtype"], "image")
        self.assertFalse(writer.options["use_videos"])
        self.assertTrue(report["complete"])
        self.assertFalse(report["engineering_subset"])
        self.assertEqual(PickV3Source(self.root).validate()["frames"], 3)

    def test_does_not_overwrite_existing_output(self):
        with self.assertRaises(FileExistsError):
            convert(PickV3Source(self.root), self.root, dataset_class=MemoryWriter)

    def test_rejects_missing_frame(self):
        self.rows[1]["frame_index"] = 2
        self.write_tables()
        with self.assertRaisesRegex(ValueError, "frame indices"):
            PickV3Source(self.root).validate()

    def test_rejects_metric_pack_semantics(self):
        self.info["features"]["action"]["names"]["axes"][-1] = "gripper_width_m"
        self.write_info()
        with self.assertRaisesRegex(ValueError, "metric Training Packs"):
            PickV3Source(self.root)

    def test_missing_video_does_not_mark_conversion_complete(self):
        path = self.root / "videos/file-000.mkv"
        video(path, [10, 80])  # Third observation is missing from the video.
        output = Path(self.temp.name) / "incomplete"
        with self.assertRaisesRegex(ValueError, "Video ends"):
            convert(PickV3Source(self.root), output, dataset_class=MemoryWriter)
        self.assertFalse(json.loads((output / "conversion.json").read_text())["complete"])

    def test_subset_is_marked_as_engineering(self):
        report = convert(PickV3Source(self.root), Path(self.temp.name) / "smoke",
                         max_episodes=1, dataset_class=MemoryWriter)
        self.assertTrue(report["engineering_subset"])
        self.assertEqual(len(report["episodes"]), 1)

    def test_export_uses_only_overview_when_source_has_three_cameras(self):
        for key in ("observation.images.camera_2", "observation.images.camera_3"):
            self.info["features"][key] = copy.deepcopy(self.info["features"][CAMERA])
        self.write_info()
        source = PickV3Source(self.root)
        features = source.output_features()
        self.assertEqual([key for key, feature in features.items() if feature["dtype"] == "image"], [CAMERA])
        self.assertNotIn("observation.images.camera_2", features)
        self.assertNotIn("observation.images.camera_3", features)

    def test_video_time_shift_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            list(decode_frames(self.root / "videos/file-000.mkv", 0.02, [0.0], 30))

    def test_three_camera_conversion_uses_each_video_offset(self):
        self.add_three_cameras()
        report = convert(PickV3Source(self.root, three_views=True), Path(self.temp.name) / "three",
                         dataset_class=MemoryWriter)
        writer = MemoryWriter.last
        self.assertEqual(writer.options["repo_id"], THREE_VIEW_REPO_ID)
        self.assertEqual(report["cameras"], list(THREE_CAMERAS))
        self.assertEqual([len(ep) for ep in writer.episodes], [2, 1])
        for camera_index, camera in enumerate(THREE_CAMERAS):
            self.assertEqual([int(f[camera][0, 0, 0]) for ep in writer.episodes for f in ep],
                             [c + camera_index * 20 for c in (10, 80, 160)])
            self.assertEqual(writer.options["features"][camera]["dtype"], "image")
        self.assertEqual([float(f["action"][6]) for ep in writer.episodes for f in ep], [255, 0, 255])
        self.assertTrue(report["complete"])

    def test_three_camera_mode_requires_all_declared_views(self):
        with self.assertRaisesRegex(ValueError, "camera_2"):
            PickV3Source(self.root, three_views=True)

    def test_three_camera_mode_rejects_missing_video_file(self):
        self.add_three_cameras()
        (self.root / "videos" / THREE_CAMERAS[2] / "file-000.mkv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "camera_3"):
            PickV3Source(self.root, three_views=True).validate()

    def test_truncated_third_camera_does_not_mark_conversion_complete(self):
        self.add_three_cameras()
        video(self.root / "videos" / THREE_CAMERAS[2] / "file-000.mkv", [0, 0, 50])
        output = Path(self.temp.name) / "truncated-three"
        with self.assertRaisesRegex(ValueError, "Video ends"):
            convert(PickV3Source(self.root, three_views=True), output, dataset_class=MemoryWriter)
        self.assertFalse(json.loads((output / "conversion.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
