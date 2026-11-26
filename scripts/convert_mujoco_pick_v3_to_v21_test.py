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

from convert_mujoco_pick_v3_to_v21 import CAMERA, PickV3Source, convert, decode_frames


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

    def test_video_time_shift_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            list(decode_frames(self.root / "videos/file-000.mkv", 0.02, [0.0], 30))


if __name__ == "__main__":
    unittest.main()
