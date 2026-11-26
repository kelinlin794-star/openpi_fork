"""Convert the historical HF Panda pick dataset to OpenPI's pinned LeRobot v2.1.

Run with --help for arguments. Source files are never changed. Only overview RGB
and numeric features are exported; other camera videos remain in the source.
Decoded RGB is stored losslessly as embedded images, without another H.264 encode.
The recorded same-row action labels, including gripper commands in 0..255, remain
unchanged. This is NOT a Safe Stitch Training Pack importer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CAMERA = "observation.images.overview"
REPO_ID = "local/mujoco_panda_pick_v21"
HF_FOLDER = "act_dp_mujoco_panda_pick_20261022_hand_delta_ema"
GENERATED = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes dataset root: {relative}")
    return path


class PickV3Source:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not (self.root / "meta/info.json").is_file() and (self.root / HF_FOLDER).is_dir():
            self.root /= HF_FOLDER
        if not (self.root / "meta/info.json").is_file():
            raise ValueError(
                "Expected the HF pick directory containing meta/info.json, data/ and videos/. "
                "A Paper A Training Pack (manifest.json and episodes/) needs a separate importer."
            )
        self.info = read_json(self.root / "meta/info.json")
        if self.info.get("codebase_version") != "v3.0" or self.info.get("fps") != 30:
            raise ValueError("This converter accepts the historical 30 FPS LeRobot v3.0 pick export only.")
        features = self.info["features"]
        if features["observation.state"]["shape"] != [9] or features["action"]["shape"] != [7]:
            raise ValueError("Expected single-arm state S9 and action A7.")
        action_names = features["action"]["names"]
        if isinstance(action_names, dict):
            action_names = action_names.get("axes", [])
        if not action_names or action_names[-1] != "gripper_target_0_255":
            raise ValueError("Expected raw 0..255 gripper labels; metric Training Packs must not use this converter.")
        if features[CAMERA]["shape"] != [480, 640, 3]:
            raise ValueError("Expected the original 480x640 overview RGB camera.")
        self.fps = self.info["fps"]
        episode_files = sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet"))
        data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not episode_files or not data_files:
            raise ValueError("Missing episode metadata or numeric data Parquet files.")
        self.episodes = sorted(
            [row for path in episode_files for row in pq.read_table(path).to_pylist()],
            key=lambda row: row["episode_index"],
        )
        self.data = pa.concat_tables([pq.read_table(path) for path in data_files])
        task_rows = pq.read_table(self.root / "meta/tasks.parquet").to_pylist()
        self.tasks = {int(row["task_index"]): row["task"] for row in task_rows}
        if len(self.tasks) != len(task_rows) or not all(isinstance(t, str) and t.strip() for t in self.tasks.values()):
            raise ValueError("Missing, duplicate or empty task instructions.")
        if len(self.episodes) != self.info["total_episodes"] or self.data.num_rows != self.info["total_frames"]:
            raise ValueError("Actual episode/frame counts differ from meta/info.json.")
        if [row["episode_index"] for row in self.episodes] != list(range(len(self.episodes))):
            raise ValueError("Source episode indices must be contiguous and start at zero.")
        self.numeric_keys = [
            key for key, feature in features.items()
            if key not in GENERATED and feature["dtype"] not in ("video", "image")
        ]
        if not set(self.numeric_keys) <= set(self.data.column_names):
            raise ValueError("Data is missing declared numeric features.")

    def rows(self, episode: dict) -> list[dict]:
        index = episode["episode_index"]
        table = self.data.filter(pc.equal(self.data["episode_index"], index))
        rows = table.to_pylist()
        n = episode["length"]
        if len(rows) != n or not n:
            raise ValueError(f"Episode {index}: empty or incorrect frame count.")
        if [row["frame_index"] for row in rows] != list(range(n)):
            raise ValueError(f"Episode {index}: missing/reordered frame indices.")
        start, stop = episode["dataset_from_index"], episode["dataset_to_index"]
        if stop - start != n or [row["index"] for row in rows] != list(range(start, stop)):
            raise ValueError(f"Episode {index}: global index mapping differs.")
        timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
        # The pinned writer generates frame_index/fps itself. Require identical
        # serialized float32 timestamps before allowing it to do so.
        if not np.array_equal(timestamps.astype(np.float32), (np.arange(n) / self.fps).astype(np.float32)):
            raise ValueError(f"Episode {index}: timestamps are not the recorded regular 30 Hz sequence.")
        for key, dim in (("observation.state", 9), ("action", 7)):
            array = np.asarray([row[key] for row in rows], dtype=np.float32)
            if array.shape != (n, dim) or not np.isfinite(array).all():
                raise ValueError(f"Episode {index}: invalid {key} shape or non-finite values.")
        gripper = np.asarray([row["action"][6] for row in rows])
        if np.any(gripper < 0) or np.any(gripper > 255):
            raise ValueError(f"Episode {index}: gripper is outside 0..255.")
        if any(int(row["task_index"]) not in self.tasks for row in rows):
            raise ValueError(f"Episode {index}: unknown task index.")
        return rows

    def video(self, episode: dict) -> tuple[Path, float, float]:
        prefix = f"videos/{CAMERA}"
        path = self.info["video_path"].format(
            video_key=CAMERA,
            chunk_index=episode[f"{prefix}/chunk_index"],
            file_index=episode[f"{prefix}/file_index"],
        )
        return (
            within(self.root, path),
            float(episode[f"{prefix}/from_timestamp"]),
            float(episode[f"{prefix}/to_timestamp"]),
        )

    def validate(self, *, require_videos: bool = True) -> dict:
        end = 0
        for episode in self.episodes:
            self.rows(episode)
            if episode["dataset_from_index"] != end:
                raise ValueError("Episode ranges overlap or leave gaps.")
            end = episode["dataset_to_index"]
            path, start, stop = self.video(episode)
            if not np.isfinite([start, stop]).all() or start < 0:
                raise ValueError("Invalid video time range.")
            if not np.isclose(stop - start, episode["length"] / self.fps, atol=1e-4, rtol=0):
                raise ValueError("Video time range differs from numeric frame count.")
            if require_videos and not path.is_file():
                raise FileNotFoundError(f"Download the original overview video first: {path}")
        if end != self.data.num_rows:
            raise ValueError("Episode ranges do not cover the dataset.")
        return {"episodes": len(self.episodes), "frames": end, "fps": self.fps, "gripper_unit": "0..255"}

    def output_features(self) -> dict:
        keys = [*self.numeric_keys, CAMERA]
        result = {key: copy.deepcopy(self.info["features"][key]) for key in keys}
        for feature in result.values():
            feature["shape"] = tuple(feature["shape"])
            if isinstance(feature.get("names"), dict):
                feature["names"] = feature["names"]["axes"]
            feature.pop("info", None)
        result[CAMERA]["dtype"] = "image"
        return result


def decode_frames(path: Path, start: float, timestamps: list[float], fps: int):
    """Use v3 episode offsets and decoded presentation times, never compressed-file row indices."""
    import av  # noqa: PLC0415

    tolerance = 0.25 / fps
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((start - 1 / fps) / stream.time_base)), stream=stream, backward=True)
        frames = iter(container.decode(stream))
        for timestamp in timestamps:
            target = start + timestamp
            for frame in frames:
                if frame.pts is None:
                    raise ValueError(f"Video frame has no presentation timestamp: {path}")
                time = float(frame.pts * stream.time_base)
                if time >= target - tolerance:
                    break
            else:
                raise ValueError(f"Video ends before observation at {target:.6f}s: {path}")
            if abs(time - target) > tolerance:
                raise ValueError(f"Video/observation mismatch: wanted {target:.6f}s, got {time:.6f}s: {path}")
            yield frame.to_ndarray(format="rgb24")


def convert(source: PickV3Source, output: Path, *, repo_id: str = REPO_ID, max_episodes: int | None = None,
            dataset_class=None) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output (including incomplete conversions): {output}")
    if output.is_relative_to(source.root) or source.root.is_relative_to(output):
        raise ValueError("Source and output must be separate, non-nested directories.")
    if max_episodes is not None and not 1 <= max_episodes <= len(source.episodes):
        raise ValueError("max_episodes must be between 1 and the source episode count.")
    source.validate()
    if dataset_class is None:
        from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset  # noqa: PLC0415

        if CODEBASE_VERSION != "v2.1":
            raise ValueError("Run this script in OpenPI's pinned LeRobot v2.1 environment.")
        dataset_class = LeRobotDataset
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = dataset_class.create(
        repo_id=repo_id, root=output, fps=source.fps, robot_type=source.info.get("robot_type", "mujoco_panda"),
        features=source.output_features(), use_videos=False,
    )
    provenance = {
        "complete": False, "source_root": str(source.root), "repo_id": repo_id,
        "source_info_sha256": hashlib.sha256((source.root / "meta/info.json").read_bytes()).hexdigest(),
        "source_format": "v3.0", "target_format": "v2.1", "action_alignment": "recorded-row-unchanged",
        "gripper_unit": "0..255", "camera": CAMERA, "image_storage": "lossless embedded RGB images",
        "engineering_subset": max_episodes is not None and max_episodes < len(source.episodes),
        "episodes": [],
    }
    report_path = output / "conversion.json"
    report_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for episode in source.episodes[:max_episodes]:
        rows = source.rows(episode)
        path, start, _ = source.video(episode)
        frames = decode_frames(path, start, [row["timestamp"] for row in rows], source.fps)
        for row, image in zip(rows, frames, strict=True):
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError("Decoded overview does not match the declared image shape/type.")
            frame = {
                key: np.asarray(row[key], dtype=source.info["features"][key]["dtype"]).reshape(
                    source.info["features"][key]["shape"]
                )
                for key in source.numeric_keys
            }
            frame.update({CAMERA: image, "task": source.tasks[int(row["task_index"])]})
            dataset.add_frame(frame)
        dataset.save_episode()
        provenance["episodes"].append({"source_episode_index": episode["episode_index"], "frames": len(rows)})
        report_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"Converted episode {episode['episode_index']}: {len(rows)} frames", flush=True)
    provenance["complete"] = True
    report_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local HF pick folder or its merged_data parent.")
    parser.add_argument("--output", type=Path, help="New, non-existing output directory; source is never overwritten.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--max-episodes", type=int, help="Engineering smoke subset only; omit for the full dataset.")
    parser.add_argument("--inspect-only", action="store_true", help="Check all numeric rows and metadata without decoding.")
    args = parser.parse_args()
    source = PickV3Source(args.source)
    if args.inspect_only:
        print(json.dumps(source.validate(require_videos=False), indent=2))
        print("Video decoding and the LeRobot writer have NOT been tested by inspect-only.")
        return
    if args.output is None:
        parser.error("--output is required unless --inspect-only is used")
    convert(source, args.output, repo_id=args.repo_id, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
