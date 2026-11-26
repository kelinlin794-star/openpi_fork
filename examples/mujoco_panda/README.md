# Historical single-arm Panda pick → π0.5

This adapter targets `shuyisong07/merged_data`, subdirectory
`act_dp_mujoco_panda_pick_20261022_hand_delta_ema` (50 episodes, 26,525 frames,
30 FPS). It is not the Push-T dataset and not a generic Paper A Training Pack
reader. Commands below run in the **server's OpenPI project directory**, after
installing the project's locked dependencies.

## 1. Download the source

```bash
uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="shuyisong07/merged_data", repo_type="dataset", allow_patterns=["act_dp_mujoco_panda_pick_20261022_hand_delta_ema/meta/**", "act_dp_mujoco_panda_pick_20261022_hand_delta_ema/data/**", "act_dp_mujoco_panda_pick_20261022_hand_delta_ema/videos/observation.images.overview/**", "act_dp_mujoco_panda_pick_20261022_hand_delta_ema/merge_manifest.json"], local_dir="/zhouhuayi/datasets/merged_data")'
```

Only overview is needed by this baseline; the two other cameras and hand videos
are not downloaded. The source directory must remain unchanged. Paths below are
examples; substitute your actual storage directory.

## 2. Convert the storage format

The source is LeRobot **v3.0**; OpenPI pins a **v2.1** reader. The converter uses
the v3 episode metadata's video offsets and decoded presentation timestamps. It
keeps every recorded state/action row in its original episode and verifies its
frame index and nominal timestamp. It does not crop, shift action labels,
interpolate, or reinterpret the recorded target as a next-control-tick target.

```bash
uv run python scripts/convert_mujoco_pick_v3_to_v21.py \
  --source /zhouhuayi/datasets/merged_data \
  --inspect-only

# First run a one-episode engineering check in its own output directory.
uv run python scripts/convert_mujoco_pick_v3_to_v21.py \
  --source /zhouhuayi/datasets/merged_data \
  --output /zhouhuayi/datasets/pick_v21_smoke \
  --max-episodes 1

# Full conversion uses a DIFFERENT output directory.
uv run python scripts/convert_mujoco_pick_v3_to_v21.py \
  --source /zhouhuayi/datasets/merged_data \
  --output /zhouhuayi/datasets/pick_v21
```

Output is written with the pinned LeRobot writer. Decoded overview RGB is stored
as lossless embedded images instead of being re-encoded into a second lossy MP4.
This can use more disk space than the original video. All numeric observation
features are retained; the other two cameras are omitted. The source is never
overwritten. Existing output directories, including incomplete runs, are refused:
choose a new output directory after a failed conversion. `conversion.json` records
source metadata identity, per-episode counts, completion and engineering subsets.

Gripper commands stay **0..255** in the converted dataset. `MujocoPandaInputs`
converts them to metres exactly once during model preprocessing. The original
state's measured finger width is already in metres and is not changed.

## 3. Verify the reader and compute normalization

```bash
export MUJOCO_PANDA_DATA_ROOT=/zhouhuayi/datasets/pick_v21
uv run python -c 'import os; from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset("local/mujoco_panda_pick_v21", root=os.environ["MUJOCO_PANDA_DATA_ROOT"]); x=d[0]; print("episodes", d.num_episodes, "frames", len(d)); print("state", x["observation.state"].shape, "action", x["action"].shape, "image", x["observation.images.overview"].shape); print("task", x["task"])'
uv run scripts/compute_norm_stats.py --config-name pi05_mujoco_panda_pick
```

The full converted dataset should have 50 episodes and 26,525 frames. Reader
images may be float CHW; the input adapter handles CHW/HWC. Recompute OpenPI's
normalization after conversion because the input adapter changes action units;
do not copy raw HF statistics as model normalization.

This prepares training data. It does not download π0.5 model weights, start a
training run or establish simulation success. At inference, the existing output
adapter returns raw 0..255 gripper commands, so it must connect to a controller
expecting those units; a metric controller needs its own output mapping.

## Can the team's Training Pack be reused?

Yes as a data source, but not by setting its directory as `dataset_root` here.
Its `manifest.json`, `arrays.npz`, `rgb_oblique.npy` and certified sequence
boundaries use a different reader contract. The historical recorded-row pick
Pack already stores metric gripper actions. It therefore needs a separate
importer/input mapping, without another 0..255-to-metres conversion. For a curated
Safe Stitch Pack, action masks, hard sequence boundaries and the frozen split
must also be preserved. This converter explicitly rejects those directories;
it does not silently flatten a Pack into continuous demonstration sequences.
