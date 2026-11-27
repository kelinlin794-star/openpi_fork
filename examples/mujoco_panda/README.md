# Historical single-arm Panda pick → π0.5

This adapter targets `shuyisong07/merged_data`, subdirectory
`act_dp_mujoco_panda_pick_20261022_hand_delta_ema` (50 episodes, 26,525 frames,
30 FPS). It is not the Push-T dataset and not a generic Paper A Training Pack
reader. Commands below run in the **server's OpenPI project directory**, after
installing the project's locked dependencies.

## Current experiment: three external views, one arm

Use **`pi05_mujoco_panda_pick_3view`** for the three-camera experiment. The old
single-view config below is retained as a separate experiment. State S9, action
A7, raw gripper conversion, task text and official JAX LoRA implementation stay
the same. This changes the cameras supplied to the model, not its architecture.

| Source camera | Input adapter field | Official model slot | Valid mask |
| --- | --- | --- | --- |
| `observation.images.overview` | `observation/overview` | `base_0_rgb` | True |
| `observation.images.camera_2` | `observation/camera_2` | `left_wrist_0_rgb` | True |
| `observation.images.camera_3` | `observation/camera_3` | `right_wrist_0_rgb` | True |

All three cameras are external views. The `wrist` names are reused interface
slots, not claims about physical camera positions. The official JAX preprocessing
in `src/openpi/models/model.py` checks these names: the base slot receives spatial
crop/rotation plus color augmentation, while the two wrist slots receive color
augmentation only. This experiment keeps that official behavior unchanged. Use
the same mapping for training, evaluation and deployment. A missing camera is an
error in three-view mode; it is never silently replaced by a valid black image.

Run the following commands in the **server** project after activating its venv.
The earlier single-view download command did not request the other two videos;
this command reuses matching downloaded files and adds the missing cameras:

```bash
source .venv/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download
folder = "act_dp_mujoco_panda_pick_20261022_hand_delta_ema"
snapshot_download(
    repo_id="shuyisong07/merged_data", repo_type="dataset",
    allow_patterns=[f"{folder}/meta/**", f"{folder}/data/**", f"{folder}/merge_manifest.json",
                    *[f"{folder}/videos/observation.images.{camera}/**"
                      for camera in ("overview", "camera_2", "camera_3")]],
    local_dir="/zhouhuayi/datasets/merged_data", max_workers=4,
)
PY

# Check metadata, then exercise the real converter on one episode.
python scripts/convert_mujoco_pick_v3_to_v21.py \
  --source /zhouhuayi/datasets/merged_data --three-views --inspect-only
python scripts/convert_mujoco_pick_v3_to_v21.py \
  --source /zhouhuayi/datasets/merged_data --three-views \
  --output /zhouhuayi/datasets/pick_v21_3view_smoke --max-episodes 1

export MUJOCO_PANDA_DATA_ROOT=/zhouhuayi/datasets/pick_v21_3view_smoke
python scripts/check_mujoco_pick_dataset.py --config-name pi05_mujoco_panda_pick_3view
python scripts/compute_norm_stats.py --config-name pi05_mujoco_panda_pick_3view
```

The converter uses **each camera's own episode video offset and decoded timestamp**
to pair the three images with the same numeric row. It exports all selected views
as lossless embedded images. The three-view output uses its own repo ID,
`local/mujoco_panda_pick_3view_v21`, and must be written to a fresh output folder;
the single-view conversion omitted two cameras and cannot be used as-is.

After the data check and normalization succeed, verify that the official **JAX**
`gs://openpi-assets/checkpoints/pi05_base/params` checkpoint is complete. A
PyTorch `model.safetensors` file is not a substitute for this loader. The training
loader attempts to download absent weights, so a working proxy is needed if the
checkpoint or tokenizer resource is not cached. Check the reader, GPU memory and
loss with a **two-step test**, using a new experiment name if rerunning:

```bash
WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/train.py pi05_mujoco_panda_pick_3view \
  --exp-name pick_3view_smoke --num-train-steps 2 --log-interval 1 --num-workers 0
```

This is a pipeline check, not a trained baseline. Only after it succeeds, convert
all 50 episodes into a different directory and recompute normalization on that
full dataset before the long run:

```bash
python scripts/convert_mujoco_pick_v3_to_v21.py \
  --source /zhouhuayi/datasets/merged_data --three-views \
  --output /zhouhuayi/datasets/pick_v21_3view
export MUJOCO_PANDA_DATA_ROOT=/zhouhuayi/datasets/pick_v21_3view
python scripts/check_mujoco_pick_dataset.py --config-name pi05_mujoco_panda_pick_3view
python scripts/compute_norm_stats.py --config-name pi05_mujoco_panda_pick_3view
```

The config defaults to batch size 1, 20,000 steps and LoRA. Its suitability for a
particular GPU is established by the actual training test, not the earlier small
matrix multiplication check. The commands above are a workflow; they do not
mean a real three-view data conversion or training run has already passed.

## Previous single-view single-arm baseline

The existing configuration name is **`pi05_mujoco_panda_pick`**. It selects only
`observation.images.overview`, mapped to `base_0_rgb`, together with the real
single-arm S9 state and A7 action. `camera_2`, `camera_3` and gesture labels do
not enter the model. Conversion already exports only this one camera.

OpenPI's official model interface requires three image slots. The two missing
wrist views remain black placeholders with `image_mask=False`; only overview
has `image_mask=True`. They are not additional observed camera views. This keeps
the official architecture and pretrained-weight structure intact; it does not
eliminate the computation on the padded image slots.

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
uv run python scripts/check_mujoco_pick_dataset.py
uv run python -c 'import os; from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset("local/mujoco_panda_pick_v21", root=os.environ["MUJOCO_PANDA_DATA_ROOT"]); x=d[0]; print("episodes", d.num_episodes, "frames", len(d)); print("state", x["observation.state"].shape, "action", x["action"].shape, "image", x["observation.images.overview"].shape); print("task", x["task"])'
uv run scripts/compute_norm_stats.py --config-name pi05_mujoco_panda_pick
```

The full converted dataset should have 50 episodes and 26,525 frames. Reader
images may be float CHW; the input adapter handles CHW/HWC. Recompute OpenPI's
normalization after conversion because the input adapter changes action units;
do not copy raw HF statistics as model normalization.

The check script reads the first, middle and last samples through the training
reader and input transforms. It verifies a single valid camera, black masked
wrist placeholders, finite S9/A7 values, the full action horizon, and gripper
conversion exactly once. It does not instantiate the model or test task success.
Creating the training data configuration also initializes the official tokenizer;
its small resource file may be downloaded on first use, so keep the proxy working.

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
