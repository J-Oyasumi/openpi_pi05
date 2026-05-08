"""Convert the user-collected `3tasks_nosvo` real-world data to LeRobot v2 format
expected by openpi pi0.5 training.

Source layout (per task folder):
    3tasks_nosvo/<task_name>/traj_<timestamp>/
        actions_49frames.npz   # 49-frame aligned state/action arrays (q, gripper_desired, ...)
        data.h5                # raw 100Hz teleop log (not used; npz is already aligned)
        left_49frames_16fps.mp4  # ZED LEFT camera, 1280x720, 16 fps, 49 frames
        zed_timestamps.json    # source frame timestamps (not used)

Output: a LeRobot dataset with these features:
    observation.images.left : (3,H,W) video / image  # ZED LEFT
    observation.state       : float32 (8,)  # [q(7), gripper_pos]
    action                  : float32 (8,)  # [q(7), gripper_desired]
    task                    : per-trajectory language prompt (from task name)

Usage:
    uv run examples/3tasks_nosvo/convert_3tasks_to_lerobot.py \
        --raw-dir /home/hanjiang/clone/openpi/3tasks_nosvo \
        --repo-id hanjiang/three_tasks_nosvo
"""

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from rich import print as rprint


TASK_PROMPTS = {
    "pineapple": "pick up the pineapple and place it in the bowl",
    "stackcube": "stack the cubes",
    "stackcups": "stack the cups",
}

FPS = 16
NUM_FRAMES = 49
IMG_H = 720
IMG_W = 1280


@dataclasses.dataclass
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-3  # relax: 16 fps -> 62.5 ms per frame
    image_writer_processes: int = 4
    image_writer_threads: int = 4
    video_backend: str | None = None


def _read_video_frames(mp4_path: Path, n_frames: int) -> np.ndarray:
    """Decode mp4 to ndarray (n_frames, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    arr = np.stack(frames, axis=0)
    if arr.shape[0] != n_frames:
        raise ValueError(f"{mp4_path}: expected {n_frames} frames, got {arr.shape[0]}")
    return arr


def _load_traj(traj_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (state, action, images) for one trajectory."""
    npz = np.load(traj_dir / "actions_49frames.npz")
    q = npz["q"].astype(np.float32)            # (N,7)
    gripper_pos = npz["gripper_pos"].astype(np.float32)        # (N,)
    gripper_desired = npz["gripper_desired"].astype(np.float32)  # (N,)

    # State: [q, gripper_pos] -> (N, 8)
    state = np.concatenate([q, gripper_pos[:, None]], axis=-1).astype(np.float32)
    # Action: next-step desired joint targets [q, gripper_desired] -> (N, 8)
    # NOTE: q here is the recorded joint position at the same timestep as the action chunk
    # source. We treat it as the proprio target (consistent with Aloha-style "next qpos" labelling
    # collected from a teleop trajectory).
    action = np.concatenate([q, gripper_desired[:, None]], axis=-1).astype(np.float32)

    images = _read_video_frames(traj_dir / "left_49frames_16fps.mp4", n_frames=q.shape[0])
    return state, action, images


def create_empty_dataset(
    repo_id: str,
    *,
    mode: Literal["video", "image"] = "video",
    cfg: DatasetConfig = DatasetConfig(),
) -> LeRobotDataset:
    state_dim = 8
    action_dim = 8
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [[f"joint_{i}" for i in range(7)] + ["gripper"]],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [[f"joint_{i}" for i in range(7)] + ["gripper"]],
        },
        "observation.images.left": {
            "dtype": mode,
            "shape": (3, IMG_H, IMG_W),
            "names": ["channels", "height", "width"],
        },
    }

    out_dir = HF_LEROBOT_HOME / repo_id
    if out_dir.exists():
        rprint(f"[yellow]Removing existing dataset at {out_dir}[/yellow]")
        shutil.rmtree(out_dir)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="xarm7",
        features=features,
        use_videos=cfg.use_videos,
        tolerance_s=cfg.tolerance_s,
        image_writer_processes=cfg.image_writer_processes,
        image_writer_threads=cfg.image_writer_threads,
        video_backend=cfg.video_backend,
    )


def populate_dataset(dataset: LeRobotDataset, raw_dir: Path) -> LeRobotDataset:
    total_episodes = 0
    for task_name, prompt in TASK_PROMPTS.items():
        task_dir = raw_dir / task_name
        if not task_dir.exists():
            rprint(f"[red]Skip missing task dir: {task_dir}[/red]")
            continue
        all_traj_dirs = sorted(task_dir.glob("traj_*"))
        traj_dirs = [
            t for t in all_traj_dirs
            if (t / "actions_49frames.npz").exists() and (t / "left_49frames_16fps.mp4").exists()
        ]
        skipped = len(all_traj_dirs) - len(traj_dirs)
        rprint(
            f"[cyan]== {task_name} ==[/cyan] {len(traj_dirs)} trajectories"
            f"{f' (skipped {skipped} incomplete)' if skipped else ''} | prompt='{prompt}'"
        )
        for traj_dir in tqdm.tqdm(traj_dirs, desc=task_name):
            state, action, images = _load_traj(traj_dir)
            n = state.shape[0]
            for i in range(n):
                # LeRobot expects (C,H,W) for video features.
                img_chw = np.transpose(images[i], (2, 0, 1))  # (3,H,W)
                dataset.add_frame(
                    {
                        "observation.state": state[i],
                        "action": action[i],
                        "observation.images.left": img_chw,
                        "task": prompt,
                    }
                )
            dataset.save_episode()
            total_episodes += 1
    rprint(f"[green]Saved {total_episodes} episodes total[/green]")
    return dataset


def main(
    raw_dir: Path = Path("/home/hanjiang/clone/openpi/3tasks_nosvo"),
    repo_id: str = "hanjiang/three_tasks_nosvo",
    mode: Literal["video", "image"] = "video",
    push_to_hub: bool = False,
):
    rprint(f"[bold blue]raw_dir[/bold blue]={raw_dir}")
    rprint(f"[bold blue]repo_id[/bold blue]={repo_id}")
    rprint(f"[bold blue]mode[/bold blue]={mode}")

    dataset = create_empty_dataset(repo_id, mode=mode)
    populate_dataset(dataset, raw_dir)
    # NOTE: lerobot v2.1 (the version pinned by openpi) no longer requires/exposes
    # `consolidate()` -- save_episode now writes the per-episode parquet/video as it goes.
    rprint(f"[green]Dataset written to {HF_LEROBOT_HOME / repo_id}[/green]")

    if push_to_hub:
        dataset.push_to_hub(tags=["xarm7", "real-world"], private=False, push_videos=True)


if __name__ == "__main__":
    tyro.cli(main)
