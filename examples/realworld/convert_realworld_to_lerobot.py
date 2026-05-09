"""Merge multiple raw xArm7 collection roots into a single LeRobot dataset.

Sources (per the user's `data/` layout):
    data/3tasks_nosvo/{pineapple, stackcube, stackcups}/traj_*/...
    data/extra/extra_data/{stackcube2, stackcup2}/traj_*/...

Each `traj_*` is expected to contain:
    actions_49frames.npz      (q, gripper_pos, gripper_desired, ...)
    left_49frames_16fps.mp4   (1280x720, 49 frames @ 16 fps, ZED LEFT)

Anything else (svo, raw 100Hz h5, ...) is ignored. Trajectories that miss the npz
or the mp4 are skipped (e.g. the one stackcups traj that was incomplete).

Output:
    $HF_LEROBOT_HOME/hanjiang/realworld

Run:
    uv run python examples/realworld/convert_realworld_to_lerobot.py \
        --data-root /home/hanjiang/clone/openpi/data \
        --repo-id hanjiang/realworld
"""

import dataclasses
import shutil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from rich import print as rprint


# (relative-source-dir, raw-task-folder, prompt) tuples.
# The "*2" variants are folded into the same prompt as their base task.
SOURCES: list[tuple[str, str, str]] = [
    ("3tasks_nosvo",      "pineapple",  "pick up the pineapple and place it in the bowl"),
    ("3tasks_nosvo",      "stackcube",  "stack the cubes"),
    ("3tasks_nosvo",      "stackcups",  "stack the cups"),
    ("extra/extra_data",  "stackcube2", "stack the cubes"),
    ("extra/extra_data",  "stackcup2",  "stack the cups"),
]

FPS = 16
IMG_H = 720
IMG_W = 1280


@dataclasses.dataclass
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-3
    image_writer_processes: int = 4
    image_writer_threads: int = 4
    video_backend: str | None = None


def _read_video_frames(mp4_path: Path, n_frames: int) -> np.ndarray:
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
    npz = np.load(traj_dir / "actions_49frames.npz")
    q = npz["q"].astype(np.float32)
    gripper_pos = npz["gripper_pos"].astype(np.float32)
    gripper_desired = npz["gripper_desired"].astype(np.float32)

    state = np.concatenate([q, gripper_pos[:, None]], axis=-1).astype(np.float32)
    action = np.concatenate([q, gripper_desired[:, None]], axis=-1).astype(np.float32)

    images = _read_video_frames(traj_dir / "left_49frames_16fps.mp4", n_frames=q.shape[0])
    return state, action, images


def create_empty_dataset(
    repo_id: str,
    *,
    mode: Literal["video", "image"] = "video",
    cfg: DatasetConfig = DatasetConfig(),
) -> LeRobotDataset:
    state_dim = action_dim = 8
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


def populate_dataset(dataset: LeRobotDataset, data_root: Path) -> int:
    total_episodes = 0
    total_frames = 0
    for rel_root, task_name, prompt in SOURCES:
        task_dir = data_root / rel_root / task_name
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
            f"[cyan]== {rel_root}/{task_name} ==[/cyan] {len(traj_dirs)} trajectories"
            f"{f' (skipped {skipped} incomplete)' if skipped else ''} | prompt='{prompt}'"
        )

        for traj_dir in tqdm.tqdm(traj_dirs, desc=f"{rel_root}/{task_name}"):
            state, action, images = _load_traj(traj_dir)
            n = state.shape[0]
            for i in range(n):
                img_chw = np.transpose(images[i], (2, 0, 1))
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
            total_frames += n

    rprint(f"[green]Saved {total_episodes} episodes, {total_frames} frames total[/green]")
    return total_episodes


def main(
    data_root: Path = Path("/home/hanjiang/clone/openpi/data"),
    repo_id: str = "hanjiang/realworld",
    mode: Literal["video", "image"] = "video",
    push_to_hub: bool = False,
):
    rprint(f"[bold blue]data_root[/bold blue]={data_root}")
    rprint(f"[bold blue]repo_id[/bold blue]={repo_id}")
    rprint(f"[bold blue]mode[/bold blue]={mode}")

    dataset = create_empty_dataset(repo_id, mode=mode)
    populate_dataset(dataset, data_root)
    rprint(f"[green]Dataset written to {HF_LEROBOT_HOME / repo_id}[/green]")

    if push_to_hub:
        dataset.push_to_hub(tags=["xarm7", "real-world"], private=False, push_videos=True)


if __name__ == "__main__":
    tyro.cli(main)
