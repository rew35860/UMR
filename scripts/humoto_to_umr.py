#!/usr/bin/env python3
"""Convert HUMOTO (InterAct layout) sequences into UMR's flat HSI/HOI layout.

HUMOTO in InterAct form gives SMPL-H, not SMPL-X: ``poses`` is (T, 156) = 52
joints, where SMPL-X wants 165 = 55. The difference is exactly the three face
joints, so the remap is lossless for every joint that exists:

    poses_x[  0: 66] = poses_h[  0: 66]     # root + 21 body
    poses_x[ 66: 75] = 0                    # jaw, left eye, right eye
    poses_x[ 75:165] = poses_h[ 66:156]     # 30 hand joints

Unlike OMOMO, the hand joints here are NON-ZERO -- HUMOTO carries real finger
articulation, which survives this conversion untouched.

Source conventions (verified against the data, not the docs): metres, Y-up,
30 fps; object pose as axis-angle ``angles`` (T,3) + ``trans`` (T,3) applied to
a mesh already in metres about its own origin. ``output_up="y"`` is written so
UMR performs the Y-up -> Z-up conversion itself (the same path its OmniContact
sample uses).

    python scripts/humoto_to_umr.py --seq-key moving_low_chair_with_both_hands-723

UMR's HSI/HOI pipeline takes ONE object per sequence; for a multi-object clip
pass --object to choose which, otherwise the first is used and the rest dropped.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
INTERACT = Path("/home/natcha/Downloads/viser_deploy/deploy_data/data/interact")
DATASETS = ("humoto_full", "humoto_public")
FPS = 30.0                 # InterAct HUMOTO carries no fps field; the viewer assumes 30
UP_AXIS = "y"              # metres, Y-up -- UMR converts to its Z-up retarget frame
COACD_THRESHOLD = 0.05
SMPLH_WIDTH, SMPLX_WIDTH = 156, 165


def find_sequence(seq: str) -> Path:
    for ds in DATASETS:
        p = INTERACT / ds / "sequences" / seq
        if (p / "human.npz").exists():
            return p
    raise FileNotFoundError(f"sequence not found in {DATASETS}: {seq}")


def find_mesh(obj: str, seq_dir: Path) -> Path:
    for root in (seq_dir.parent.parent / "objects", INTERACT / "objects"):
        p = root / obj / f"{obj}.obj"
        if p.exists():
            return p
    raise FileNotFoundError(f"object mesh not found for {obj!r}")


# ---------------------------------------------------------------------------
# Step 1: SMPL-H (156) -> SMPL-X (165), inserting the three zero face joints
# ---------------------------------------------------------------------------
def smplh_to_smplx(poses: np.ndarray) -> np.ndarray:
    if poses.shape[1] == SMPLX_WIDTH:
        return poses.astype(np.float32)
    if poses.shape[1] != SMPLH_WIDTH:
        raise ValueError(f"expected SMPL-H {SMPLH_WIDTH} or SMPL-X {SMPLX_WIDTH} pose width, got {poses.shape[1]}")
    out = np.zeros((len(poses), SMPLX_WIDTH), dtype=np.float32)
    out[:, 0:66] = poses[:, 0:66]        # root + body
    out[:, 75:165] = poses[:, 66:156]    # hands; [66:75] stays zero (jaw/eyes)
    return out


# ---------------------------------------------------------------------------
# Step 2: object collision assets, shared per object across sequences
# ---------------------------------------------------------------------------
def prepare_object_assets(obj: str, mesh_src: Path, assets_dir: Path, force: bool) -> tuple[Path, list[Path]]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    visual = assets_dir / f"{obj}_visual.obj"
    pieces = sorted(assets_dir.glob(f"{obj}_collision_*.obj"))
    if visual.exists() and pieces and not force:
        print(f"    reuse assets for {obj}: {len(pieces)} collision parts")
        return visual, pieces

    mesh = trimesh.load(mesh_src, force="mesh")
    # UMR converts the object POSE from Y-up to Z-up by conjugation
    # (R_z = B R_y B^T). That only reproduces the source transform if the mesh's
    # LOCAL vertices are rotated by B as well, so pre-rotate here:
    #   B: (x, y, z) -> (x, -z, y)     [y_up_to_z_up_matrix("y", convert=True)]
    # Without this the object's shape sits 90 deg off its own trajectory.
    v = np.asarray(mesh.vertices, np.float64)
    mesh.vertices = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
    mesh.export(visual)
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32)),
        threshold=COACD_THRESHOLD,
    )
    for stale in pieces:
        stale.unlink()
    out = []
    for i, (v, f) in enumerate(parts):
        p = assets_dir / f"{obj}_collision_{i}.obj"
        trimesh.Trimesh(vertices=v, faces=f).export(p)
        out.append(p)
    print(f"    CoACD {obj}: {len(out)} convex parts")
    return visual, out


def write_object_xml(obj: str, seq_dir: Path, assets_dir: Path, visual: Path, pieces: list[Path]) -> Path:
    # meshes are already in metres about their own origin, so scale stays 1
    meshdir = os.path.relpath(assets_dir, seq_dir)
    lines = [f'<mujoco model="{obj}">',
             f'  <compiler angle="radian" meshdir="{meshdir}" />', "  <asset>",
             f'    <mesh name="{obj}_visual_mesh" file="{visual.name}" scale="1 1 1" />']
    for i, p in enumerate(pieces):
        lines.append(f'    <mesh name="{obj}_collision_{i}_mesh" file="{p.name}" scale="1 1 1" />')
    lines += ["  </asset>", "  <worldbody>", f'    <body name="{obj}">',
              f'      <freejoint name="{obj}_freejoint" />',
              f'      <geom name="{obj}_visual" type="mesh" mesh="{obj}_visual_mesh"'
              ' group="2" contype="0" conaffinity="0" rgba="1 1 1 1" />']
    for i in range(len(pieces)):
        lines.append(f'      <geom name="{obj}_collision_{i}" type="mesh" mesh="{obj}_collision_{i}_mesh"'
                     ' group="3" contype="1" conaffinity="1" rgba="0.25 0.45 0.8 0.15" />')
    lines += ["    </body>", "  </worldbody>", "</mujoco>", ""]
    path = seq_dir / f"{obj}.xml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def convert(seq: str, out_root: Path, object_name: str | None, force: bool) -> Path | None:
    src = find_sequence(seq)
    human = dict(np.load(src / "human.npz", allow_pickle=True))
    poses = smplh_to_smplx(np.asarray(human["poses"], np.float32))
    trans = np.asarray(human["trans"], np.float32).reshape(len(poses), 3)
    betas = np.asarray(human["betas"], np.float32).reshape(-1)[:10]
    gender = str(human["gender"]).lower() or "neutral"

    objs = sorted(p.stem.removeprefix("object_") for p in src.glob("object_*.npz"))
    if not objs:
        print(f"[HUMOTO] SKIP {seq}: no object_*.npz")
        return None
    obj = object_name or objs[0]
    if obj not in objs:
        raise ValueError(f"{seq}: object {obj!r} not in {objs}")
    if len(objs) > 1:
        print(f"[HUMOTO] {seq}: {len(objs)} objects {objs}; using {obj!r}, dropping the rest")

    seq_dir = out_root / "train_and_test" / seq
    seq_dir.mkdir(parents=True, exist_ok=True)
    np.save(seq_dir / "poses.npy", poses)
    np.save(seq_dir / "transl.npy", trans)
    np.save(seq_dir / "betas.npy", betas)
    np.save(seq_dir / "gender.npy", np.array(gender))
    np.save(seq_dir / "model_type.npy", np.array("smplx"))
    np.save(seq_dir / "mocap_framerate.npy", np.array(FPS, dtype=np.float32))
    np.save(seq_dir / "output_up.npy", np.array(UP_AXIS))

    od = dict(np.load(src / f"object_{obj}.npz", allow_pickle=True))
    pos = np.asarray(od["trans"], np.float64).reshape(-1, 3)
    quat = R.from_rotvec(np.asarray(od["angles"], np.float64).reshape(-1, 3)).as_quat()   # xyzw
    n = min(len(pos), len(poses))
    with (seq_dir / f"prop_{obj}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["px", "py", "pz", "qx", "qy", "qz", "qw"])
        for i in range(n):
            w.writerow([*pos[i].tolist(), *quat[i].tolist()])

    visual, pieces = prepare_object_assets(obj, find_mesh(obj, src), out_root / "object_mjcf" / "assets", force)
    write_object_xml(obj, seq_dir, out_root / "object_mjcf" / "assets", visual, pieces)
    hands_live = bool(np.abs(poses[:, 75:165]).max() > 1e-6)
    print(f"[HUMOTO] {seq}: frames={n} object={obj} gender={gender} "
          f"fingers={'yes' if hands_live else 'flat'} -> {seq_dir}")
    return seq_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-key", action="append", default=[], help="Sequence name; repeatable.")
    ap.add_argument("--object", action="append", default=[], help="Object per --seq-key (same order); optional.")
    ap.add_argument("--out-root", type=Path, default=ROOT / "sample_data" / "humoto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.seq_key:
        raise SystemExit("pass at least one --seq-key")
    objs = list(args.object) + [None] * (len(args.seq_key) - len(args.object))
    made = [p for s, o in zip(args.seq_key, objs)
            if (p := convert(s, args.out_root.resolve(), o, args.force))]
    print(f"\n[HUMOTO] converted {len(made)}/{len(args.seq_key)} into {args.out_root}")


if __name__ == "__main__":
    main()
