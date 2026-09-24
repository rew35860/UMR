#!/usr/bin/env python3
"""Convert OMOMO sequences into UMR's flat SMPL-X plus object-trajectory layout.

OMOMO publishes every sequence inside one pickle keyed by an integer index, with
the object meshes kept separately under ``captured_objects/``. The HSI/HOI
pipeline instead reads one directory per sequence. This script bridges the two,
following the layout documented in ``sample_data/omomo/README.md``.

    python scripts/omomo_to_umr.py \
        --omomo-root ../omomo_release/data \
        --seq-key sub1_plasticbox_015

Object meshes are shared between sequences under ``object_mjcf/assets/`` and are
convex-decomposed once with CoACD; MuJoCo otherwise replaces a mesh collision
geom with its convex hull, which lets the robot's hands sink into the object.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent

# SMPL-X axis-angle layout: 55 joints x 3. OMOMO only provides the root
# orientation and the 21 body joints; jaw, eyes and both hands stay at zero.
SMPLX_NUM_JOINTS = 55
SMPLX_POSE_WIDTH = SMPLX_NUM_JOINTS * 3
OMOMO_BODY_JOINTS = 21
OMOMO_FPS = 30.0
OMOMO_UP_AXIS = "z"
SPLIT_FILES = {
    "train": "train_diffusion_manip_seq_joints24.p",
    "test": "test_diffusion_manip_seq_joints24.p",
}


# ---------------------------------------------------------------------------
# Step 1: index the OMOMO pickles by sequence name
# ---------------------------------------------------------------------------
def load_sequence_index(omomo_root: Path, splits: list[str]) -> dict[str, dict]:
    """Map ``seq_name -> sequence dict`` across the requested splits."""
    index: dict[str, dict] = {}
    for split in splits:
        path = omomo_root / SPLIT_FILES[split]
        if not path.exists():
            raise FileNotFoundError(f"OMOMO split not found: {path}")
        print(f"[OMOMO] loading {split} split: {path}")
        payload = joblib.load(path)
        for entry in payload.values():
            index[str(np.asarray(entry["seq_name"]).item() if np.ndim(entry["seq_name"]) else entry["seq_name"])] = entry
        print(f"[OMOMO]   {split}: {len(payload)} sequences")
    return index


def scalar_str(value) -> str:
    """OMOMO stores gender as a 0-d numpy string array."""
    array = np.asarray(value)
    return str(array.item() if array.ndim == 0 else array.reshape(-1)[0])


# ---------------------------------------------------------------------------
# Step 2: human motion -> poses/transl/betas plus the scalar metadata files
# ---------------------------------------------------------------------------
def write_human_motion(entry: dict, seq_dir: Path) -> int:
    root_orient = np.asarray(entry["root_orient"], dtype=np.float32).reshape(-1, 3)
    pose_body = np.asarray(entry["pose_body"], dtype=np.float32).reshape(-1, OMOMO_BODY_JOINTS, 3)
    trans = np.asarray(entry["trans"], dtype=np.float32).reshape(-1, 3)

    frames = len(root_orient)
    if not (len(pose_body) == len(trans) == frames):
        raise ValueError(
            f"frame count mismatch: root_orient={len(root_orient)} "
            f"pose_body={len(pose_body)} trans={len(trans)}"
        )

    poses = np.zeros((frames, SMPLX_NUM_JOINTS, 3), dtype=np.float32)
    poses[:, 0] = root_orient
    poses[:, 1 : 1 + OMOMO_BODY_JOINTS] = pose_body

    # UMR truncates to the first 10 shape components; OMOMO stores 16.
    betas = np.asarray(entry["betas"], dtype=np.float32).reshape(-1)[:10]

    np.save(seq_dir / "poses.npy", poses.reshape(frames, SMPLX_POSE_WIDTH))
    np.save(seq_dir / "transl.npy", trans)
    np.save(seq_dir / "betas.npy", betas)
    np.save(seq_dir / "gender.npy", np.array(scalar_str(entry["gender"]).lower()))
    np.save(seq_dir / "model_type.npy", np.array("smplx"))
    np.save(seq_dir / "mocap_framerate.npy", np.array(OMOMO_FPS, dtype=np.float32))
    np.save(seq_dir / "output_up.npy", np.array(OMOMO_UP_AXIS))
    return frames


# ---------------------------------------------------------------------------
# Step 3: shared object assets (visual copy + CoACD collision pieces)
# ---------------------------------------------------------------------------
def prepare_object_assets(
    object_name: str,
    omomo_root: Path,
    assets_dir: Path,
    threshold: float,
    force: bool,
) -> tuple[Path, list[Path]]:
    source = omomo_root / "captured_objects" / f"{object_name}_cleaned_simplified.obj"
    if not source.exists():
        raise FileNotFoundError(f"OMOMO object mesh not found: {source}")

    assets_dir.mkdir(parents=True, exist_ok=True)
    visual = assets_dir / f"{object_name}_visual.obj"
    existing = sorted(assets_dir.glob(f"{object_name}_collision_*.obj"))
    if visual.exists() and existing and not force:
        print(f"[OMOMO]   reuse assets for {object_name}: {len(existing)} collision parts")
        return visual, existing

    mesh = trimesh.load(source, force="mesh")
    mesh.export(visual)

    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)),
        threshold=threshold,
    )
    for stale in existing:
        stale.unlink()
    collisions = []
    for index, (vertices, faces) in enumerate(parts):
        part_path = assets_dir / f"{object_name}_collision_{index}.obj"
        trimesh.Trimesh(vertices=vertices, faces=faces).export(part_path)
        collisions.append(part_path)
    print(f"[OMOMO]   CoACD {object_name}: {len(collisions)} convex parts (threshold={threshold})")
    return visual, collisions


# ---------------------------------------------------------------------------
# Step 4: per-sequence MJCF. Mesh scale is an MJCF attribute, so the shared
# asset stays unscaled and each sequence applies its own factor.
# ---------------------------------------------------------------------------
def write_object_xml(
    object_name: str,
    seq_dir: Path,
    assets_dir: Path,
    visual: Path,
    collisions: list[Path],
    scale: float,
) -> Path:
    meshdir = os.path.relpath(assets_dir, seq_dir)
    scale_attr = f"{scale:.9g} {scale:.9g} {scale:.9g}"
    lines = [
        f'<mujoco model="{object_name}">',
        f'  <compiler angle="radian" meshdir="{meshdir}" />',
        "  <asset>",
        f'    <mesh name="{object_name}_visual_mesh" file="{visual.name}" scale="{scale_attr}" />',
    ]
    for index, part in enumerate(collisions):
        lines.append(
            f'    <mesh name="{object_name}_collision_{index}_mesh" file="{part.name}" scale="{scale_attr}" />'
        )
    lines += [
        "  </asset>",
        "  <worldbody>",
        f'    <body name="{object_name}">',
        f'      <freejoint name="{object_name}_freejoint" />',
        f'      <geom name="{object_name}_visual" type="mesh" mesh="{object_name}_visual_mesh"'
        ' group="2" contype="0" conaffinity="0" rgba="1 1 1 1" />',
    ]
    for index in range(len(collisions)):
        lines.append(
            f'      <geom name="{object_name}_collision_{index}" type="mesh"'
            f' mesh="{object_name}_collision_{index}_mesh" group="3" contype="1" conaffinity="1"'
            ' rgba="0.25 0.45 0.8 0.15" />'
        )
    lines += ["    </body>", "  </worldbody>", "</mujoco>", ""]

    xml_path = seq_dir / f"{object_name}.xml"
    xml_path.write_text("\n".join(lines), encoding="utf-8")
    return xml_path


# ---------------------------------------------------------------------------
# Step 5: per-frame object pose. OMOMO applies scale * R @ v + t; the uniform
# scale moves into the MJCF mesh, leaving translation and rotation here.
# ---------------------------------------------------------------------------
def write_object_prop(entry: dict, object_name: str, seq_dir: Path, frames: int) -> float:
    positions = np.asarray(entry["obj_trans"], dtype=np.float64).reshape(frames, 3)
    rotations = np.asarray(entry["obj_rot"], dtype=np.float64).reshape(frames, 3, 3)
    quats = Rotation.from_matrix(rotations).as_quat()  # (x, y, z, w)

    prop_path = seq_dir / f"prop_{object_name}.csv"
    with prop_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["px", "py", "pz", "qx", "qy", "qz", "qw"])
        for position, quat in zip(positions, quats):
            writer.writerow([*position.tolist(), *quat.tolist()])

    scales = np.asarray(entry["obj_scale"], dtype=np.float64).reshape(-1)
    spread = float((scales.max() - scales.min()) / max(abs(scales.mean()), 1e-9))
    if spread > 0.05:
        print(f"[OMOMO]   WARNING obj_scale varies {spread:.1%} within sequence; baking the mean")
    return float(scales.mean())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def convert_sequence(
    seq_key: str,
    entry: dict,
    omomo_root: Path,
    out_root: Path,
    threshold: float,
    force: bool,
) -> Path | None:
    if any(key.startswith("obj_bottom") for key in entry):
        print(f"[OMOMO] SKIP {seq_key}: two-part object (independent top/bottom motion) is unsupported")
        return None

    object_name = seq_key.split("_")[1]
    seq_dir = out_root / "train_and_test" / seq_key
    seq_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OMOMO] convert {seq_key} (object={object_name})")

    frames = write_human_motion(entry, seq_dir)
    scale = write_object_prop(entry, object_name, seq_dir, frames)
    visual, collisions = prepare_object_assets(
        object_name, omomo_root, out_root / "object_mjcf" / "assets", threshold, force
    )
    write_object_xml(object_name, seq_dir, out_root / "object_mjcf" / "assets", visual, collisions, scale)
    print(f"[OMOMO]   frames={frames} scale={scale:.6f} -> {seq_dir}")
    return seq_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--omomo-root", type=Path, required=True, help="OMOMO data/ directory.")
    parser.add_argument("--out-root", type=Path, default=ROOT / "sample_data" / "omomo")
    parser.add_argument("--seq-key", action="append", default=[], help="Sequence name; repeatable.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="Convert at most N sequences (0 = no limit).")
    parser.add_argument("--coacd-threshold", type=float, default=0.05)
    parser.add_argument("--force", action="store_true", help="Rebuild shared object assets.")
    args = parser.parse_args()

    omomo_root = args.omomo_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    splits = ["train", "test"] if args.split == "both" else [args.split]

    index = load_sequence_index(omomo_root, splits)
    if args.seq_key:
        missing = [key for key in args.seq_key if key not in index]
        if missing:
            print(f"[OMOMO] unknown sequence(s): {', '.join(missing)}", file=sys.stderr)
            raise SystemExit(1)
        keys = list(args.seq_key)
    else:
        keys = sorted(index)
    if args.limit:
        keys = keys[: args.limit]

    written = [
        path
        for key in keys
        if (path := convert_sequence(key, index[key], omomo_root, out_root, args.coacd_threshold, args.force))
    ]
    print(f"\n[OMOMO] converted {len(written)}/{len(keys)} sequence(s) into {out_root}")


if __name__ == "__main__":
    main()
