"""Evaluate a shared HiPHI bind mesh using recorded world bone transforms."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def load_template(path_str, expected_hash):
    path = Path(path_str)
    if sha256(path) != expected_hash:
        raise ValueError(f'Shared skin checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as data:
        asset = {key: data[key] for key in data.files}
    v, w, j, f = (asset[k] for k in ('rest_vertices', 'weights', 'rest_joints', 'faces'))
    if (v.ndim != 2 or v.shape[1] != 3 or j.shape != (55, 3)
            or w.shape != (len(v), 55) or not np.isfinite(v).all()
            or not np.isfinite(j).all() or not np.isfinite(w).all()
            or np.min(w) < 0 or not np.allclose(w.sum(1), 1, atol=1e-5)
            or f.ndim != 2 or f.shape[1] != 3 or not len(f)
            or f.min() < 0 or f.max() >= len(v)):
        raise ValueError(f'Invalid shared skin: {path}')
    for array in asset.values():
        array.flags.writeable = False
    return asset


def skin_vertices(asset, rotations, joints):
    rotations = np.asarray(rotations, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    if rotations.shape != (len(joints), 55, 3, 3) or joints.shape[1:] != (55, 3):
        raise ValueError('Expected world rotations (T,55,3,3) and joints (T,55,3)')
    rest, weights, bind = (asset[k] for k in ('rest_vertices', 'weights', 'rest_joints'))
    result = np.zeros((len(joints), len(rest), 3), dtype=np.float32)
    # Small batches bound intermediates; retain every nonzero weight.
    for start in range(0, len(joints), 32):
        stop = min(start + 32, len(joints))
        for k in range(55):
            active = weights[:, k] > 0
            local = rest[active] - bind[k]
            posed = np.einsum('tij,vj->tvi', rotations[start:stop, k], local)
            posed += joints[start:stop, k, None]
            result[start:stop, active] += posed * weights[active, k, None]
    return result


def evaluate_manifest(path, frame_ids, count=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('format') != 'umr_hiphi_skin_v2' or manifest.get('coordinate_system') != 'y_up_metres':
        raise ValueError(f'Invalid reusable skin manifest: {path}')
    asset = load_template(str((path.parent / manifest['skin']).resolve()), manifest['skin_sha256'])
    rotations = np.load(path.parent / manifest['rotations'], mmap_mode='r', allow_pickle=False)
    joints = np.load(path.parent / manifest['joints'], mmap_mode='r', allow_pickle=False)
    expected = int(manifest['frame_count'])
    if count is not None and count != expected:
        raise ValueError('Motion metadata and skin frame counts differ')
    if rotations.shape != (expected, 55, 3, 3) or joints.shape != (expected, 55, 3):
        raise ValueError('Invalid motion transform array shapes')
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or (len(ids) and (ids.min() < 0 or ids.max() >= expected)):
        raise ValueError('Frame IDs must be a one-dimensional array of valid integer indices')
    r, j = np.array(rotations[ids]), np.array(joints[ids])
    if not np.isfinite(r).all() or not np.isfinite(j).all():
        raise ValueError('Nonfinite HiPHI bone transforms')
    vertices = skin_vertices(asset, r, j)
    return vertices, j, np.asarray(asset['faces'], dtype=np.int32)
