"""Fixed-topology surface cache; template must share vertex ordering and faces.

The accompanying SMPL-X poses provide metadata only for this route. The stored
surface may use a different rig. HiPHI supplies its reconstructed rest mesh as
an overlay for UMR correspondence learning.
"""
import json
from pathlib import Path
import numpy as np


def motion_vertices_joints(sequence, frame_ids):
    path = Path(sequence['mesh_motion_manifest'])
    manifest = json.loads(path.read_text())
    if manifest.get('format') == 'umr_hiphi_skin_v2':
        from hiphi_skinning import evaluate_manifest
        if sequence.get('output_up') != 'y':
            raise ValueError('Reusable HiPHI skin must use Y-up metres')
        selected_v, selected_j, faces = evaluate_manifest(path, frame_ids, len(sequence['pose_aa']))
        return _scale_surface(sequence, frame_ids, selected_v, selected_j, faces)
    if manifest.get('format') != 'umr_vertex_cache_v1':
        raise ValueError(f'Unsupported mesh cache: {path}')
    if manifest.get('coordinate_system') != 'y_up_metres' or sequence.get('output_up') != 'y':
        raise ValueError('Vertex cache and sequence must both use Y-up metres')
    vertices = np.load(path.parent / manifest['vertices'], mmap_mode='r', allow_pickle=False)
    joints = np.load(path.parent / manifest['joints'], mmap_mode='r', allow_pickle=False)
    faces = np.load(path.parent / manifest['faces'], allow_pickle=False)
    count = len(sequence['pose_aa'])
    if vertices.ndim != 3 or vertices.shape[0] != count or vertices.shape[2] != 3:
        raise ValueError(f'Invalid vertex cache shape {vertices.shape}, expected ({count}, V, 3)')
    if joints.shape != (count, 55, 3):
        raise ValueError(f'Expected SMPL-X joint ordering (T,55,3), got {joints.shape}')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= vertices.shape[1]:
        raise ValueError('Invalid vertex-cache triangle indices')
    selected_v = np.asarray(vertices[frame_ids], dtype=np.float32)
    selected_j = np.asarray(joints[frame_ids], dtype=np.float32)
    if not np.isfinite(selected_v).all() or not np.isfinite(selected_j).all():
        raise ValueError('Nonfinite cached surface')
    return _scale_surface(sequence, frame_ids, selected_v, selected_j, faces)


def _scale_surface(sequence, frame_ids, selected_v, selected_j, faces):
    scale = float(sequence.get('human_scale', 1.))
    mode = str(sequence.get('human_scale_mode', 'off'))
    if mode == 'world':
        selected_v *= scale
        selected_j *= scale
    elif mode == 'local':
        origin = np.asarray(sequence['trans_orig'])[frame_ids, None]
        selected_v = (selected_v-origin)*scale+origin
        selected_j = (selected_j-origin)*scale+origin
    return selected_v, selected_j, np.asarray(faces, dtype=np.int32)
