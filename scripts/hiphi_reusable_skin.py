#!/usr/bin/env python3
"""Build a HiPHI skin once, then bind compatible sequences. No UMR jobs launched.

Default: one dataset template, calibrated from Placing-put_0030. Optional
--template-mode actor fits a template once per actor.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix, diags
from hiphi_to_umr import MAP, ROOT, DATA, read_source, fit_shape, target_tracks, write_obj
from hiphi_skinning import sha256, load_template
from nr_source import _parse_bvh_cached
from smplx_model_loader import build_smplx_model

VERSION = 1


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def safe_id(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Unsafe or unsupported identifier: {value!r}')
    return value


def schema(nodes):
    return [[n['name'], nodes[n['parent']]['name'] if n['parent'] >= 0 else None,
             list(n['channels'])] for n in nodes]


def source(seq, stride):
    try:
        result = read_source(seq, stride)
        md = json.loads((seq / 'metadata.json').read_text())
        _, frames, dt = _parse_bvh_cached(str((seq / 'motion_actor.bvh').resolve()))
        if len(frames) != int(md['frame_count']) or not np.isclose(1 / dt, md['fps'], rtol=1e-4):
            raise ValueError('BVH frame count or rate differs from metadata')
        return result
    finally:
        _parse_bvh_cached.cache_clear()


def make_template(reference, destination, model_dir):
    """Never replace a template implicitly; recalibration needs a new output root."""
    manifest_path = destination / 'template.json'
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if meta['builder_version'] != VERSION:
            raise ValueError('Template version changed; choose a new --out directory')
        asset = load_template(str((destination / 'skin.npz').resolve()), meta['skin_sha256'])
        if sha256(destination / 'body/rest_mesh.obj') != meta['rest_obj_sha256']:
            raise ValueError('Shared rest OBJ changed; choose a new output directory')
        return meta, asset
    md = json.loads((reference / 'metadata.json').read_text())
    nodes, pos, rot, rest, ids, fps = source(reference, 30)
    lookup = {n['name']: i for i, n in enumerate(nodes)}
    idx = np.array([lookup[name] for name in MAP])
    gender = str(md['actor_metadata']['gender']).lower()
    model = build_smplx_model(model_dir, gender, 1).to('cpu')
    for p in model.parameters():
        p.requires_grad_(False)
    beta, jt = fit_shape(model, rest, idx, md['actor_metadata']['height_cm'] / 100)
    joints, parents = jt.numpy(), model.parents.numpy()
    _, _, bind, correction = target_tracks(pos[:1], rot[:1], rest, idx, joints, parents)
    with torch.no_grad():
        smpl_vertices = (model.v_template + torch.einsum('vck,k->vc', model.shapedirs, beta)).numpy()
    weights, faces = model.lbs_weights.numpy(), np.asarray(model.faces, np.int32)
    vertices = np.zeros_like(smpl_vertices)
    for k in range(55):
        vertices += weights[:, k, None] * ((smpl_vertices - joints[k]) @ correction[k].T + bind[k])
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))).tocsr()
    average = diags(1 / np.maximum(np.asarray(adjacency.sum(1)).ravel(), 1)) @ adjacency
    displacement = vertices - smpl_vertices
    for _ in range(30):
        displacement = .5 * displacement + .5 * (average @ displacement)
    vertices = (smpl_vertices + displacement).astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.template-', dir=destination.parent))
    try:
        (stage / 'body').mkdir()
        (stage / 'model').mkdir()
        write_obj(stage / 'body/rest_mesh.obj', vertices, faces)
        np.savez_compressed(stage / 'skin.npz', rest_vertices=vertices, rest_joints=bind.astype(np.float32),
                            weights=weights, faces=faces, source_rest=rest,
                            source_names=np.array([n['name'] for n in nodes]), model_joints=joints,
                            parents=parents, betas=beta.numpy())
        digest = sha256(stage / 'skin.npz')
        template_id = f'hiphi_{destination.name}_{digest[:16]}'
        np.savez(stage / 'model/parameters.npz', scale=1.)
        write_json(stage / 'model/umr_smplx_overlay.json', {
            'format': 'umr_smplx_overlay_v1', 'variant': template_id, 'gender': gender,
            'base_model_dir': os.path.relpath(model_dir.resolve(), destination / 'model'),
            'template_obj': '../body/rest_mesh.obj',
            'parameters_npz': 'parameters.npz'})
        meta = {'format': 'hiphi_shared_template_v1', 'builder_version': VERSION,
                'template_id': template_id, 'skin_sha256': digest,
                'rest_obj_sha256': sha256(stage / 'body/rest_mesh.obj'),
                'reference_motion': md['motion_id'], 'reference_actor': md['actor_id'],
                'reference_metadata': md['actor_metadata'], 'gender': gender,
                'reference_bvh_sha256': sha256(reference / 'motion_actor.bvh'),
                'schema': schema(nodes), 'coordinate_system': 'y_up_metres',
                'vertices': len(vertices), 'faces': len(faces),
                'provenance': 'Reconstructed SMPL-X proxy with a mapped 55-joint HiPHI rig; not an actor scan.'}
        write_json(stage / 'template.json', meta)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f'Created shared template {template_id}', flush=True)
    return meta, load_template(str((destination / 'skin.npz').resolve()), digest)


def resolve_mesh(dataset_root, seq, item):
    value = Path(item['mesh_path'])
    for candidate in (dataset_root / value, seq / value, dataset_root / 'object_meshes' / value.name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Object mesh {value} (dataset root {dataset_root})')


def fingerprint(seq, md, dataset_root, stride, template_id):
    files = [seq / 'metadata.json', seq / 'motion_actor.bvh']
    for item in md.get('objects', []):
        files += [seq / item['trajectory_path'], resolve_mesh(dataset_root, seq, item)]
    content = [VERSION, stride, template_id, [sha256(p) for p in files]]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def export_objects(seq, md, dataset_root, stage, ids, source_count):
    exported, used = [], set()
    for item in md.get('objects', []):
        name = safe_id(item.get('object_id', item['mesh_id']))
        if name in used:
            raise ValueError(f'Duplicate object ID: {name}')
        used.add(name)
        rows = np.atleast_1d(np.genfromtxt(seq / item['trajectory_path'], delimiter=',', names=True))
        fields = ['px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw']
        if not set(fields).issubset(rows.dtype.names or ()) or len(rows) != source_count:
            raise ValueError(f'Object {name}: expected named pose columns and {source_count} rows')
        if 'frame' in rows.dtype.names and not np.array_equal(rows['frame'], np.arange(source_count)):
            raise ValueError(f'Object {name}: nonconsecutive or shifted frame IDs')
        if 'time_sec' in rows.dtype.names and not np.allclose(
                rows['time_sec'], np.arange(source_count) / float(md['fps']), atol=2e-5, rtol=0):
            raise ValueError(f'Object {name}: timestamps are not synchronized to the body')
        selected = np.column_stack([rows[f][ids] for f in fields])
        if not np.isfinite(selected).all() or not np.allclose(np.linalg.norm(selected[:, 3:], axis=1), 1, atol=.005):
            raise ValueError(f'Object {name}: nonfinite position or invalid quaternion')
        np.savetxt(stage / f'prop_{name}.csv', selected, delimiter=',', comments='', header=','.join(fields))
        mesh = trimesh.load(resolve_mesh(dataset_root, seq, item), force='mesh', process=False)
        vertices = np.asarray(mesh.vertices) * .01
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError(f'Invalid object mesh {name}')
        # Source tracks stay Y-up; UMR conjugates rotations, requiring Z-up OBJ locals.
        vertices = vertices[:, [0, 2, 1]].copy()
        vertices[:, 1] *= -1
        write_obj(stage / f'{name}.obj', vertices, mesh.faces)
        xml = ET.Element('mujoco', model=name)
        ET.SubElement(ET.SubElement(xml, 'asset'), 'mesh', name='object_mesh', file=f'{name}.obj')
        body = ET.SubElement(ET.SubElement(xml, 'worldbody'), 'body', name=name)
        ET.SubElement(body, 'freejoint')
        ET.SubElement(body, 'geom', type='mesh', mesh='object_mesh', rgba='0.8 0.45 0.2 1')
        ET.ElementTree(xml).write(stage / f'{name}.xml', encoding='unicode')
        exported.append(dict(item, exported_name=name, trajectory=f'prop_{name}.csv', mesh=f'{name}.obj'))
    return exported


def make_config(out, template, meta, md, objects):
    config = json.loads((ROOT / 'humanoid_retarget_defaults_hsi_hoi_standard.json').read_text())
    config['motion'].update(data=str(out), seq_key=md['motion_id'])
    config['smpl_template']['name'] = meta['template_id']
    config['correspondence']['shared_template_key'] = meta['template_id']
    config['smplx_model_dir'] = str(template / 'model')
    config['retarget'].update(zero_source_finger_pose=False, out=str(out / 'g1_retarget.npz'))
    config['view']['enabled'] = False
    config['solver'].update(retarget_object_size='original', object_contact_map_cost=10. if objects else 0.,
                            robot_object_hard_constraint=bool(objects), robot_object_penetration_soft_cost=0.)
    config['hsi_hoi']['object'].update(output_up='y', object_scale=1., require_explicit_selection=True)
    if len(objects) == 1:
        config['hsi_hoi']['object']['name'] = objects[0]['exported_name']
    return config


def bind_sequence(seq, out, template, meta, asset, dataset_root, stride):
    md = json.loads((seq / 'metadata.json').read_text())
    signature = fingerprint(seq, md, dataset_root, stride, meta['template_id'])
    if (out / 'conversion.json').exists():
        existing = json.loads((out / 'conversion.json').read_text())
        if existing['input_signature'] != signature:
            raise ValueError(f'Inputs or template changed: {out}; use a new --out directory')
        required = ['mesh_motion.json', 'umr_config.json', 'body/rest_mesh.obj',
                    'model/umr_smplx_overlay.json']
        required += [f'{name}.npy' for name in ('poses', 'transl', 'betas', 'gender', 'model_type',
                     'mocap_framerate', 'output_up', 'source_frame_ids', 'bone_world_rotations', 'joints')]
        for item in existing['objects']:
            required += [item['trajectory'], item['mesh'], item['exported_name'] + '.xml']
        if any(not (out / name).is_file() for name in required):
            raise ValueError(f'Incomplete export {out}; restore missing files or use a new --out directory')
        if not (out / 'mesh_motion_required.npy').exists():
            np.save(out / 'mesh_motion_required.npy', True)
        print(f'Reusing completed {md["motion_id"]}', flush=True)
        return existing
    nodes, pos, rot, rest, ids, fps = source(seq, stride)
    if schema(nodes) != meta['schema']:
        raise ValueError('BVH joint names, parents, or channel schema differ from the shared template')
    if not len(pos) or not np.isfinite(pos).all() or not np.isfinite(rot).all():
        raise ValueError('Empty or nonfinite source motion')
    if len(np.arange(0, int(md['frame_count']), stride)) != len(ids):
        raise ValueError('BVH frame count differs from metadata')
    lookup = {n['name']: i for i, n in enumerate(nodes)}
    idx = np.array([lookup[name] for name in MAP])
    # Calibration is frozen with the shared skin, never recomputed per sequence.
    target, rotations, bind, correction = target_tracks(pos, rot, asset['source_rest'], idx,
                                                        asset['model_joints'], asset['parents'])
    if not np.allclose(bind, asset['rest_joints'], atol=1e-6):
        raise ValueError('Bind calibration changed unexpectedly')
    init_global = rotations @ correction[None]
    local = init_global.copy()
    for k in range(1, 55):
        local[:, k] = init_global[:, asset['parents'][k]].transpose(0, 2, 1) @ init_global[:, k]
    poses = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(len(ids), 165).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.motion-', dir=out.parent))
    try:
        arrays = dict(poses=poses, transl=(target[:, 0] - asset['model_joints'][0]).astype(np.float32),
                      betas=np.zeros(10, np.float32), gender=meta['gender'], model_type='smplx',
                      mocap_framerate=fps, output_up='y', source_frame_ids=ids,
                      mesh_motion_required=True,
                      bone_world_rotations=rotations.astype(np.float32), joints=target.astype(np.float32))
        for name, value in arrays.items():
            np.save(stage / f'{name}.npy', value)
        objects = export_objects(seq, md, dataset_root, stage, ids, int(md['frame_count']))
        write_json(stage / 'mesh_motion.json', {
            'format': 'umr_hiphi_skin_v2', 'coordinate_system': 'y_up_metres',
            'skin': os.path.relpath(template / 'skin.npz', out), 'skin_sha256': meta['skin_sha256'],
            'template_id': meta['template_id'], 'rotations': 'bone_world_rotations.npy', 'joints': 'joints.npy',
            'frame_count': len(ids), 'topology': 'smplx', 'objects': objects,
            'pose_semantics': 'Unfitted mapped angles for metadata only; evaluate this skin manifest.'})
        (stage / 'body').symlink_to(os.path.relpath(template / 'body', out), target_is_directory=True)
        (stage / 'model').symlink_to(os.path.relpath(template / 'model', out), target_is_directory=True)
        config = make_config(out, template, meta, md, objects)
        write_json(stage / 'umr_config.json', config)
        for obj in objects if len(objects) > 1 else []:
            selected = copy.deepcopy(config)
            selected['hsi_hoi']['object']['name'] = obj['exported_name']
            selected['retarget']['out'] = str(out / f'g1_{obj["exported_name"]}_retarget.npz')
            write_json(stage / f'umr_config_{obj["exported_name"]}.json', selected)
        report = {'motion_id': md['motion_id'], 'actor_id': md['actor_id'], 'actor_metadata': md['actor_metadata'],
                  'template_id': meta['template_id'], 'template_reference_actor': meta['reference_actor'],
                  'input_signature': signature, 'source_directory': str(seq), 'stride': stride,
                  'source_frames': md['frame_count'], 'output_frames': len(ids), 'fps': fps,
                  'mirrored': bool(md.get('mirrored', False)), 'objects': objects,
                  'source_rest_difference_max_m': float(np.linalg.norm(rest - asset['source_rest'], axis=1).max()),
                  'rest_skin_shared': True, 'umr_training_run': False, 'umr_retargeting_run': False}
        write_json(stage / 'conversion.json', report)
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f'Bound {md["motion_id"]}: {len(ids)} frames, {len(objects)} objects, {meta["template_id"]}', flush=True)
    return report


def main(args):
    torch.set_num_threads(4)
    root, out = args.dataset_root.resolve(), args.out.resolve()
    inventory = []
    for path in sorted(root.rglob('metadata.json')):
        if (path.parent / 'motion_actor.bvh').is_file():
            md = json.loads(path.read_text())
            if md.get('dataset') == 'HiPHI':
                inventory.append((path.parent, md))
    if not inventory:
        raise ValueError(f'No extracted HiPHI sequences found under {root}')
    references = {md['motion_id']: path for path, md in inventory}
    selected = [(p, md) for p, md in inventory if not args.sequence or md['motion_id'] in args.sequence]
    if args.sequence and set(args.sequence) - set(references):
        raise ValueError(f'Unknown sequences: {set(args.sequence) - set(references)}')
    templates, reports, errors = {}, [], []
    for seq, md in selected:
        try:
            key = 'shared' if args.template_mode == 'shared' else safe_id(md['actor_id'])
            template = out / 'templates' / key
            if key not in templates:
                if (template / 'template.json').exists():
                    ref = None  # Later dataset shards do not need the calibration BVH.
                    saved = json.loads((template / 'template.json').read_text())
                    if key == 'shared' and saved['reference_motion'] != args.reference:
                        raise ValueError('Existing template uses a different --reference; use a new --out directory')
                else:
                    ref = references.get(args.reference) if key == 'shared' else next(
                        (p for p, m in inventory if m['actor_id'] == md['actor_id'] and not m.get('mirrored', False)), None)
                    if ref is None:
                        raise ValueError('No calibration clip available; supply --reference or build the actor template from an unmirrored clip first')
                templates[key] = make_template(ref, template, args.model_dir.resolve())
            meta, asset = templates[key]
            reports.append(bind_sequence(seq, out / 'sequences' / safe_id(md['motion_id']),
                                         template, meta, asset, root, args.stride))
        except Exception as exc:
            errors.append({'sequence': md['motion_id'], 'error': f'{type(exc).__name__}: {exc}'})
            print(f'FAILED {md["motion_id"]}: {exc}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'batch_report.json', {'template_mode': args.template_mode,
               'requested_sequences': len(selected), 'completed': reports, 'errors': errors})
    print(f'{len(reports)} completed, {len(errors)} failed. No UMR training or retargeting launched.', flush=True)
    return bool(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DATA)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'smpl')
    parser.add_argument('--out', type=Path, default=ROOT / 'sample_data/hiphi_reusable')
    parser.add_argument('--sequence', action='append', help='Motion ID; repeat to select clips. Default: all extracted clips.')
    parser.add_argument('--reference', default='Placing-put_0030')
    parser.add_argument('--template-mode', choices=['shared', 'actor'], default='shared')
    parser.add_argument('--stride', type=int, default=3)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    raise SystemExit(main(args))
