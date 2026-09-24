#!/usr/bin/env python3
"""Offline geometry and UMR loader checks; never train or retarget a robot."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from hiphi_skinning import evaluate_manifest, load_template, sha256, skin_vertices
from hiphi_reusable_skin import ROOT, DATA, MAP, source, export_objects, make_config
from humanoid_retarget_pipeline_hsi_hoi import (load_hsi_hoi_config, configure_single_smpl_template,
    export_standard_motion_npz, discover_samp_objects, load_samp_motion)
from smpl_surface_retarget_common import (load_flat_smplx_sequence, load_smplx_npz_motion,
    source_motion_vertices_joints, source_template_vertices_joints_faces)


def must_fail(fn, text):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(text)


def validate(root):
    torch.set_num_threads(4)
    reports, cache_keys, checked_templates = [], {}, set()
    for seq in sorted((root / 'sequences').iterdir()):
        if not (seq / 'mesh_motion.json').exists():
            continue
        manifest = json.loads((seq / 'mesh_motion.json').read_text())
        conversion = json.loads((seq / 'conversion.json').read_text())
        template = (seq / manifest['skin']).resolve().parent
        meta = json.loads((template / 'template.json').read_text())
        asset = load_template(str(template / 'skin.npz'), manifest['skin_sha256'])
        assert (seq / 'body/rest_mesh.obj').resolve() == template / 'body/rest_mesh.obj'
        assert sha256(template / 'body/rest_mesh.obj') == meta['rest_obj_sha256']
        motion = load_flat_smplx_sequence(seq)
        config = load_hsi_hoi_config(ROOT / 'robot_configs/humanoid_retarget_unitree_g1_example.json', seq / 'umr_config.json')
        configure_single_smpl_template(config, seq.name, seq)
        key = (config['correspondence']['dataset']['out'], config['correspondence']['train']['out_dir'])
        if manifest['template_id'] in cache_keys:
            assert cache_keys[manifest['template_id']] == key, 'Same template got different correspondence paths'
        cache_keys[manifest['template_id']] = key
        changed = copy.deepcopy(config)
        changed['correspondence']['train']['epochs'] += 1
        configure_single_smpl_template(changed, seq.name, seq)
        assert changed['correspondence']['train']['out_dir'] != key[1], 'Training changes must invalidate cache'
        bad = copy.deepcopy(config)
        bad['correspondence']['shared_template_key'] = 'wrong_skin'
        must_fail(lambda: configure_single_smpl_template(bad, seq.name, seq), 'Mismatched template accepted')
        if manifest['template_id'] not in checked_templates:
            rest = skin_vertices(asset, np.eye(3)[None, None].repeat(55, 1), asset['rest_joints'][None])[0]
            np.testing.assert_allclose(rest, asset['rest_vertices'], atol=5e-7)
            r = Rotation.from_euler('xyz', [.3, -.6, .2]).as_matrix()
            t = np.array([2., 3., -4.])
            posed = skin_vertices(asset, np.tile(r, (1, 55, 1, 1)), (asset['rest_joints'] @ r.T + t)[None])[0]
            np.testing.assert_allclose(posed, asset['rest_vertices'] @ r.T + t, atol=2e-6)
            # The actual correspondence-template loader must preserve vertex indexing.
            v, j, f, _, _ = source_template_vertices_joints_faces(motion, config['smpl_template'], template / 'model')
            np.testing.assert_allclose(v, asset['rest_vertices'], atol=1e-6)
            np.testing.assert_array_equal(f, asset['faces'])
            checked_templates.add(manifest['template_id'])
        n = len(motion['pose_aa'])
        minimum_y, maximum_step, previous = float('inf'), 0., None
        for start in range(0, n, 64):
            ids = np.arange(start, min(n, start + 64))
            v, j, f = source_motion_vertices_joints(motion, ids, template / 'model')
            assert v.shape == (len(ids), 10475, 3) and j.shape == (len(ids), 55, 3)
            assert np.isfinite(v).all() and np.isfinite(j).all()
            minimum_y = min(minimum_y, float(v[:, :, 1].min()))
            if previous is not None:
                maximum_step = max(maximum_step, float(np.linalg.norm(v[0] - previous, axis=-1).max()))
            maximum_step = max(maximum_step, float(np.linalg.norm(np.diff(v, axis=0), axis=-1).max(initial=0)))
            previous = v[-1]
        ids = np.array([n - 1, 0, n // 2, 0])
        v, j, f = evaluate_manifest(seq / 'mesh_motion.json', ids)
        np.testing.assert_array_equal(v[1], v[3])
        must_fail(lambda: evaluate_manifest(seq / 'mesh_motion.json', np.array([-1])), 'Negative frame accepted')
        must_fail(lambda: evaluate_manifest(seq / 'mesh_motion.json', np.array([n])), 'Past-end frame accepted')
        with tempfile.TemporaryDirectory(prefix='hiphi-loader-') as tmp:
            exported = export_standard_motion_npz(seq.name, seq, Path(tmp), False)
            exported_motion = load_smplx_npz_motion(exported)
            vv, jj, ff = source_motion_vertices_joints(exported_motion, ids, template / 'model')
            np.testing.assert_array_equal(vv, v)
            np.testing.assert_array_equal(jj, j)
        nodes, positions, _, _, source_ids, _ = source(Path(conversion['source_directory']), conversion['stride'])
        lookup = {node['name']: k for k, node in enumerate(nodes)}
        major = [1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21]
        expected = positions[:, [lookup[MAP[k]] for k in major]]
        actual = np.load(seq / 'joints.npy', mmap_mode='r')[:, major]
        error = float(np.linalg.norm(expected - actual, axis=-1).max())
        assert error < 1e-6, 'Recorded limbs or hands changed during binding'
        np.testing.assert_array_equal(source_ids, np.load(seq / 'source_frame_ids.npy'))
        assert len(discover_samp_objects(seq)) == len(manifest['objects'])
        assert not (seq / 'g1_retarget.npz').exists(), 'Unexpected retarget output in this new dataset'
        row = dict(sequence=seq.name, actor=conversion['actor_id'], frames=n, finite=True,
                   mirrored=bool(conversion.get('mirrored', False)),
                   template_id=manifest['template_id'], min_vertex_y_m=minimum_y,
                   max_vertex_step_m=maximum_step, recorded_major_joint_max_error_m=error,
                   objects=len(manifest['objects']), umr_flat_and_npz_loading=True)
        reports.append(row)
        print(json.dumps(row), flush=True)
    # Controlled fixtures cover cardinality without pretending they are real clips.
    ref = next((root / 'sequences').glob('*/conversion.json'))
    conversion = json.loads(ref.read_text())
    seq = Path(conversion['source_directory'])
    md = json.loads((seq / 'metadata.json').read_text())
    for count in (0, 2):
        fixture = copy.deepcopy(md)
        fixture['objects'] = [] if count == 0 else [dict(md['objects'][0], object_id=f'test_object_{i}') for i in range(count)]
        with tempfile.TemporaryDirectory(prefix='hiphi-object-fixture-') as tmp:
            directory = Path(tmp)
            objects = export_objects(seq, fixture, DATA, directory, np.array([0, 3, 6]), md['frame_count'])
            assert len(discover_samp_objects(directory)) == count
            cfg = make_config(directory, template, meta, fixture, objects)
            assert cfg['solver']['robot_object_hard_constraint'] == bool(count)
            assert cfg['solver']['object_contact_map_cost'] == (10. if count else 0.)
            if count:
                assert cfg['hsi_hoi']['object']['require_explicit_selection']
                assert 'name' not in cfg['hsi_hoi']['object']
                assert objects[0]['exported_name'] != objects[1]['exported_name']
                original = trimesh.load(DATA / md['objects'][0]['mesh_path'], force='mesh', process=False)
                exported = trimesh.load(directory / objects[0]['mesh'], force='mesh', process=False)
                expected = np.asarray(original.vertices)[:, [0, 2, 1]] * .01
                expected[:, 1] *= -1
                np.testing.assert_allclose(exported.vertices, expected, atol=1e-8)
                from retarget_smpl_to_humanoid_surface_vector import load_object_contact_source
                args = SimpleNamespace(config_data=cfg, data=str(directory), object_contact_map_cost=10.,
                    robot_object_hard_constraint=True, robot_object_penetration_soft_cost=0.,
                    retarget_object_size='original', object_contact_map_samples=32, seed=0,
                    object_contact_map_snap_threshold=0., object_contact_map_threshold=.1,
                    object_contact_map_max_points=32)
                slots = np.zeros((1, 2, 3), np.float32)
                must_fail(lambda: load_object_contact_source(args, np.array([0]), slots, 1., 0.),
                          'Multiple objects silently selected the first instance')
                cfg['hsi_hoi']['object']['name'] = 'missing'
                must_fail(lambda: load_object_contact_source(args, np.array([0]), slots, 1., 0.),
                          'Missing selected object silently fell back')
                cfg['hsi_hoi']['object']['name'] = objects[1]['exported_name']
                selected = load_object_contact_source(args, np.array([0]), slots, 1., 0.)
                assert selected['name'] == objects[1]['exported_name']
    legacy = ROOT / 'sample_data/hiphi/Placing-put_0030/rest_skin'
    with tempfile.TemporaryDirectory(prefix='hiphi-manifest-fixture-') as tmp:
        missing = Path(tmp)
        np.save(missing / 'mesh_motion_required.npy', True)
        must_fail(lambda: load_flat_smplx_sequence(missing), 'Missing native manifest fell back to SMPL-X')
        must_fail(lambda: load_samp_motion(missing), 'Missing native manifest fell back to SMPL-X')
    if legacy.exists():
        old_motion = load_flat_smplx_sequence(legacy)
        actual = source_motion_vertices_joints(old_motion, np.array([0, 10]), legacy / 'model')[0]
        np.testing.assert_array_equal(actual, np.load(legacy / 'vertices.npy', mmap_mode='r')[[0, 10]])
    result = dict(sequences=reports, total_frames=sum(r['frames'] for r in reports),
                  templates=len(checked_templates), correspondence_cache_paths={k: list(v) for k, v in cache_keys.items()},
                  synthetic_body_only_and_two_object_export=True, legacy_v1_compatible=legacy.exists(),
                  all_checks_passed=True, umr_training_run=False, umr_retargeting_run=False,
                  limitation='Geometry checks are not a surface-accuracy or collision-free guarantee. Body-only and multiple-object coverage uses controlled fixtures.')
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'PASS: {len(reports)} clips, {result["total_frames"]} frames, {len(checked_templates)} templates', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'sample_data/hiphi_reusable')
    validate(parser.parse_args().root.resolve())
