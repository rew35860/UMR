#!/usr/bin/env python3
"""Validate HiPHI-to-UMR surfaces through UMR's actual loader and HOI exporter."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import smpl_surface_retarget_common as common
from humanoid_retarget_pipeline_hsi_hoi import export_standard_motion_npz
from smplx_model_loader import build_smplx_model

ROOT=Path(__file__).resolve().parents[1]


def validate(base):
    metrics=json.loads((base/'metrics.json').read_text())
    result={}
    for route in ('smplx','rest_skin'):
        path=base/route
        expected=np.load(path/'vertices.npy',mmap_mode='r')
        frames=np.unique(np.linspace(0,len(expected)-1,6,dtype=int))
        source=common.load_flat_smplx_sequence(path)
        config=json.loads((path/'umr_config.json').read_text())
        model_dir=Path(config['smplx_model_dir'])
        verts,joints,faces=common.source_motion_vertices_joints(source,frames,model_dir,
            smplx_device='cpu',smplx_batch_size=6,zero_source_finger_pose=False)
        max_error=float(np.abs(verts-expected[frames]).max())
        assert max_error<2e-5, (route,max_error)
        with tempfile.TemporaryDirectory(prefix='hiphi_validate_') as tmp:
            exported=export_standard_motion_npz(path.name,path,Path(tmp),False)
            seq=common.load_smplx_npz_motion(exported)
            vv,jj,ff=common.source_motion_vertices_joints(seq,frames,model_dir,
                smplx_device='cpu',smplx_batch_size=6,zero_source_finger_pose=False)
            np.testing.assert_allclose(vv,verts,atol=2e-5)
        assert expected.shape==(len(source["pose_aa"]),10475,3),expected.shape
        assert len(faces)==20908 and np.isfinite(verts).all()
        rest=trimesh.load(path/'body/rest_mesh.obj',process=False,force='mesh')
        model=build_smplx_model(model_dir,'male',1)
        if route=='rest_skin':
            np.testing.assert_allclose(model.v_template.detach().numpy(),rest.vertices,atol=1e-7)
            skin=np.load(path/'skin.npz')
            np.testing.assert_allclose(skin['weights'].sum(1),1,atol=1e-6)
            rotations=np.load(path/'bone_world_rotations.npy',mmap_mode='r')
            cached_joints=np.load(path/'joints.npy',mmap_mode='r')
            k=int(frames[2])
            reconstructed=sum(skin['weights'][:,j,None]*(
                (skin['rest_vertices']-skin['rest_joints'][j])@rotations[k,j].T
                +cached_joints[k,j]) for j in range(55))
            np.testing.assert_allclose(reconstructed,expected[k],atol=3e-6)
        else:
            flat,_,_=common.source_motion_vertices_joints(source,frames,model_dir,
                smplx_device='cpu',smplx_batch_size=6,zero_source_finger_pose=True)
            finger_effect=float(np.abs(flat-verts).max())
            assert finger_effect>.01, 'Articulated fingers were silently discarded'
        edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),1),axis=0)
        baseline=np.linalg.norm(rest.vertices[edges[:,0]]-rest.vertices[edges[:,1]],axis=1)
        sampled=expected[::30]
        lengths=np.linalg.norm(sampled[:,edges[:,0]]-sampled[:,edges[:,1]],axis=-1)
        stretch=lengths[:,baseline>1e-4]/baseline[baseline>1e-4]
        obj=trimesh.load(path/'Box_A_1.obj',process=False,force='mesh')
        original=trimesh.load(ROOT.parent/'HiPHI/object_meshes/Box_A_1.obj',process=False,force='mesh')
        basis=common.Y_UP_TO_Z_UP_MATRIX
        np.testing.assert_allclose(obj.vertices,(original.vertices*.01)@basis.T,atol=1e-7)
        track=np.loadtxt(path/'prop_Box_A_1.csv',delimiter=',',skiprows=1)
        src=np.loadtxt(ROOT.parent/'HiPHI/extracted/HiPHI/data/Placing/put/Placing-put_0030/object_tracks/Box_A_1.csv',delimiter=',',skiprows=1)
        np.testing.assert_allclose(track,src[::3,2:9],atol=1e-12)
        # Physical surface diagnostics, evaluated in the recorded world frame.
        ground=expected[:,:,1].min(1)
        result[route]={'loader_max_vertex_error_m':max_error,'hoi_export_roundtrip':'passed',
            'topology':'10475 vertices, 20908 triangles', 'object_track_and_basis':'passed',
            'animation_edge_stretch_p99':float(np.percentile(stretch,99)),
            'animation_edge_stretch_p999':float(np.percentile(stretch,99.9)),
            'floor_penetration_max_mm':float(max(0,-ground.min())*1000),
            'floor_penetration_p95_mm':float(np.percentile(np.maximum(0,-ground),95)*1000),
            'rest_mesh_height_m':float(np.ptp(rest.vertices[:,1]))}
    for route, checks in result.items():
        metrics.setdefault('validation', {}).setdefault(route, {}).update(checks)
    (base/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence',type=Path,default=ROOT/'sample_data/hiphi/Placing-put_0030')
    validate(ap.parse_args().sequence)
