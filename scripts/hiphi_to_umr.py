#!/usr/bin/env python3
"""Fit HiPHI BVH to SMPL-X and reconstruct a separately skinned HiPHI rest mesh.

All saved geometry is metres, Y-up. Object transforms retain the recorded world
frame. The rest-mesh route is a reconstruction, not a released HiPHI actor scan.
Run with the UMR conda environment; see sample_data/hiphi/README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rodrigues, batch_rigid_transform

from nr_source import _parse_bvh
from smplx_model_loader import build_smplx_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "HiPHI"
BODY = ["Hips", "LeftUpLeg", "RightUpLeg", "Spine2", "LeftLeg", "RightLeg",
        "Spine3", "LeftFoot", "RightFoot", "Spine4", "LeftToeBase", "RightToeBase",
        "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
        "LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "Head", "Head", "Head"]
MAP = BODY + [f"{s}Hand{f}{k}" for s in ("Left", "Right")
              for f in ("Index", "Middle", "Pinky", "Ring", "Thumb") for k in (1, 2, 3)]


def read_source(seq: Path, stride: int):
    nodes, frames, dt = _parse_bvh(seq / "motion_actor.bvh")
    ids = np.arange(0, len(frames), stride)
    frames = frames[ids]
    n, j = len(ids), len(nodes)
    rotations = np.zeros((n, j, 3, 3))
    positions = np.zeros((n, j, 3))
    rest = np.zeros((j, 3))
    # Median position channels describe the capture skeleton more accurately
    # than BVH OFFSET for this release; retain both in the provenance archive.
    for k, node in enumerate(nodes):
        ch = node["channels"]
        values = frames[:, node["channel_start"]:node["channel_start"] + len(ch)]
        t = np.broadcast_to(node["offset"] * .01, (n, 3)).copy()
        if "Xposition" in ch:
            t = values[:, [ch.index(a + "position") for a in "XYZ"]] * .01
        rch = [a for a in ch if a.endswith("rotation")]
        r = (Rotation.from_euler("".join(a[0] for a in rch),
                                values[:, [ch.index(a) for a in rch]], degrees=True).as_matrix()
             if rch else np.broadcast_to(np.eye(3), (n, 3, 3)))
        p = node["parent"]
        rotations[:, k] = r if p < 0 else rotations[:, p] @ r
        positions[:, k] = t if p < 0 else positions[:, p] + np.einsum("tij,tj->ti", rotations[:, p], t)
        rest[k] = 0 if p < 0 else rest[p] + np.median(t, axis=0)
    return nodes, positions, rotations, rest, ids, 1. / dt / stride


def shortest_rotation(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cross = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -.999999:
        axis = np.cross(a, [1, 0, 0] if abs(a[0]) < .9 else [0, 1, 0])
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def fit_shape(model, src_rest, source_index, height):
    dev = model.v_template.device
    beta = torch.zeros(10, device=dev, requires_grad=True)
    j0 = model.J_regressor @ model.v_template
    jd = torch.einsum("jv,vck->jck", model.J_regressor, model.shapedirs)
    pairs = [(1, 4), (2, 5), (4, 7), (5, 8), (16, 18), (17, 19), (18, 20), (19, 21),
             (16, 17), (1, 2), (7, 10), (8, 11), (1, 16), (2, 17)]
    a, b = np.array(pairs).T
    h = src_rest[source_index]
    lengths = torch.tensor(np.linalg.norm(h[a] - h[b], axis=1), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam([beta], lr=.045)
    for step in range(300):
        j = j0 + torch.einsum("jck,k->jc", jd, beta)
        v = model.v_template + torch.einsum("vck,k->vc", model.shapedirs, beta)
        loss = ((torch.linalg.vector_norm(j[a] - j[b], dim=-1) - lengths)**2).mean()
        loss = loss + .3 * ((v[:, 1].max() - v[:, 1].min()) - height)**2 + .00008 * beta.square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            beta.clamp_(-3, 3)
    print("Shape", beta.detach().cpu().numpy().round(3), "loss", float(loss), flush=True)
    return beta.detach(), (j0 + torch.einsum("jck,k->jc", jd, beta)).detach()


def target_tracks(pos, rot, rest, idx, joints, parents):
    h = rest[idx].copy()
    p = pos[:, idx].copy()
    g = rot[:, idx].copy()
    # Hips in BVH is at the hip sockets, while the SMPL pelvis is above them.
    offset = joints[0] - .5 * (joints[1] + joints[2])
    h[0] = .5 * (h[1] + h[2]) + offset
    p[:, 0] = .5 * (p[:, 1] + p[:, 2]) + np.einsum("tij,j->ti", g[:, 0], offset)
    # The five HiPHI spine links collapse onto three SMPL-X links. Locate the
    # two intermediate joints along the recorded chain, using template height.
    for k in (3, 6):
        alpha = np.clip((joints[k, 1] - joints[0, 1]) / (joints[9, 1] - joints[0, 1]), 0, 1)
        h[k] = h[0] * (1-alpha) + h[9] * alpha
        p[:, k] = p[:, 0] * (1-alpha) + p[:, 9] * alpha
    # BVH clavicle origins sit at shoulder height; SMPL collar joints sit below
    # the shoulder sockets. Preserve this anatomical offset to avoid shrugged
    # shoulders in the reconstructed skin. The head marker likewise needs a
    # neck-to-head calibration rather than treating unlike joint centers as equal.
    for k, shoulder in ((13,16),(14,17)):
        offset = joints[k] - joints[shoulder]
        h[k] = h[shoulder] + offset
        p[:,k] = p[:,shoulder] + np.einsum("tij,j->ti",g[:,k],offset)
    offset = np.array([0., max(0., (joints[15,1]-joints[12,1])-(h[15,1]-h[12,1])), 0.])
    h[15] += offset
    p[:,15] += np.einsum("tij,j->ti",g[:,15],offset)
    # Face articulation is not recorded; rigidly follow the captured head.
    for k in (22, 23, 24):
        offset = joints[k] - joints[15]
        h[k] = h[15] + offset
        p[:, k] = p[:, 15] + np.einsum("tij,j->ti", g[:, 15], offset)
    # Calibrate the model's slightly bent canonical limbs to the BVH T-pose.
    correction = np.broadcast_to(np.eye(3), (55, 3, 3)).copy()
    for k in range(55):
        children = np.flatnonzero(parents == k)
        if k in (0, 9, 12, 15, 20, 21) or not len(children):
            continue
        c = children[0]
        correction[k] = shortest_rotation(joints[c] - joints[k], h[c] - h[k])
    # Wrist calibration uses all finger bases; no arbitrary twist around palm.
    for k in (20, 21):
        children = np.flatnonzero(parents == k)
        a, b = joints[children] - joints[k], h[children] - h[k]
        correction[k] = Rotation.align_vectors(b, a)[0].as_matrix()
    return p, g, h, correction


def fit_motion(joints, parents, target, global_rot, correction, device, steps, model, beta):
    t = len(target)
    init_global = global_rot @ correction[None]
    local = init_global.copy()
    for k in range(1, 55):
        local[:, k] = init_global[:, parents[k]].transpose(0, 2, 1) @ init_global[:, k]
    init = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(t, 55, 3)
    init[:, 22:25] = 0
    aa = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)
    trans = torch.tensor(target[:, 0] - joints[0], dtype=torch.float32, device=device, requires_grad=True)
    targets = torch.tensor(target, dtype=torch.float32, device=device)
    jt = torch.tensor(joints, dtype=torch.float32, device=device).expand(t, -1, -1)
    pt = torch.tensor(parents, dtype=torch.long, device=device)
    init_m = torch.tensor(local, dtype=torch.float32, device=device)
    weights = torch.ones(55, device=device)
    weights[[0, 3, 6, 9, 12, 13, 14, 15]] = .15
    weights[22:25] = 0
    weights[25:] = .3
    weights[[7, 8, 10, 11, 20, 21]] = 2
    rest_vertices = model.v_template + torch.einsum("vck,k->vc",model.shapedirs,beta)
    soles = torch.nonzero(rest_vertices[:,1] < rest_vertices[:,1].min()+.045).flatten()[::3]
    sole_v = rest_vertices[soles].detach()
    sole_weights = model.lbs_weights[soles].detach()
    sole_pd = model.posedirs.reshape(486,-1,3)[:,soles].detach()
    opt = torch.optim.Adam([aa, trans], lr=.025)
    for step in range(steps):
        matrices = batch_rodrigues(aa.reshape(-1, 3)).reshape(t, 55, 3, 3)
        posed, skin_transforms = batch_rigid_transform(matrices, jt, pt)
        pred = posed + trans[:, None]
        position_loss = ((pred - targets).square().sum(-1) * weights).mean()
        prior = (matrices - init_m).square().mean()
        # Penalize acceleration of corrections, preserving captured motion.
        delta = matrices - init_m
        smooth = (delta[2:] - 2*delta[1:-1] + delta[:-2]).square().mean() if t > 2 else prior * 0
        pose_feature=(matrices[:,1:]-torch.eye(3,device=device)).reshape(t,-1)
        sole_posed=sole_v[None]+torch.einsum("tf,fvc->tvc",pose_feature,sole_pd)
        sole_transforms=torch.einsum("vj,tjkl->tvkl",sole_weights,skin_transforms)
        sole_world=torch.einsum("tvij,tvj->tvi",sole_transforms[:,:,:3,:3],sole_posed)+sole_transforms[:,:,:3,3]+trans[:,None]
        floor_loss=torch.relu(-sole_world[:,:,1]).square().mean()
        loss = position_loss + .0003 * prior + .005 * smooth + 2.0 * floor_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step == steps * 3 // 4:
            for group in opt.param_groups:
                group["lr"] = .008
        if step % 100 == 0 or step == steps-1:
            error = torch.linalg.vector_norm(pred[:, :22] - targets[:, :22], dim=-1).mean() * 1000
            print(f"IK {step}/{steps}: body error {float(error):.2f} mm; objective {float(loss):.6f}", flush=True)
    return aa.detach().cpu().numpy().reshape(t, 165), trans.detach().cpu().numpy()


def write_obj(path, v, f):
    trimesh.Trimesh(v, f, process=False).export(path)


def write_layout(path, poses, trans, betas, gender, fps, ids, metadata, rows, mesh):
    path.mkdir(parents=True, exist_ok=True)
    for name, a in dict(poses=poses, transl=trans, betas=betas, gender=gender, model_type="smplx",
                        mocap_framerate=fps, output_up="y", source_frame_ids=ids).items():
        np.save(path / f"{name}.npy", a)
    obj = metadata["objects"][0]["mesh_id"]
    np.savetxt(path / f"prop_{obj}.csv", rows[:, 2:9], delimiter=",", comments="", header="px,py,pz,qx,qy,qz,qw")
    # UMR conjugates object rotations to Z-up; local vertices must match.
    vv = np.asarray(mesh.vertices)[:, [0, 2, 1]].copy(); vv[:, 1] *= -1
    write_obj(path / f"{obj}.obj", vv, mesh.faces)
    (path / f"{obj}.xml").write_text(
        f'<mujoco model="{obj}"><asset><mesh name="object_mesh" file="{obj}.obj"/></asset>'
        f'<worldbody><body name="{obj}"><freejoint/><geom type="mesh" mesh="object_mesh" '
        'rgba="0.8 0.45 0.2 1"/></body></worldbody></mujoco>\n')


def convert(args):
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = args.sequence.resolve()
    md = json.loads((seq / "metadata.json").read_text())
    out = args.out.resolve() / md["motion_id"]
    out.mkdir(parents=True, exist_ok=True)
    nodes, pos, rot, rest, ids, fps = read_source(seq, args.stride)
    idx = np.array([{n["name"]: i for i, n in enumerate(nodes)}[name] for name in MAP])
    gender = md["actor_metadata"]["gender"].lower()
    model = build_smplx_model(args.model_dir, gender, 1).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    beta, jt = fit_shape(model, rest, idx, md["actor_metadata"]["height_cm"] / 100)
    joints = jt.cpu().numpy()
    parents = model.parents.cpu().numpy()
    target, global_rot, native_joints, correction = target_tracks(pos, rot, rest, idx, joints, parents)
    poses, trans = fit_motion(joints, parents, target, global_rot, correction, device, args.steps, model, beta)
    betas = beta.cpu().numpy()
    faces = np.asarray(model.faces, np.int32)
    with torch.no_grad():
        rest_v = (model.v_template + torch.einsum("vck,k->vc", model.shapedirs, beta)).cpu().numpy()
    skin = model.lbs_weights.cpu().numpy()
    # Reconstruct a bind mesh around the actor's actual rest skeleton. Its
    # skin weights/topology come from SMPL-X, not an invented surface shell.
    native_v = np.zeros_like(rest_v)
    for k in range(55):
        native_v += skin[:, k, None] * ((rest_v-joints[k]) @ correction[k].T + native_joints[k])
    # Smooth the deformation field, not the mesh, retaining face/finger detail
    # while removing sharp transitions between neighboring skinning regions.
    from scipy.sparse import coo_matrix, diags
    edges=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    edges=np.vstack([edges,edges[:,::-1]])
    adjacency=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(rest_v),len(rest_v))).tocsr()
    average=diags(1/np.maximum(np.asarray(adjacency.sum(1)).ravel(),1))@adjacency
    displacement=native_v-rest_v
    for _ in range(30): displacement=.5*displacement+.5*(average@displacement)
    native_v=(rest_v+displacement).astype(np.float32)
    rows = np.loadtxt(seq / md["objects"][0]["trajectory_path"], delimiter=",", skiprows=1)[ids]
    mesh = trimesh.load(DATA / md["objects"][0]["mesh_path"], force="mesh", process=False)
    mesh.vertices *= .01
    t, nv = len(ids), len(rest_v)
    paths = {route: out / route for route in ("smplx", "rest_skin")}
    for route, path in paths.items():
        write_layout(path, poses, trans, betas if route == "smplx" else np.zeros(10), gender, fps, ids, md, rows, mesh)
        np.save(path / "faces.npy", faces)
        np.save(path / "rest_joints.npy", joints if route == "smplx" else native_joints)
        (path / "body").mkdir(exist_ok=True)
        write_obj(path / "body/rest_mesh.obj", rest_v if route == "smplx" else native_v, faces)
    np.savez_compressed(paths["smplx"] / "motion.npz", poses=poses, trans=trans,
                        betas=betas, gender=gender, mocap_framerate=fps, output_up="y",
                        source_frame_ids=ids)
    mem = {r: np.lib.format.open_memmap(p / "vertices.npy", mode="w+", dtype="float32", shape=(t,nv,3)) for r,p in paths.items()}
    jmem = {r: np.lib.format.open_memmap(p / "joints.npy", mode="w+", dtype="float32", shape=(t,55,3)) for r,p in paths.items()}
    # Evaluate SMPL-X with pose correctives and explicit non-PCA finger poses.
    for start in range(0, t, 64):
        end = min(start+64,t); n = end-start
        bm = build_smplx_model(args.model_dir, gender, n).to(device)
        with torch.no_grad():
            aa = torch.tensor(poses[start:end], device=device)
            r = bm(betas=beta[None].expand(n,-1), transl=torch.tensor(trans[start:end],device=device),
                   global_orient=aa[:,:3], body_pose=aa[:,3:66], jaw_pose=aa[:,66:69],
                   leye_pose=aa[:,69:72], reye_pose=aa[:,72:75],
                   left_hand_pose=aa[:,75:120], right_hand_pose=aa[:,120:165])
            mem["smplx"][start:end] = r.vertices.cpu().numpy()
            jmem["smplx"][start:end] = r.joints[:,:55].cpu().numpy()
        del bm
        # Exact BVH world transforms, including non-root translation channels.
        result = np.zeros((n,nv,3),np.float32)
        for k in range(55):
            active = skin[:,k] > 1e-7
            local = native_v[active] - native_joints[k]
            pv = np.einsum("tij,vj->tvi", global_rot[start:end,k], local) + target[start:end,k,None]
            result[:,active] += pv * skin[active,k,None]
        mem["rest_skin"][start:end] = result
        jmem["rest_skin"][start:end] = target[start:end]
        if start % 512 == 0:
            print(f"Mesh evaluation {start}/{t}", flush=True)
    for a in [*mem.values(),*jmem.values()]: a.flush()
    np.savez(out / "source_skeleton.npz", positions=pos.astype("float32"), rotations=rot.astype("float32"),
             rest_joints=rest, offsets_cm=np.array([n["offset"] for n in nodes]),
             parents=np.array([n["parent"] for n in nodes]), names=np.array([n["name"] for n in nodes]),
             source_frame_ids=ids, fps=fps)
    np.save(paths["rest_skin"] / "bone_world_rotations.npy", global_rot.astype(np.float32))
    np.savez(paths["rest_skin"] / "skin.npz", weights=skin, rest_vertices=native_v, rest_joints=native_joints,
             faces=faces, source_names=np.array(MAP), parents=parents)
    manifest = {"format":"umr_vertex_cache_v1", "vertices":"vertices.npy", "joints":"joints.npy", "faces":"faces.npy",
                "coordinate_system":"y_up_metres", "topology":"smplx", "provenance":"HIPHI BVH + reconstructed SMPL-X skin"}
    (paths["rest_skin"] / "mesh_motion.json").write_text(json.dumps(manifest,indent=2))
    # Existing UMR overlay support makes the native rest mesh available to
    # correspondence learning, with no change to its vertex topology.
    overlay = paths["rest_skin"] / "model"; overlay.mkdir(exist_ok=True)
    np.savez(overlay / "parameters.npz", scale=1.)
    (overlay / "umr_smplx_overlay.json").write_text(json.dumps({
        "format":"umr_smplx_overlay_v1", "variant":f"hiphi_{md['actor_id']}_rest_skin", "gender":gender,
        "base_model_dir":str(args.model_dir.resolve()), "template_obj":"../body/rest_mesh.obj",
        "parameters_npz":"parameters.npz"},indent=2))
    report = {"motion_id":md["motion_id"],"actor":md["actor_metadata"],"source_frames":int(md["frame_count"]),
              "output_frames":t,"fps":fps,"duration_seconds":t/fps,"stride":args.stride,
              "units":"metres", "up_axis":"y", "object_track":"original; no per-frame snapping or recentering",
              "source_directory":str(seq), "objects":md["objects"],
              "rest_mesh_provenance":"reconstructed from SMPL-X template; original HiPHI body mesh is not released",
              "joint_map":{str(k):name for k,name in enumerate(MAP)},"betas":betas.tolist(),"routes":{}}
    major = [1,2,4,5,7,8,10,11,16,17,18,19,20,21]
    for route in paths:
        err = np.linalg.norm(jmem[route][:,major]-target[:,major],axis=-1)*1000
        report["routes"][route] = {"major_joint_error_mean_mm":float(err.mean()),"major_joint_error_p95_mm":float(np.percentile(err,95)),
            "minimum_vertex_y_m":float(mem[route][:,:,1].min()),"all_vertices_finite":bool(np.isfinite(mem[route]).all())}
        config = json.loads((ROOT / "humanoid_retarget_defaults_hsi_hoi_standard.json").read_text())
        config["motion"]["data"]=str(paths[route]); config["motion"]["seq_key"]=md["motion_id"]+"_"+route
        config["smpl_template"]["name"]="hiphi_"+md["actor_id"]+"_"+route
        config["retarget"]["zero_source_finger_pose"]=False
        config["view"]["enabled"]=False
        config["solver"]["retarget_object_size"]="original"
        config["solver"]["object_contact_map_cost"]=10.
        config["smplx_model_dir"]=str(args.model_dir.resolve() if route=="smplx" else overlay)
        config["retarget"]["out"]=str(paths[route] / "g1_retarget.npz")
        (paths[route] / "umr_config.json").write_text(json.dumps(config,indent=2))
    (out / "metrics.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report["routes"],indent=2), flush=True)
    print("Wrote",out,flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence",type=Path,default=DATA / "extracted/HiPHI/data/Placing/put/Placing-put_0030")
    ap.add_argument("--model-dir",type=Path,default=ROOT / "smpl")
    ap.add_argument("--out",type=Path,default=ROOT / "sample_data/hiphi")
    ap.add_argument("--stride",type=int,default=3)
    ap.add_argument("--steps",type=int,default=500)
    args=ap.parse_args()
    if args.stride < 1: ap.error("stride must be positive")
    convert(args)
