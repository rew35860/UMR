# HiPHI human meshes for UMR

`Placing-put_0030` is converted by two routes. Both include the full 59.266-second
recording (5,334 source frames at 90.0009 Hz, sampled every third frame to 1,778
frames at 30.0003 Hz), articulated hands, the full-size Box_A_1, and its original
world trajectory. Geometry is in metres, Y-up. No per-frame object snapping,
body recentering, or time compression is applied.

The public [HiPHI dataset](https://huggingface.co/datasets/noitomrobotics/HiPHI)
from Noitom Robotics provides BVH human motion and object meshes/trajectories;
it does not provide this actor's original human mesh. The rest skin below is a
reconstruction from the locally available SMPL-X model, not a recovered scan.

| Route | Motion MP4 | Motion PNG | Rest PNG | Rest OBJ |
| --- | --- | --- | --- | --- |
| Fitted SMPL-X | [Video](Placing-put_0030/smplx/human_mesh.mp4) | [Preview](Placing-put_0030/smplx/motion_preview.png) | [Rest](Placing-put_0030/smplx/rest_pose.png) | [Mesh](Placing-put_0030/smplx/body/rest_mesh.obj) |
| HIPHI rest skin | [Video](Placing-put_0030/rest_skin/human_mesh.mp4) | [Preview](Placing-put_0030/rest_skin/motion_preview.png) | [Rest](Placing-put_0030/rest_skin/rest_pose.png) | [Mesh](Placing-put_0030/rest_skin/body/rest_mesh.obj) |

Also see [motion comparison](Placing-put_0030/motion_comparison.png),
[rest comparison](Placing-put_0030/rest_pose_comparison.png), and
[measured validation](Placing-put_0030/metrics.json).

## SMPL-X route

`hiphi_to_umr.py` evaluates the BVH using each joint's position channels instead
of adding its OFFSET a second time. It fits 10 male SMPL-X shape coefficients
to actor height and limb lengths, calibrates joint definitions and canonical
bone directions, and optimizes pose/translation against the captured joints.
A rotation prior, temporal regularizer, and sole-ground penalty limit distortion.
Extra spine joints are collapsed; jaw/eyes have no captured articulation.

`smplx/poses.npy` is `(1778,165)` full axis-angle SMPL-X pose, with explicit
45-dimensional left/right hand poses. `transl.npy`, `betas.npy`, `gender.npy`,
`mocap_framerate.npy`, and `output_up.npy` use UMR's flat HOI input format.
The same parameters are also bundled in `smplx/motion.npz`.

Main-limb joint error is approximately **17.2 mm mean / 40.9 mm 95th percentile**.
This is a fitted motion, not an exact conversion of the different skeleton.

## Rest-skin route

The fitted template supplies a human surface, topology, and skin weights. The
converter reconstructs a rest mesh around HIPHI's measured rest skeleton,
calibrates the pelvis/clavicle/head definitions, and smooths its deformation
field. Animation uses HIPHI world rotations and translations directly, with
virtual spine/face joints where the two joint layouts differ. It does not use
SMPL-X pose correctives to produce the animated surface.

`rest_skin/skin.npz` contains the rest vertices, joints, faces, weights, and bone
mapping. `bone_world_rotations.npy` and `joints.npy` contain its animation:

```python
vertices_t = sum(weights[:, j, None] * (
    (rest_vertices - rest_joints[j]) @ bone_world_rotations[t, j].T
    + joints[t, j]
) for j in range(55))
```

`rest_skin/model/umr_smplx_overlay.json` exposes the matching rest surface to
UMR's correspondence training. `mesh_motion.json` selects the cached animated
surface through `vertex_cache_source.py`. The companion `poses.npy` is the
SMPL-X fit used for sequence metadata compatibility; **loading only that pose
file does not reproduce the rest-skin route**. Keep the cache manifest and
vertices/joints/faces files together, and use the supplied UMR config.

Both surfaces have 10,475 vertices and 20,908 triangles. The native joint error
is nearly zero by construction and is not an independent surface-quality score.

## Run

From the UMR repository directory:

```bash
# Rebuild both human mesh routes.
conda run -n umr python scripts/hiphi_to_umr.py

# Render the actual surfaces (PyTorch3D is installed in the humoto environment).
conda run -n humoto python scripts/render_hiphi_mesh.py

# Check UMR loading, HOI export, finger preservation, object basis and topology.
conda run -n umr python scripts/validate_hiphi_mesh.py

# Full UMR robot retargeting, SMPL-X route.
conda run -n umr python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --defaults sample_data/hiphi/Placing-put_0030/smplx/umr_config.json --skip-view

# Full UMR robot retargeting, reconstructed rest-skin route.
conda run -n umr python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --defaults sample_data/hiphi/Placing-put_0030/rest_skin/umr_config.json --skip-view
```

Both complete UMR runs were executed successfully: each produced finite
`qpos` with shape `(1778,36)`. Results:
[SMPL-X G1 motion](Placing-put_0030/smplx/g1_retarget.npz) and
[rest-skin G1 motion](Placing-put_0030/rest_skin/g1_retarget.npz).
The full human videos are separate from these robot trajectories; robot
physical feasibility has not been validated in a dynamics simulation.
Both H.264 MP4s were decoded in full: 1,778 frames each, 800 × 640 pixels,
30.0003 fps, and 59.266074 seconds. Decoded middle/end frames were visually
inspected. Verification details are saved in `metrics.json`.

The supplied configs preserve fingers, use full-size objects, and keep separate
correspondence caches. For a short integration check, append
`--start 444 --max-frames 3 --out /tmp/hiphi_g1_smoke.npz`.

The adapter preserves explicit fingers when `zero_source_finger_pose` is false;
existing configurations that request zeroed fingers still get zeroed fingers.
It also carries the rest-skin manifest through UMR's temporary HOI motion export.

## Quality and limits

The PNGs include six moments across the entire clip, plus another camera angle
and a rest view. The videos show the full source duration. Both routes retain
the recorded box track exactly; object-local vertices receive the same axis
change as UMR's conjugated object rotation.

These are reconstructed body surfaces, not collision-free contact solutions.
Measured worst floor intrusion is about **10.1 mm (SMPL-X)** and **13.0 mm
(rest skin)**. Finger/object contact and self-intersection are not explicitly
optimized, so these inputs should not be described as ground-truth actor
geometry or exact contact surfaces. The original HIPHI motion, the local
SMPL-X model, and their respective provenance remain identifiable.
