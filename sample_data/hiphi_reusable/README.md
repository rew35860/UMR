# Reusable HiPHI human skin for UMR

The default export uses **one shared rest mesh for every sequence and actor**.
The mesh is calibrated once from `Placing-put_0030` / actor A008. New clips do
not reconstruct it or fit SMPL-X again. The same vertices, faces, skin weights,
bind joints, and calibration are reused, so surface correspondences have a
stable meaning across motions and objects.

## Files

- `templates/shared/body/rest_mesh.obj`: the single reusable human mesh.
- `templates/shared/skin.npz`: bind vertices/joints, weights, faces and calibration.
- `templates/shared/model/`: UMR's correspondence-template overlay.
- `templates/shared/rest_pose.png`: rendered rest mesh.
- `overview.png`: the same mesh animated across the nine checked recordings.
- `sequences/<motion_id>/mesh_motion.json`: relative reference to the shared skin.
- `sequences/<motion_id>/bone_world_rotations.npy` and `joints.npy`: recorded,
  calibrated motion in metres, Y-up. They replace a large per-frame vertex cache.
- `sequences/<motion_id>/body/rest_mesh.obj`: symlink to the shared OBJ, not a copy.
- `sequences/<motion_id>/prop_<object_id>.csv`, OBJ and XML: all recorded objects.
- `sequences/<motion_id>/umr_config.json`: settings for later UMR use.
- `sequences/<motion_id>/motion_preview.png`, `human_mesh.mp4`: actual skinned
  geometry and recorded object tracks. Videos show the full clip at about 10 fps;
  motion arrays retain about 30 fps. The camera follows the body horizontally.
- `batch_report.json` and `validation.json`: conversion and geometry/loader checks.

## Convert another clip or the extracted dataset

From the UMR directory:

```bash
/home/natcha/miniconda3/envs/umr/bin/python scripts/hiphi_reusable_skin.py
```

This discovers all extracted HiPHI folders under `../HiPHI`, using `metadata.json`
and `motion_actor.bvh`. It reuses the existing template and completed exports.
Add `--sequence MOTION_ID` (repeatable) to select clips, or `--dataset-root PATH`
for another dataset location. Keep the same `--out` to reuse the existing skin
with later dataset shards; those shards do not need the calibration BVH.

The default `--stride 3` samples the approximately 90 Hz recording at about 30 Hz.
Use `--stride 1` and a **new output directory** to retain the original frame rate.
On the first build, `--reference MOTION_ID` can select another calibration clip.
Existing templates are never silently refitted. Changed source content, stride,
or template identity requires a new output directory. Failed clips are reported
individually; the batch continues and exits nonzero if any failed.

## Optional actor-specific proportions

```bash
/home/natcha/miniconda3/envs/umr/bin/python scripts/hiphi_reusable_skin.py \
  --template-mode actor --out sample_data/hiphi_reusable_actors
```

This fits one template per actor and reuses it for every clip from that actor.
Eight local clips have also been converted in this mode: six templates total;
the three A035 clips share exactly one template. Each distinct template gets
its own correspondence cache. Use the default shared mode when one surface
correspondence model across the dataset is the priority.

## UMR integration

`vertex_cache_source.py` now evaluates `umr_hiphi_skin_v2` on demand using
world-space linear blend skinning. Both UMR's flat-motion loader and its exported
NPZ loader use this path. The older `umr_vertex_cache_v1` route remains supported.
Skinning computes vertices for requested frames; a caller requesting an entire
clip still receives the entire clip in memory, but storage no longer duplicates
all human vertices for every frame.

The generated `correspondence.shared_template_key` makes the HSI/HOI pipeline
reuse the same correspondence dataset and training directory for the same skin,
robot configuration, and correspondence settings. Sequence IDs and object IDs
do not enter that identity. Changed training settings generate a different cache
path. As with UMR's existing caches, changes to external robot asset file contents
require explicitly rebuilding the correspondence assets.

No new UMR training or robot retargeting has been run for these reusable exports.
Only geometry generation, rendering, and UMR loader/configuration checks were run.
Older sequence-specific outputs in `sample_data/hiphi/Placing-put_0030` are separate;
the earlier SMPL-X and rest-skin routes already have retarget results.

After authorization, a typical later UMR command would be:

```bash
python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults sample_data/hiphi_reusable/sequences/Placing-put_0030/umr_config.json \
  --stage all --skip-view
```

The `poses.npy` angles are **unfitted mapped rotations for layout compatibility**,
not an SMPL-X reconstruction. They must be used with `mesh_motion.json`; using
them as ordinary SMPL-X poses produces a different surface. A marker prevents
UMR from silently falling back when this required manifest is missing. The model
gender in a shared export describes the proxy template; the recorded actor's
gender and other metadata remain in `conversion.json`.

## Objects, coverage and limitations

Human skinning does not depend on the object category or object count. Body-only
exports disable object constraints. Multiple objects retain separate tracks and
meshes, including separate instances of the same mesh. For multiple-object clips,
UMR currently constrains **one selected object per run**: use the generated
`umr_config_<object_id>.json`. The generic config requires explicit selection and
will not silently choose the first object. This is not simultaneous multi-object
robot optimization.

The eight original local clips plus the real mirrored `Placing-put_0030__mirror`
recording span six actors, 163–183 cm, both recorded genders, and box, table, ball,
stool/chair, and bucket interactions. Shared-mode validation checked all 16,229
output frames, exact template reuse, finite geometry, preserved
recorded limb/hand positions, skinning invariance, flat/NPZ loading, correspondence
cache identity, and legacy-cache compatibility. Body-only and two-object exports
were tested with controlled fixtures, not real captured examples. The mirrored
recording uses the same shared skin and its released mirrored object asset; the
BVH and object tracks are read as recorded with no extra reflection.

This is a reconstructed body proxy derived from SMPL-X, not a released HiPHI
actor scan or the full original capture rig. It uses 55 mapped joints, including
collapsed spine links and virtual face joints. Shared mode retains one template's
body thickness and shape while following each actor's recorded joints. Skin
accuracy and hand/object contact are approximate. Some upright-table carrying
poses visibly intersect the shared body proxy. Floor penetration reaches about
1.9 cm in the checked shared clips; these meshes are not collision-free. Compatible
joint names, hierarchy, channels, units, and frame alignment are required, and
incompatible inputs fail explicitly. The entire HiPHI release has not been tested.

## Reproduce checks and previews

```bash
/home/natcha/miniconda3/envs/umr/bin/python scripts/validate_hiphi_reusable.py
/home/natcha/miniconda3/envs/humoto/bin/python scripts/render_hiphi_reusable.py
```

Use `--root sample_data/hiphi_reusable_actors` for the actor exports. Rendering also
supports repeated `--sequence MOTION_ID`, `--preview-only`, and `--video-stride 1`
for full-rate video. Moving the whole directory preserves relative skin links;
absolute model/config paths must be updated if relocating to another machine.
The underlying SMPL-X assets and their license are still required.
