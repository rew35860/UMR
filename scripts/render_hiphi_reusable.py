#!/usr/bin/env python3
"""Render real shared-skin geometry and all recorded objects; use humoto Python."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from pytorch3d.renderer import look_at_view_transform
from render_hiphi_mesh import Renderer, label
from hiphi_skinning import evaluate_manifest, load_template


def render(seq, video_stride, preview_only):
    manifest_path = seq / 'mesh_motion.json'
    manifest = json.loads(manifest_path.read_text())
    report = json.loads((seq / 'conversion.json').read_text())
    asset = load_template(str((seq / manifest['skin']).resolve()), manifest['skin_sha256'])
    tracks, meshes = [], []
    for obj in manifest['objects']:
        mesh = trimesh.load(seq / obj['mesh'], force='mesh', process=False)
        # Exported OBJ locals are Z-up; the recorded prop track is Y-up.
        v = np.asarray(mesh.vertices)[:, [0, 2, 1]].copy()
        v[:, 2] *= -1
        mesh.vertices = v
        meshes.append(mesh)
        tracks.append(np.atleast_2d(np.loadtxt(seq / obj['trajectory'], delimiter=',', skiprows=1)))
    combined = trimesh.util.concatenate(meshes) if meshes else None
    joints = np.load(seq / 'joints.npy', mmap_mode='r')
    n, fps = len(joints), report['fps']
    # Fixed view for each clip; include the complete root trajectory and body.
    low, high = joints.min(axis=(0, 1)), joints.max(axis=(0, 1))
    center = (low + high) / 2
    center[1] = .85
    distance = max(3.5, float(np.linalg.norm(high - low)) * 1.6)
    renderer = Renderer(asset['faces'], center, [.48, .73, .59], combined, dist=distance)

    def frame(k, vertices=None):
        # Track the body horizontally so long carrying paths do not shrink it
        # to a few pixels. The mesh, objects and floor stay in world coordinates.
        focus = joints[k, 0].copy()
        focus[1] = .85
        r, t = look_at_view_transform(dist=3.8, elev=12, azim=65, at=(tuple(focus),))
        camera = renderer.render.rasterizer.cameras
        camera.R, camera.T = r.to(renderer.dev), t.to(renderer.dev)
        if vertices is None:
            vertices = evaluate_manifest(manifest_path, np.array([k]))[0][0]
        objects = [mesh.vertices @ Rotation.from_quat(rows[k, 3:7]).as_matrix().T + rows[k, :3]
                   for mesh, rows in zip(meshes, tracks)]
        ov = np.concatenate(objects) if objects else None
        return label(renderer.frame(vertices, ov), f'HiPHI shared skin | {report["actor_id"]}',
                     f'{seq.name} | {k/fps:.2f} s | same mesh, recorded motion')

    keys = np.linspace(0, n - 1, 6, dtype=int)
    frames = [Image.fromarray(frame(k)) for k in keys]
    grid = Image.new('RGB', (1800, 960))
    for i, im in enumerate(frames):
        im.save(seq / f'frame_{keys[i]:04d}.png')
        grid.paste(im.resize((600, 480)), ((i % 3) * 600, (i // 3) * 480))
    grid.save(seq / 'motion_preview.png')
    rest_png = (seq / manifest['skin']).resolve().parent / 'rest_pose.png'
    if not rest_png.exists():
        rest = asset['rest_vertices'].copy()
        rest[:, 1] -= rest[:, 1].min()
        rest[:, [0, 2]] -= (rest[:, [0, 2]].max(0) + rest[:, [0, 2]].min(0)) / 2
        rr = Renderer(asset['faces'], [0, .9, 0], [.48, .73, .59], size=(800, 800), azim=15, dist=3.5)
        Image.fromarray(label(rr.frame(rest), 'HiPHI | reusable rest skin',
                             'One shared mesh | 10,475 vertices | reconstructed proxy')).save(rest_png)
    print(f'PNG ready: {seq.name}', flush=True)
    if preview_only:
        return
    ffmpeg = shutil.which('ffmpeg') or str(next((Path.home() / 'miniconda3/envs/umr/lib').glob(
        'python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*')))
    writer = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                               '-s', '800x640', '-pix_fmt', 'rgb24', '-r', str(fps / video_stride), '-i', '-',
                               '-an', '-c:v', 'libx264', '-threads', '4', '-crf', '19', '-pix_fmt', 'yuv420p',
                               '-movflags', '+faststart', str(seq / 'human_mesh.mp4')], stdin=subprocess.PIPE)
    ids = np.arange(0, n, video_stride)
    try:
        for start in range(0, len(ids), 32):
            batch = ids[start:start + 32]
            vertices = evaluate_manifest(manifest_path, batch)[0]
            for k, v in zip(batch, vertices):
                writer.stdin.write(frame(k, v).tobytes())
            if start % 160 == 0:
                print(f'{seq.name}: video {start}/{len(ids)}', flush=True)
    finally:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError('ffmpeg encoding failed')
    (seq / 'render.json').write_text(json.dumps(dict(video_frames=len(ids), fps=fps / video_stride,
        video_stride=video_stride, motion_frames=n, duration_seconds=len(ids) / (fps / video_stride)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'sample_data/hiphi_reusable')
    parser.add_argument('--sequence', action='append')
    parser.add_argument('--video-stride', type=int, default=3, help='3 gives ~10 fps previews of the full ~30 fps motion.')
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    if args.video_stride < 1:
        parser.error('--video-stride must be positive')
    torch.set_num_threads(4)
    for seq in sorted((args.root / 'sequences').iterdir()):
        if (seq / 'mesh_motion.json').exists() and (not args.sequence or seq.name in args.sequence):
            render(seq, args.video_stride, args.preview_only)
