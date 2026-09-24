#!/usr/bin/env python3
"""Render actual HiPHI conversion vertices, with recorded object track, to PNG/MP4.

Uses the local humoto environment (PyTorch3D); no generated or illustrative body
images. Camera and floor are fixed in the common source world frame.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import trimesh
import subprocess
import shutil
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader,
    PointLights, TexturesVertex, BlendParams)

ROOT = Path(__file__).resolve().parents[1]
try:
    FONT = ImageFont.truetype('DejaVuSans.ttf', 20)
    SMALL = ImageFont.truetype('DejaVuSans.ttf', 16)
except OSError:
    FONT = SMALL = ImageFont.load_default()


def floor_mesh(center, extent=3., n=24):
    v, f, c = [], [], []
    for i in range(n):
        for j in range(n):
            x, z = center[0]-extent+2*extent*i/n, center[2]-extent+2*extent*j/n
            step=2*extent/n; k=len(v)
            v.extend([[x,0,z],[x+step,0,z],[x+step,0,z+step],[x,0,z+step]])
            f.extend([[k,k+2,k+1],[k,k+3,k+2]])
            c.extend([[.31,.35,.40] if (i+j)%2 else [.38,.42,.47]]*4)
    return np.array(v,np.float32),np.array(f,np.int64),np.array(c,np.float32)


class Renderer:
    def __init__(self, faces, center, color, obj=None, size=(640,800), azim=65, dist=3.5):
        self.dev='cuda'; self.size=size
        fv,ff,fc=floor_mesh(center)
        self.floor=torch.tensor(fv,device=self.dev)
        nv=int(faces.max())+1
        arrays=[faces]; colors=[np.tile(color,(nv,1))]; offset=nv
        self.obj=obj
        if obj is not None:
            arrays.append(np.asarray(obj.faces)+offset)
            colors.append(np.tile([.84,.47,.20],(len(obj.vertices),1)))
            offset+=len(obj.vertices)
        arrays.append(ff+offset); colors.append(fc)
        self.faces=torch.tensor(np.concatenate(arrays),device=self.dev,dtype=torch.int64)
        self.colors=torch.tensor(np.concatenate(colors),device=self.dev,dtype=torch.float32)[None]
        r,t=look_at_view_transform(dist=dist,elev=12,azim=azim,at=(tuple(center),))
        cam=FoVPerspectiveCameras(device=self.dev,R=r,T=t,fov=42)
        lights=PointLights(device=self.dev,location=[np.asarray(center)+[2,4,3]],
            ambient_color=[[.5,.5,.5]],diffuse_color=[[.6,.6,.6]],specular_color=[[.05,.05,.05]])
        self.render=MeshRenderer(MeshRasterizer(cameras=cam,raster_settings=RasterizationSettings(
            image_size=size,blur_radius=0,faces_per_pixel=1,max_faces_per_bin=80000)),
            SoftPhongShader(device=self.dev,cameras=cam,lights=lights,blend_params=BlendParams(background_color=(.87,.90,.94))))

    @torch.no_grad()
    def frame(self, v, ov=None):
        verts=[torch.as_tensor(np.array(v),device=self.dev,dtype=torch.float32)]
        if ov is not None: verts.append(torch.tensor(ov,device=self.dev,dtype=torch.float32))
        verts.append(self.floor)
        mesh=Meshes(verts=[torch.cat(verts)],faces=[self.faces],textures=TexturesVertex(self.colors))
        return (self.render(mesh)[0,:,:,:3].clamp(0,1).cpu().numpy()*255).astype(np.uint8)


def label(a,title,subtitle=''):
    im=Image.fromarray(a); dr=ImageDraw.Draw(im)
    dr.rectangle((0,0,im.width,65),fill=(28,35,46))
    dr.text((16,8),title,font=FONT,fill='white')
    dr.text((16,36),subtitle,font=SMALL,fill=(205,218,235))
    return np.asarray(im)


def main(args):
    torch.set_num_threads(4)
    base=args.sequence
    md=json.loads((base/'metrics.json').read_text())
    arrays={r:np.load(base/r/'vertices.npy',mmap_mode='r') for r in ('smplx','rest_skin')}
    faces=np.load(base/'smplx/faces.npy')
    rows=np.loadtxt(base/'smplx/prop_Box_A_1.csv',delimiter=',',skiprows=1)
    mesh=trimesh.load(ROOT.parent/'HiPHI/object_meshes/Box_A_1.obj',force='mesh',process=False)
    mesh.vertices*=.01
    ov=np.einsum('tij,vj->tvi',Rotation.from_quat(rows[:,3:7]).as_matrix(),mesh.vertices)+rows[:,:3,None].transpose(0,2,1)
    center=[1.95,.86,-2.60]
    titles={'smplx':'SMPL-X | fitted body + fingers','rest_skin':'HIPHI rest skin | reconstructed body'}
    colors={'smplx':[.32,.63,.83],'rest_skin':[.48,.73,.59]}
    renderers={r:Renderer(faces,center,colors[r],mesh) for r in arrays}
    n=len(rows); fps=md['fps']
    # Include the lowest box pose and representative frames across the clip.
    keys=sorted(set([0,int(np.argmin(rows[:,1])),n//4,n//2,3*n//4,n-1]))
    for r,vs in arrays.items():
        ims=[]
        for k in keys:
            a=label(renderers[r].frame(vs[k],ov[k]),titles[r],f"Placing-put_0030  |  {k/fps:.2f} s  |  frame {k*md['stride']} at 90 Hz")
            Image.fromarray(a).save(base/r/f'frame_{k:04d}.png')
            ims.append(Image.fromarray(a).resize((600,480)))
        grid=Image.new('RGB',(1800,960),(28,35,46))
        for i,im in enumerate(ims): grid.paste(im,((i%3)*600,(i//3)*480))
        grid.save(base/r/'motion_preview.png')
        stress=md.get('validation',{}).get(r,{}).get('largest_vertex_step_frame')
        if stress is not None:
            a=label(renderers[r].frame(vs[stress],ov[stress]),titles[r],f'Largest motion step | {stress/fps:.2f} s')
            Image.fromarray(a).save(base/r/'largest_motion_step.png')
        rest=trimesh.load(base/r/'body/rest_mesh.obj',force='mesh',process=False)
        rv=np.asarray(rest.vertices).copy()
        rv[:,0]-=(rv[:,0].max()+rv[:,0].min())/2
        rv[:,2]-=(rv[:,2].max()+rv[:,2].min())/2
        rv[:,1]-=rv[:,1].min()
        rr=Renderer(faces,[0,.9,0],colors[r],size=(800,800),azim=15,dist=3.5)
        a=label(rr.frame(rv),titles[r],f"Rest mesh | actor {md['actor'].get('gender')} | 10,475 vertices")
        Image.fromarray(a).save(base/r/'rest_pose.png')
        rr2=Renderer(faces,center,colors[r],mesh,azim=155)
        k=int(np.argmin(rows[:,1]))
        Image.fromarray(label(rr2.frame(vs[k],ov[k]),titles[r],'Alternate view | lowest box pose')).save(base/r/'alternate_view.png')
    comparison=np.concatenate([np.asarray(Image.open(base/r/'rest_pose.png')) for r in arrays],axis=1)
    Image.fromarray(comparison).save(base/'rest_pose_comparison.png')
    comparison=np.concatenate([np.asarray(Image.open(base/r/'motion_preview.png')) for r in arrays],axis=0)
    Image.fromarray(comparison).save(base/'motion_comparison.png')
    print('Previews ready',flush=True)
    if args.preview_only: return
    ffmpeg=shutil.which('ffmpeg') or str(next((Path.home()/'miniconda3/envs/umr/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*')))
    writers={r:subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
        '-s','800x640','-pix_fmt','rgb24','-r',str(fps),'-i','-','-an','-c:v','libx264',
        '-threads','4','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(base/r/'human_mesh.mp4')], stdin=subprocess.PIPE) for r in arrays}
    try:
        for k in range(n):
            for r,vs in arrays.items():
                a=label(renderers[r].frame(vs[k],ov[k]),titles[r],f'Placing-put_0030  |  {k/fps:.2f} / {n/fps:.2f} s  |  original box track')
                writers[r].stdin.write(a.tobytes())
            if k%150==0: print(f'Render {k}/{n}',flush=True)
    finally:
        for w in writers.values():
            w.stdin.close()
            if w.wait() != 0: raise RuntimeError("ffmpeg encoding failed")
    print('MP4s ready',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence',type=Path,default=ROOT/'sample_data/hiphi/Placing-put_0030')
    ap.add_argument('--preview-only',action='store_true')
    main(ap.parse_args())
