import pathlib, json, shutil, xml.etree.ElementTree as ET, subprocess,time,pickle
import numpy as np,joblib
from scipy.spatial.transform import Rotation as R,Slerp
root=pathlib.Path('/home/natcha/Downloads/UMR');out=root.parent/'umr_wrist_study';src=root/'sample_data/omomo/train_and_test/sub10_smallbox_084'
p=np.load(src/'poses.npy').reshape(166,55,3)
with (root/'smpl/SMPLX_FEMALE.pkl').open('rb') as f: model=pickle.load(f,encoding='latin1')
parents=np.asarray(model['kintree_table'])[0]; print('parents',parents[:22].tolist(),flush=True)
for split in ['test','train']:
 data=joblib.load(root.parent/f'omomo_release/data/{split}_diffusion_manip_seq_joints24.p')
 matches=[v for v in data.values() if str(v['seq_name'])=='sub10_smallbox_084']
 if matches:
  raw=np.asarray(matches[0]['pose_body'],dtype=np.float32).reshape(166,21,3)
  print('raw split',split,'max abs difference',float(abs(raw-p[:,1:22]).max()),flush=True)
  (out/'source_verification.json').write_text(json.dumps({'split':split,'max_abs_difference':float(abs(raw-p[:,1:22]).max()),'parents':parents[:22].astype(int).tolist()},indent=2));break
for label,joints in [('source_wrist_slerp',[20]),('source_elbow_wrist_slerp',[18,20])]:
 dst=out/label/'sub10_smallbox_084';dst.mkdir(parents=True,exist_ok=True)
 for f in src.iterdir():
  if f.is_file():shutil.copy2(f,dst/f.name)
 tree=ET.parse(dst/'smallbox.xml');compiler=tree.getroot().find('compiler');compiler.set('meshdir',str((src/compiler.get('meshdir')).resolve()));tree.write(dst/'smallbox.xml')
 pose=p.copy()
 for j in joints:pose[61:72,j]=Slerp([60,72],R.from_rotvec(p[[60,72],j]))(np.arange(61,72)).as_rotvec()
 np.save(dst/'poses.npy',pose.reshape(166,165))
 cmd=['/home/natcha/miniconda3/envs/umr/bin/python','scripts/humanoid_retarget_pipeline_hsi_hoi.py','--defaults',str(out/'seed0_a.json'),'--data',str(dst),'--seq-key','sub10_smallbox_084','--stage','retarget','--skip-view','--out',str(out/(label+'.npz'))]
 t=time.monotonic()
 with (out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 result={'name':label,'returncode':r.returncode,'seconds':time.monotonic()-t};print(json.dumps(result),flush=True)
 with (out/'timings.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
 if r.returncode:break
