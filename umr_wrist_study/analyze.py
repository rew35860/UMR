import pathlib,json,re,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
root=pathlib.Path(__file__).resolve().parent
base=np.load(root/'seed0_a.npz',allow_pickle=True);q=base['qpos'];names=base['robot_joint_names'];ids=[i+7 for i,n in enumerate(names) if 'wrist' in n]
timings={x['name']:x['seconds'] for x in map(json.loads,(root/'timings.jsonl').read_text().splitlines())}
rows=[]
for path in sorted(root.glob('*.npz')):
 b=np.load(path)['qpos']; log=path.with_suffix('.log').read_text();cost=re.search(r'cost mean=([\d.]+) max=([\d.]+)',log)
 rows.append(dict(run=path.stem,seconds=round(timings.get(path.stem,0),2),max_qpos_diff=float(abs(b-q).max()),max_wrist_diff_deg=float(np.rad2deg(abs(b[:,ids]-q[:,ids]).max())),left_roll_min_deg=float(np.rad2deg(b[:,ids[0]].min())),left_roll_max_deg=float(np.rad2deg(b[:,ids[0]].max())),max_wrist_step_deg=float(np.rad2deg(abs(np.diff(b[:,ids],axis=0)).max())),left_roll_limit_frames=int(np.sum(np.isclose(b[:,ids[0]],-1.97222,atol=.001))),logged_cost_mean=float(cost[1]) if cost else None))
(root/'metrics.json').write_text(json.dumps(rows,indent=2))
with (root/'metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
p=np.load(root.parent/'UMR/sample_data/omomo/train_and_test/sub10_smallbox_084/poses.npy').reshape(166,55,3)
global_hand=R.identity(166)
for j in [0,3,6,9,13,16,18,20]:global_hand=global_hand*R.from_rotvec(p[:,j])
step=np.rad2deg((global_hand[:-1].inv()*global_hand[1:]).magnitude())
fig,axs=plt.subplots(3,1,figsize=(11,10),sharex=True,layout='constrained')
axs[0].plot(np.arange(1,166)/30,step,color='black',label='Source left-hand world rotation change')
axs[0].set_ylabel('Degrees / frame');axs[0].legend();axs[0].set_title('sub10_smallbox_084: scaled box, cached correspondence')
for n in ['seed0_a','seed1','seed2','seed3']:
 b=np.load(root/(n+'.npz'))['qpos'];axs[1].plot(np.arange(166)/30,np.rad2deg(b[:,ids[0]]),label=n.replace('seed0_a','seed 0 (three identical runs)'))
axs[1].set_ylabel('Left wrist roll (degrees)');axs[1].legend()
for n in ['seed0_a','no_object_collision','hand_normals5','source_wrist_slerp','source_elbow_wrist_slerp']:
 path=root/(n+'.npz')
 if path.exists():axs[2].plot(np.arange(166)/30,np.rad2deg(np.load(path)['qpos'][:,ids[0]]),label=n)
axs[2].set_ylabel('Left wrist roll (degrees)');axs[2].set_xlabel('Time (seconds)');axs[2].legend()
for ax in axs:
 ax.axvspan(61/30,72/30,color='red',alpha=.09);ax.grid(alpha=.25)
fig.savefig(root/'wrist_diagnosis.png',dpi=160)
for row in rows:print(json.dumps(row))
print('source world jumps',[(i,int(i+1),float(step[i])) for i in np.flatnonzero(step>90)])
