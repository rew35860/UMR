import json,pathlib,subprocess,time
root=pathlib.Path('/home/natcha/Downloads/UMR'); out=root.parent/'umr_wrist_study'
base=json.loads((out/'seed0_a.json').read_text())
variants={'bidirectional':{'trajectory_warm_start_mode':'bidirectional'},'no_object_collision':{'robot_object_hard_constraint':False},'hand_normals5':{'body_segment':{**base['solver']['body_segment'],'cost_config':{n:{'sample_slots':15,'point_cost':10.,'normal_cost':5.} for n in ['leftHand','rightHand']}}},'all_hand_points':{'body_segment':{**base['solver']['body_segment'],'cost_config':{n:{'sample_slots':4096,'point_cost':10.,'normal_cost':1.} for n in ['leftHand','rightHand']}}}}
for name,changes in variants.items():
 cfg=json.loads(json.dumps(base));cfg['solver'].update(changes);config=out/(name+'.json');config.write_text(json.dumps(cfg,indent=2))
 cmd=['/home/natcha/miniconda3/envs/umr/bin/python','scripts/humanoid_retarget_pipeline_hsi_hoi.py','--defaults',str(config),'--data','sample_data/omomo','--seq-key','sub10_smallbox_084','--stage','retarget','--skip-view','--out',str(out/(name+'.npz'))]
 t=time.monotonic()
 with (out/(name+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 result={'name':name,'returncode':p.returncode,'seconds':time.monotonic()-t};print(json.dumps(result),flush=True)
 with (out/'timings.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
 if p.returncode:break
