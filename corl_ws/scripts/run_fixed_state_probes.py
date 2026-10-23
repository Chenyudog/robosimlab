"""Matched observation-only target shifts, feature differences, held-out scenes."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import json,hashlib,copy
import numpy as np
import torch
from run_baseline import ROOT,Policy,save_json,benchmark,get_libero_path,OffScreenRenderEnv
from check_counterfactual import capture,restore
from collect_probe_activations import Recorder
OUT=ROOT/'outputs/fixed_state_probe_v2'
SHIFTS=[(axis,dx) for axis in (0,1) for dx in (-.01,-.005,.005,.01)]
SPLIT={'train':list(range(20,32)),'validation':list(range(32,36)),'test':list(range(36,40))}

def penetrations(env,target):
    # Include all descendant bodies, not only each object's root body.
    def under(body,root):
        while body>0:
            if body==root:return True
            body=int(env.sim.model.body_parentid[body])
        return False
    root=env.env.obj_body_id[target]
    other=[env.env.obj_body_id[n] for n in env.env.objects_dict if n!=target]
    target_geoms={g for g in range(env.sim.model.ngeom) if under(int(env.sim.model.geom_bodyid[g]),root)}
    other_geoms={g for g in range(env.sim.model.ngeom) if any(under(int(env.sim.model.geom_bodyid[g]),o) for o in other)}
    return [float(c.dist) for c in env.sim.data.contact[:env.sim.data.ncon]
            if ((c.geom1 in target_geoms and c.geom2 in other_geoms) or (c.geom2 in target_geoms and c.geom1 in other_geoms)) and c.dist < -1e-4]

def image_summary(obs):
    return np.concatenate([obs[k].astype(float).mean((0,1)) for k in ['agentview_image','robot0_eye_in_hand_image']])/255.

def collect():
    OUT.mkdir(exist_ok=True)
    protocol={'initial_states':list(range(20,40)),'split':SPLIT,'shifts_axis_metres':SHIFTS,'task_id':0,
        'seed':7,'target':'akita_black_bowl_1','features':'shifted minus nominal pooled activations; same state and noise',
        'target_label':'requested world x,y displacement in metres; no label input to policy',
        'sampling':'After10 settle steps at reset, no physics step after each target displacement.',
        'scope':'New exploratory scenes; not robot success or feedback repair. Fixed-state control excludes state change.'}
    # JSON round trip normalizes tuples so resumption compares the same representation.
    protocol=json.loads(json.dumps(protocol));f=OUT/'protocol.json'
    if f.exists() and json.loads(f.read_text())!=protocol:raise RuntimeError('Protocol drift')
    save_json(f,protocol);save_json(OUT/'status.json',{'stage':'loading_model'})
    torch.set_num_threads(4);policy=Policy(ROOT/'models/pi0_libero_finetuned',vision_fp32=True);recorder=Recorder(policy.model)
    suite=benchmark.get_benchmark_dict()['libero_spatial']();task=suite.get_task(0)
    initial_states=torch.load(get_libero_path('init_states')+'/'+task.problem_folder+'/'+task.init_states_file.split('/')[-1],weights_only=False)
    if len(initial_states)<40:raise RuntimeError('At least40 initial states required')
    env=OffScreenRenderEnv(bddl_file_name=get_libero_path('bddl_files')+'/'+task.problem_folder+'/'+task.bddl_file,camera_heights=256,camera_widths=256)
    try:
        for i in range(20,40):
            output=OUT/f'scene_{i:03d}.npz'
            if output.exists():continue
            save_json(OUT/'status.json',{'stage':'collecting','initial_state':i})
            np.random.seed(7);torch.manual_seed(7);env.seed(7);env.reset();obs=env.set_init_state(initial_states[i])
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            # env.step observation can precede its final mj_forward by a few micrometres.
            # Canonicalize nominal and shifted observations with the same refresh sequence.
            env.sim.forward();env._update_observables(force=True);obs=env.env._get_observations()
            nominal=copy.deepcopy(obs);snapshot=capture(env);target='akita_black_bowl_1'
            if target not in env.obj_of_interest:raise RuntimeError('Wrong target')
            body=env.env.obj_body_id[target];joint=env.env.objects_dict[target].joints[0]
            basepos=env.sim.data.body_xpos[body].copy();baseq=env.sim.data.get_joint_qpos(joint).copy()
            base_robot=env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
            recorder.start();base_action=policy.infer(nominal,task.language,7);basefeatures=recorder.finish()
            features={k:[] for k in basefeatures};labels=[];pixel=[];actiondelta=[];invalid=[]
            for axis,amount in SHIFTS:
                restore(env,snapshot);q=baseq.copy();q[axis]+=amount
                env.sim.data.set_joint_qpos(joint,q);env.sim.forward()
                expected=np.zeros(3);expected[axis]=amount
                if not np.allclose(env.sim.data.body_xpos[body]-basepos,expected,atol=1e-10,rtol=0):raise RuntimeError('Object shift mismatch')
                if not np.array_equal(base_robot,env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes]):raise RuntimeError('Robot changed')
                pen=penetrations(env,target)
                if pen:
                    invalid.append({'axis':axis,'shift':amount,'reason':'interpenetration','minimum_contact_dist':min(pen)});continue
                env._update_observables(force=True);shifted=env.env._get_observations()
                for key in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']:
                    if not np.allclose(shifted[key],nominal[key],atol=1e-12,rtol=0):raise RuntimeError(f'State changed {key}')
                if np.array_equal(shifted['agentview_image'],nominal['agentview_image']):
                    invalid.append({'axis':axis,'shift':amount,'reason':'no_visible_image_change'});continue
                recorder.start();action=policy.infer(shifted,task.language,7);shiftfeatures=recorder.finish()
                for key in features:features[key].append(shiftfeatures[key]-basefeatures[key])
                labels.append(expected[:2]);pixel.append(image_summary(shifted)-image_summary(nominal));actiondelta.append(action-base_action)
            save_json(OUT/f'scene_{i:03d}_validity.json',{'initial_state':i,'valid':len(labels),'invalid':invalid})
            if not labels:raise RuntimeError(f'No valid shifts at scene{i}; inspect before continuing')
            np.savez_compressed(output,**{k:np.stack(v) for k,v in features.items()},target=np.stack(labels),
                pixel_mean=np.stack(pixel),state_delta=np.zeros((len(labels),8)),action_delta=np.stack(actiondelta))
            print('FIXED_STATE',i,'valid',len(labels),'invalid',len(invalid),flush=True)
    finally:
        env.close()
        for h in recorder.handles:h.remove()
    save_json(OUT/'status.json',{'stage':'fitting'})

def fit():
    episodes={i:np.load(OUT/f'scene_{i:03d}.npz') for i in range(20,40)}
    keys=[k for k in episodes[20].files if k.startswith(('prefix_','expert_'))]+['pixel_mean','state_delta']
    labels={s:np.concatenate([episodes[i]['target'] for i in ids]) for s,ids in SPLIT.items()}
    mse=lambda a,b:float(np.mean((a-b)**2));constant=labels['train'].mean(0)
    constant_mse=mse(labels['test'],constant);reports=[];predictions={}
    for key in keys:
        xs={s:np.concatenate([episodes[i][key] for i in ids]).astype(np.float64) for s,ids in SPLIT.items()}
        mean=xs['train'].mean(0);std=xs['train'].std(0);std[std<1e-6]=1
        xs={s:(a-mean)/std for s,a in xs.items()};x=xs['train'];yc=labels['train']-constant;gram=x@x.T
        trials=[]
        for alpha in (.1,1.,10.,100.,1000.):
            dual=np.linalg.solve(gram+alpha*np.eye(len(x)),yc)
            trials.append((mse(xs['validation']@x.T@dual+constant,labels['validation']),alpha,dual))
        val,alpha,dual=min(trials,key=lambda a:a[0]);pred=xs['test']@x.T@dual+constant;predictions[key]=pred
        # Permute labels separately within each scene; preserves its target distribution.
        rng=np.random.default_rng(7);shuffled=np.concatenate([rng.permutation(episodes[i]['target']) for i in SPLIT['train']])-constant
        random_dual=np.linalg.solve(gram+alpha*np.eye(len(x)),shuffled)
        direction=np.sum(pred*labels['test'],axis=1)>0
        reports.append({'feature':key,'alpha':alpha,'validation_mse_m2':val,'test_mse_m2':mse(pred,labels['test']),
            'test_coordinate_rmse_mm':float(np.sqrt(mse(pred,labels['test']))*1000),
            'test_r2_vs_training_mean':1-mse(pred,labels['test'])/constant_mse,
            'test_correct_direction_fraction':float(direction.mean()),
            'shuffled_train_label_test_rmse_mm':float(np.sqrt(mse(xs['test']@x.T@random_dual+constant,labels['test']))*1000)})
    best=min((r for r in reports if r['feature'].startswith(('prefix_','expert_'))),key=lambda r:r['validation_mse_m2'])
    result={'split':SPLIT,'samples':{s:len(y) for s,y in labels.items()},'selected_by_validation':best['feature'],
        'constant_test_rmse_mm':float(np.sqrt(constant_mse)*1000),'probes':reports,
        'limitations':['One task, four held-out scenes; exploratory, no significance claim.',
        'Robot state, inference noise and task text fixed within each scene. Target labels never enter policy.',
        'Physically invalid shifts excluded individually; axes and directions may have unequal coverage.',
        'Readout of imposed image changes is not proof of action use, object semantics or closed-loop correction.',
        'Correct direction metric measures positive dot product, not precise angle accuracy.']}
    save_json(OUT/'results.json',result);np.savez_compressed(OUT/'test_predictions.npz',target=labels['test'],**predictions)
    text='# Fixed-state displacement probes\n\n'+f'Samples: {result["samples"]}; validation-selected: {best["feature"]}; constant test RMSE: {result["constant_test_rmse_mm"]:.3f} mm.\n\n'
    text+='| Feature | Validation MSE | Test coordinate RMSE(mm) | Test R2 | Correct direction | Shuffled labels RMSE(mm) |\n|---|---:|---:|---:|---:|---:|\n'
    for r in reports:text+=f'| {r["feature"]} | {r["validation_mse_m2"]:.8f} | {r["test_coordinate_rmse_mm"]:.3f} | {r["test_r2_vs_training_mean"]:.3f} | {r["test_correct_direction_fraction"]:.3f} | {r["shuffled_train_label_test_rmse_mm"]:.3f} |\n'
    text+='\n'+'\n'.join('- '+s for s in result['limitations'])+'\n';(OUT/'RESULTS.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    ax.bar([r['feature'] for r in reports],[r['test_coordinate_rmse_mm'] for r in reports])
    ax.axhline(result['constant_test_rmse_mm'],color='red',linestyle='--',label='Training-mean control')
    ax.set_ylabel('Held-out displacement coordinate RMSE (mm)');ax.tick_params(axis='x',rotation=65)
    ax.set_title('Fixed robot state, new held-out scenes, same inference noise');ax.legend()
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    save_json(OUT/'status.json',{'stage':'complete','scenes':20,'samples':result['samples'],'selected_feature':best['feature']})
    print(text,flush=True)

if __name__=='__main__':
    try:
        collect();fit()
    except BaseException as e:
        OUT.mkdir(exist_ok=True);save_json(OUT/'error.json',{'type':type(e).__name__,'error':str(e)});raise
