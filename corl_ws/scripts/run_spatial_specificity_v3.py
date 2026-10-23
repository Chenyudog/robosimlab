"""Spatial/token-position probes and distractor shifts; fresh scenes40..49 test."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.pop('VLA_VISION_CPU',None)
import copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
from run_baseline import ROOT,Policy,save_json,benchmark,get_libero_path,OffScreenRenderEnv
from check_counterfactual import capture,restore
from run_fixed_state_probes import penetrations,image_summary,SHIFTS
OUT=ROOT/'outputs/spatial_specificity_v3'
SPLIT={'train':list(range(20,32)),'validation':list(range(32,36)),'test':list(range(40,50))}
SCENES=sum(SPLIT.values(),[])
LAYERS=(0,5,11,17)
OBJECTS={'target':'akita_black_bowl_1','distractor':'akita_black_bowl_2'}

def progress(stage,**kw):
    save_json(OUT/'status.json',dict(stage=stage,**kw))
    text=(ROOT/'当前进度.md').read_text()
    marker='\n## v3实时状态\n'
    text=text.split(marker)[0]+marker+'\n'+json.dumps(dict(stage=stage,**kw),ensure_ascii=False,indent=2)+'\n'
    tmp=ROOT/'当前进度.tmp';tmp.write_text(text);tmp.replace(ROOT/'当前进度.md')

class Recorder:
    def __init__(self,model):
        self.active=False;self.handles=[]
        for group,layers in [('prefix',model.paligemma_with_expert.paligemma.language_model.layers),('expert',model.paligemma_with_expert.gemma_expert.model.layers)]:
            for idx in LAYERS:self.handles.append(layers[idx].register_forward_hook(self.hook(group,idx)))
    def start(self):self.active=True;self.count={};self.values={}
    def hook(self,group,idx):
        def call(m,a,o):
            if not self.active:return
            key=f'{group}_{idx}';n=self.count.get(key,0);self.count[key]=n+1;x=o[0]
            if group=='prefix':
                if n!=0 or x.shape[1]<512:raise RuntimeError('Unexpected prefix shape')
                grids=[]
                for begin in (0,256):
                    grid=x[:,begin:begin+256].float().reshape(1,4,4,4,4,x.shape[-1]).mean((2,4))
                    grids.append(grid.flatten().cpu().numpy())
                self.values[key+'_grid4']=np.concatenate(grids)
            elif n in (0,9):
                self.values[key+f'_denoise{n}_positions']=x[:,[1,11,26,50]].float().flatten().cpu().numpy()
        return call
    def finish(self):
        self.active=False
        if any(self.count.get(f'{g}_{i}')!=n for g,n in [('prefix',1),('expert',10)] for i in LAYERS):raise RuntimeError('Unexpected forward count')
        return self.values

def collect():
    OUT.mkdir(exist_ok=True)
    protocol=json.loads(json.dumps({'split':SPLIT,'objects':OBJECTS,'layers':LAYERS,'shifts':SHIFTS,
        'prefix_pool':'Two cameras each4x4 spatial cells, mean within each4x4 token cell; retain channels',
        'expert_pool':'Action positions0,10,25,49 (suffix index1,11,26,50), denoise0 and9',
        'vision_execution':'FP32 vision tower/projector on GPU; BF16 language/expert on GPU; vocabulary embedding on CPU',
        'training':'Fit target-shift displacement decoder on target samples only; evaluate on target and distractor shifts',
        'controls':'pixel RGB mean; zero state change; within-scene shuffled training labels',
        'scope':'Exploratory single-task readout. Fresh test scenes40..49; no causal/closed-loop claim.'}))
    p=OUT/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise RuntimeError('Protocol drift')
    save_json(p,protocol)
    # Leave other GPU jobs untouched; wait before loading until sufficient free memory.
    import subprocess,time
    for attempt in range(360):
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free>=6800:break
        progress('等待空闲显存',free_mib=free,required_mib=6800)
        time.sleep(5)
    else:raise RuntimeError('GPU free memory below6800MiB for30minutes; no other process stopped')
    progress('加载模型（视觉与动作专家在GPU执行）')
    torch.set_num_threads(4);policy=Policy(ROOT/'models/pi0_libero_finetuned',vision_fp32=True);rec=Recorder(policy.model)
    suite=benchmark.get_benchmark_dict()['libero_spatial']();task=suite.get_task(0)
    init=torch.load(Path(get_libero_path('init_states'))/task.problem_folder/Path(task.init_states_file).name,weights_only=False)
    if len(init)<50:raise RuntimeError('Need50 initial states')
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
    try:
        for number,i in enumerate(SCENES):
            if all((OUT/f'{kind}_{i:03d}.npz').exists() for kind in OBJECTS):continue
            progress('采集空间激活',initial_state=i,completed_scenes=number,total_scenes=len(SCENES))
            np.random.seed(7);torch.manual_seed(7);env.seed(7);env.reset();obs=env.set_init_state(init[i])
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            env.sim.forward();env._update_observables(force=True);nominal=copy.deepcopy(env.env._get_observations());snap=capture(env)
            robot=env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
            rec.start();baseaction=policy.infer(nominal,task.language,7);base=rec.finish()
            for kind,obj in OBJECTS.items():
                path=OUT/f'{kind}_{i:03d}.npz'
                if path.exists():continue
                restore(env,snap);joint=env.env.objects_dict[obj].joints[0];body=env.env.obj_body_id[obj]
                baseq=env.sim.data.get_joint_qpos(joint).copy();basepos=env.sim.data.body_xpos[body].copy()
                values={k:[] for k in base};labels=[];pixels=[];actions=[];invalid=[];state_deltas=[]
                for axis,amount in SHIFTS:
                    restore(env,snap);q=baseq.copy();q[axis]+=amount;env.sim.data.set_joint_qpos(joint,q);env.sim.forward()
                    expected=np.zeros(3);expected[axis]=amount
                    if not np.allclose(env.sim.data.body_xpos[body]-basepos,expected,atol=1e-10,rtol=0):raise RuntimeError('Incorrect displacement')
                    if not np.array_equal(robot,env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes]):raise RuntimeError('Robot changed')
                    pen=penetrations(env,obj)
                    if pen:invalid.append({'axis':axis,'amount':amount,'reason':'interpenetration','min_contact_dist':min(pen)});continue
                    env._update_observables(force=True);obs=env.env._get_observations()
                    for key in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']:
                        if not np.allclose(obs[key],nominal[key],atol=1e-12,rtol=0):raise RuntimeError(f'State changed:{key}')
                    if np.array_equal(obs['agentview_image'],nominal['agentview_image']):invalid.append({'axis':axis,'amount':amount,'reason':'no_visible_change'});continue
                    rec.start();action=policy.infer(obs,task.language,7);features=rec.finish()
                    for key in base:values[key].append(features[key]-base[key])
                    labels.append(expected[:2]);pixels.append(image_summary(obs)-image_summary(nominal));actions.append(action-baseaction)
                    state_deltas.append(np.concatenate([obs[key]-nominal[key] for key in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']]))
                save_json(OUT/f'{kind}_{i:03d}_validity.json',{'scene':i,'object':obj,'valid':len(labels),'invalid':invalid})
                if not labels:raise RuntimeError(f'No valid shifts:{i}/{kind}')
                np.savez_compressed(path,**{k:np.stack(v) for k,v in values.items()},target=np.stack(labels),pixel_mean=pixels,state_delta=state_deltas,action_delta=actions)
                print('SPATIAL',i,kind,'valid',len(labels),'invalid',len(invalid),flush=True)
    finally:
        env.close()
        for h in rec.handles:h.remove()


def fit():
    progress('拟合与评估探针')
    target={i:np.load(OUT/f'target_{i:03d}.npz') for i in SCENES}
    distractor={i:np.load(OUT/f'distractor_{i:03d}.npz') for i in SPLIT['test']}
    labels={s:np.concatenate([target[i]['target'] for i in ids]) for s,ids in SPLIT.items()}
    keys=[k for k in target[20].files if k.startswith(('prefix_','expert_'))]+['pixel_mean','state_delta']
    constant=labels['train'].mean(0);mse=lambda a,b:float(np.mean((a-b)**2));constant_mse=mse(labels['test'],constant)
    reports=[];predictions={}
    for key in keys:
        xs={s:np.concatenate([target[i][key] for i in ids]).astype(float) for s,ids in SPLIT.items()}
        xd=np.concatenate([distractor[i][key] for i in SPLIT['test']]).astype(float)
        mean=xs['train'].mean(0);std=xs['train'].std(0);std[std<1e-6]=1
        xs={s:(a-mean)/std for s,a in xs.items()};xd=(xd-mean)/std;x=xs['train'];gram=x@x.T;y=labels['train']-constant
        trials=[]
        for alpha in (.1,1.,10.,100.,1000.):
            dual=np.linalg.solve(gram+alpha*np.eye(len(x)),y);trials.append((mse(xs['validation']@x.T@dual+constant,labels['validation']),alpha,dual))
        val,alpha,dual=min(trials,key=lambda a:a[0]);pred=xs['test']@x.T@dual+constant;pd=xd@x.T@dual+constant;predictions[key]=pred
        rng=np.random.default_rng(7);shuffled=np.concatenate([rng.permutation(target[i]['target']) for i in SPLIT['train']])-constant
        shuffled_dual=np.linalg.solve(gram+alpha*np.eye(len(x)),shuffled)
        r={'feature':key,'alpha':alpha,'validation_mse_m2':val,'target_test_rmse_mm':float(np.sqrt(mse(pred,labels['test']))*1000),
           'target_test_r2':1-mse(pred,labels['test'])/constant_mse,'target_direction_fraction':float((np.sum(pred*labels['test'],axis=1)>0).mean()),
           'distractor_decoded_target_rmse_to_zero_mm':float(np.sqrt(np.mean(pd**2))*1000),
           'shuffled_test_rmse_mm':float(np.sqrt(mse(xs['test']@x.T@shuffled_dual+constant,labels['test']))*1000)}
        # Matched scene/shift masks prevent unequal collision filtering from defining the comparison.
        target_norm=[];distractor_norm=[];offset_t=offset_d=0
        for i in SPLIT['test']:
            tlabels=target[i]['target'];dlabels=distractor[i]['target']
            for j,label in enumerate(tlabels):
                matches=np.flatnonzero(np.all(np.isclose(dlabels,label,atol=1e-12,rtol=0),axis=1))
                if len(matches)==1:
                    target_norm.append(float(np.linalg.norm(pred[offset_t+j])))
                    distractor_norm.append(float(np.linalg.norm(pd[offset_d+matches[0]])))
            offset_t+=len(tlabels);offset_d+=len(dlabels)
        r['matched_samples']=len(target_norm);r['matched_mean_target_decoded_norm_mm']=float(np.mean(target_norm)*1000)
        r['matched_mean_distractor_decoded_norm_mm']=float(np.mean(distractor_norm)*1000)
        reports.append(r);print('PROBE',r,flush=True)
    best=min((r for r in reports if r['feature'].startswith(('prefix_','expert_'))),key=lambda r:r['validation_mse_m2'])
    result={'split':SPLIT,'samples':{s:len(y) for s,y in labels.items()},'distractor_test_samples':sum(len(distractor[i]['target']) for i in SPLIT['test']),
        'selected_by_validation':best['feature'],'constant_test_rmse_mm':float(np.sqrt(constant_mse)*1000),'probes':reports,
        'limitations':['Single task; spatial pooling still loses information.',
        'A displacement readout is not evidence of causal use or semantic features.',
        'Distractor differs in spatial position; lower readout may reflect a spatial shortcut, not language goal binding.',
        'Target/distractor matching excludes invalid shifts; no statistical significance claim or closed-loop success measurement.']}
    save_json(OUT/'results.json',result);np.savez_compressed(OUT/'test_predictions.npz',target=labels['test'],**predictions)
    text='# Spatial specificity v3\n\n'+f'Target samples {result["samples"]}; distractor test {result["distractor_test_samples"]}; validation-selected {best["feature"]}. Constant RMSE {result["constant_test_rmse_mm"]:.3f}mm.\n\n'
    text+='| Feature | Target test RMSE(mm) | R2 | Shuffled RMSE(mm) | Matched target norm(mm) | Matched distractor norm(mm) |\n|---|---:|---:|---:|---:|---:|\n'
    for r in reports:text+=f'| {r["feature"]} | {r["target_test_rmse_mm"]:.3f} | {r["target_test_r2"]:.3f} | {r["shuffled_test_rmse_mm"]:.3f} | {r["matched_mean_target_decoded_norm_mm"]:.3f} | {r["matched_mean_distractor_decoded_norm_mm"]:.3f} |\n'
    text+='\n'+'\n'.join('- '+s for s in result['limitations'])+'\n';(OUT/'RESULTS.md').write_text(text)
    progress('实验完成',selected_feature=best['feature'],target_samples=result['samples'])

if __name__=='__main__':
    try:collect();fit()
    except BaseException as e:
        OUT.mkdir(exist_ok=True);save_json(OUT/'error.json',{'type':type(e).__name__,'error':str(e)})
        progress('实验受阻',error=str(e));raise
