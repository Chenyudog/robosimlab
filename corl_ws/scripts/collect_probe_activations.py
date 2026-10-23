"""Natural h50 planning-boundary activations; no counterfactual snapshot restart."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from collections import deque
from run_baseline import Policy,ROOT,save_json,benchmark,get_libero_path,OffScreenRenderEnv,quat_axis

OUT=ROOT/'outputs/probe_pilot_v1'
LAYERS=(0,5,11,17)

class Recorder:
    def __init__(self,model):
        self.active=False;self.handles=[]
        groups={'prefix':model.paligemma_with_expert.paligemma.language_model.layers,
                'expert':model.paligemma_with_expert.gemma_expert.model.layers}
        for group,layers in groups.items():
            for i in LAYERS:
                self.handles.append(layers[i].register_forward_hook(self.hook(group,i)))
    def start(self):self.counts={};self.features={};self.active=True
    def hook(self,group,i):
        def callback(module,args,output):
            if not self.active:return
            key=f'{group}_{i}';n=self.counts.get(key,0);self.counts[key]=n+1
            x=output[0] if isinstance(output,tuple) else output
            if group=='prefix':
                # Each 224px SigLIP camera contributes 256 tokens; exclude language padding.
                if n!=0 or x.shape[1]<512:raise RuntimeError(f'Unexpected prefix layout {x.shape}')
                self.features[key]=torch.cat([x[:,:256].float().mean(1),x[:,256:512].float().mean(1)],-1)[0].cpu().numpy()
            elif n in (0,9):
                self.features[f'{key}_denoise{n}']=x[:,1:].float().mean(1)[0].cpu().numpy()
        return callback
    def finish(self):
        self.active=False
        for i in LAYERS:
            if self.counts.get(f'prefix_{i}')!=1 or self.counts.get(f'expert_{i}')!=10:
                raise RuntimeError(f'Unexpected hook counts: {self.counts}')
        return self.features

def patch_smoke(policy,obs,text,output):
    """First-denoise expert block11; identity must preserve outputs. No repair claim."""
    layer=policy.model.paligemma_with_expert.gemma_expert.model.layers[11]
    saved=[];calls=[0]
    def grab(m,a,o):
        if calls[0]==0:saved.append(o[0].detach().clone())
        calls[0]+=1
    h=layer.register_forward_hook(grab)
    try:original=policy.infer(obs,text,7)
    finally:h.remove()
    results={};arrays={'original':original}
    for mode in ['identity','zero']:
        calls[0]=0
        def replace(m,a,o):
            n=calls[0];calls[0]+=1
            if n!=0:return o
            value=saved[0] if mode=='identity' else torch.zeros_like(o[0])
            return (value,)+o[1:]
        h=layer.register_forward_hook(replace)
        try:action=policy.infer(obs,text,7)
        finally:h.remove()
        arrays[mode]=action
        results[mode]={'max_abs_action_change':float(np.max(np.abs(action-original))),
                       'first_action_l2_change':float(np.linalg.norm(action[0]-original[0]))}
    if results['identity']['max_abs_action_change']>1e-5:raise RuntimeError('Identity patch changed actions')
    save_json(output/'patch_smoke.json',{'layer':11,'denoise_step':0,'seed':7,'results':results,
        'scope':'Technical identity and zero-ablation smoke only. No donor patch, closed-loop or mechanistic conclusion.'})
    np.savez_compressed(output/'patch_smoke_actions.npz',**arrays)

def main():
    OUT.mkdir(exist_ok=True)
    protocol={'initial_states':list(range(20)),'task_id':0,'suite':'libero_spatial','seed':7,
        'replan_steps':50,'max_steps':220,'layers_zero_based':list(LAYERS),
        'prefix_pool':'separate camera token means, each256 tokens; language excluded',
        'expert_pool':'mean50 action tokens, exclude state token; denoise indices0 and9',
        'target':'target body world xyz minus eef world xyz in metres; evaluation label only',
        'split':{'train':list(range(12)),'validation':list(range(12,16)),'test':list(range(16,20))},
        'scope':'One task, coarse pooled features, exploratory small sample; not mechanism proof.'}
    f=OUT/'protocol.json'
    if f.exists() and json.loads(f.read_text())!=protocol:raise RuntimeError('Protocol drift')
    save_json(f,protocol)
    save_json(OUT/'status.json',{'stage':'loading_model'})
    torch.set_num_threads(4)
    policy=Policy(ROOT/'models/pi0_libero_finetuned',vision_fp32=True)
    recorder=Recorder(policy.model)
    suite=benchmark.get_benchmark_dict()['libero_spatial']();task=suite.get_task(0)
    init=torch.load(Path(get_libero_path('init_states'))/task.problem_folder/Path(task.init_states_file).name,weights_only=False)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
    try:
        for initial in range(20):
            output=OUT/f'episode_{initial:03d}.npz'
            if output.exists():continue
            save_json(OUT/'status.json',{'stage':'collecting','initial_state':initial})
            np.random.seed(7);torch.manual_seed(7);env.seed(7);env.reset();obs=env.set_init_state(init[initial])
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            if initial==0:patch_smoke(policy,obs,task.language,OUT)
            body=env.env.obj_body_id['akita_black_bowl_1']
            features={};labels=[];states=[];steps=[];actions=[];queue=deque()
            for step in range(220):
                if not queue:
                    labels.append(env.sim.data.body_xpos[body].copy()-obs['robot0_eef_pos'])
                    states.append(np.concatenate([obs['robot0_eef_pos'],quat_axis(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']]))
                    steps.append(step);recorder.start()
                    chunk=policy.infer(obs,task.language,7+step)
                    for name,value in recorder.finish().items():features.setdefault(name,[]).append(value)
                    queue.extend(chunk[:50])
                action=np.clip(queue.popleft(),-1,1);actions.append(action)
                obs,_,done,_=env.step(action.tolist())
                if done:break
            np.savez_compressed(output,**{k:np.stack(v) for k,v in features.items()},target=np.stack(labels),
                state=np.stack(states),control_step=steps,actions=actions,initial_state=initial,success=bool(env.check_success()))
            print('COLLECT',initial,'samples',len(labels),'success',bool(env.check_success()),flush=True)
        save_json(OUT/'status.json',{'stage':'collection_complete','episodes':20})
    finally:
        env.close()
        for h in recorder.handles:h.remove()

if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(exist_ok=True);save_json(OUT/'error.json',{'type':type(e).__name__,'error':str(e)});raise
