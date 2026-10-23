"""Validation-selected block patch on held-out initial observations; no repair claim."""
import copy,json
import numpy as np
import torch
from run_baseline import ROOT,Policy,save_json,benchmark,get_libero_path,OffScreenRenderEnv
from check_counterfactual import capture,restore
OUT=ROOT/'outputs/activation_patch_v1'

def forward_capture(policy,layer,index,obs,text):
    calls=[0];saved=[]
    def hook(m,a,o):
        if calls[0]==index:saved.append(o[0].detach().clone())
        calls[0]+=1
    h=layer.register_forward_hook(hook)
    try:action=policy.infer(obs,text,7)
    finally:h.remove()
    if len(saved)!=1:raise RuntimeError('Capture hook not called at expected index')
    return action,saved[0]

def forward_patch(policy,layer,index,value,obs,text):
    calls=[0]
    def hook(m,a,o):
        n=calls[0];calls[0]+=1
        if n!=index:return o
        if o[0].shape!=value.shape:raise RuntimeError('Patch shape mismatch')
        return (value,)+o[1:]
    h=layer.register_forward_hook(hook)
    try:return policy.infer(obs,text,7)
    finally:h.remove()

def main():
    OUT.mkdir(exist_ok=True)
    results=json.loads((ROOT/'outputs/probe_pilot_v1/probe_results.json').read_text())
    selected=min((r for r in results['probes'] if r['feature']!='state'),key=lambda r:r['validation_mse_m2'])
    key=selected['feature'];parts=key.split('_');group=parts[0];idx=int(parts[1]);call=int(parts[2].removeprefix('denoise')) if group=='expert' else 0
    save_json(OUT/'protocol.json',{'selected_feature':key,'selection':'minimum validation MSE only',
        'initial_states':[16,17,18,19],'shift_m':.005,'seed':7,
        'patch':'whole selected block output at a single forward call; same recipient input and noise',
        'conditions':['identity','shifted_observation_donor','channel_permuted_donor'],
        'scope':'Single-observation action sensitivity. Not closed-loop recovery, natural circuit, or task success.'})
    save_json(OUT/'status.json',{'stage':'loading_model'})
    torch.set_num_threads(4);policy=Policy(ROOT/'models/pi0_libero_finetuned',vision_fp32=True)
    layers=policy.model.paligemma_with_expert.gemma_expert.model.layers if group=='expert' else policy.model.paligemma_with_expert.paligemma.language_model.layers
    layer=layers[idx]
    suite=benchmark.get_benchmark_dict()['libero_spatial']();task=suite.get_task(0)
    initial_states=torch.load(get_libero_path('init_states')+'/'+task.problem_folder+'/'+task.init_states_file.split('/')[-1],weights_only=False)
    env=OffScreenRenderEnv(bddl_file_name=get_libero_path('bddl_files')+'/'+task.problem_folder+'/'+task.bddl_file,camera_heights=256,camera_widths=256)
    reports=[]
    try:
        for i in (16,17,18,19):
            save_json(OUT/'status.json',{'stage':'patching','initial_state':i})
            env.seed(7);env.reset();obs=env.set_init_state(initial_states[i])
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            nominal=copy.deepcopy(obs);snapshot=capture(env)
            target='akita_black_bowl_1';body=env.env.obj_body_id[target];joint=env.env.objects_dict[target].joints[0]
            original_xyz=env.sim.data.body_xpos[body].copy();robot_qpos=env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
            q=env.sim.data.get_joint_qpos(joint).copy();q[0]+=.005;env.sim.data.set_joint_qpos(joint,q);env.sim.forward()
            actual=env.sim.data.body_xpos[body]-original_xyz
            if not np.allclose(actual,[.005,0,0],atol=1e-10,rtol=0):raise RuntimeError('Incorrect displacement')
            if not np.array_equal(robot_qpos,env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes]):raise RuntimeError('Robot changed')
            movable=set(env.env.obj_body_id[n] for n in env.env.objects_dict if n!=target)
            penetration=[]
            for c in env.sim.data.contact[:env.sim.data.ncon]:
                a=int(env.sim.model.geom_bodyid[c.geom1]);b=int(env.sim.model.geom_bodyid[c.geom2])
                if ((a==body and b in movable) or (b==body and a in movable)) and c.dist < -1e-4:penetration.append(float(c.dist))
            if penetration:
                reports.append({'initial_state':i,'status':'invalid_interpenetration','penetration_distances':penetration});continue
            env._update_observables(force=True);shifted=copy.deepcopy(env.env._get_observations());restore(env,snapshot)
            base,own=forward_capture(policy,layer,call,nominal,task.language)
            donor_action,donor=forward_capture(policy,layer,call,shifted,task.language)
            if float(np.max(np.abs(nominal['agentview_image'].astype(float)-shifted['agentview_image'])))==0:raise RuntimeError('Camera did not reflect displacement')
            generator=torch.Generator(device=donor.device).manual_seed(7)
            permutation=torch.randperm(donor.shape[-1],generator=generator,device=donor.device)
            arrays={'base':base,'shifted_input':donor_action};conditions={}
            for name,value in [('identity',own),('shifted_observation_donor',donor),('channel_permuted_donor',donor[...,permutation])]:
                action=forward_patch(policy,layer,call,value,nominal,task.language);arrays[name]=action
                conditions[name]={'first_action_delta':(action[0]-base[0]).tolist(),
                    'first_action_l2':float(np.linalg.norm(action[0]-base[0])),
                    'chunk_l2':float(np.linalg.norm(action-base))}
            if conditions['identity']['chunk_l2']>1e-4:raise RuntimeError('Identity patch changed actions')
            record={'initial_state':i,'status':'complete','selected_feature':key,'conditions':conditions,
                'direct_shift_first_action_delta':(donor_action[0]-base[0]).tolist()}
            reports.append(record);np.savez_compressed(OUT/f'initial{i}_actions.npz',**arrays)
            save_json(OUT/'results.json',{'episodes':reports,'scope':'Action sensitivity only; whole-block replacement can affect many concepts.'})
            print('PATCH',i,conditions,flush=True)
        save_json(OUT/'results.json',{'episodes':reports,'scope':'Action sensitivity only; channel permutation is an out-of-distribution control, not proof of specificity.'})
        save_json(OUT/'status.json',{'stage':'complete','valid':sum(r['status']=='complete' for r in reports),'attempted':4})
    finally:env.close()
if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(exist_ok=True);save_json(OUT/'error.json',{'error':str(e),'type':type(e).__name__});raise
