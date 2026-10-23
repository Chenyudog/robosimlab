"""Paired fresh-observation rollouts from one successful baseline snapshot."""
from pathlib import Path
import json,time
import numpy as np
import torch
import imageio.v2 as imageio
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from check_counterfactual import capture,restore
from run_baseline import save_json


def run_pairs(policy,baseline_output,output,max_steps=220,delta=.02,replan_steps=1):
    if not 1 <= replan_steps <= 50:raise ValueError("replan_steps must be 1..50")
    baseline_output=Path(baseline_output);output=Path(output)
    meta=json.loads((baseline_output/'manifest.json').read_text())
    result=json.loads((baseline_output/'result.json').read_text())
    if not result['success']:raise RuntimeError('Ordinary-control baseline did not succeed')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    suite=benchmark.get_benchmark_dict()[meta['suite']]();task=suite.get_task(meta['task_id'])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                           camera_heights=256,camera_widths=256)
    try:
        env.seed(meta['seed']);env.reset()
        snapshot=torch.load(baseline_output/'snapshot.pt',weights_only=False)
        obs=restore(env,snapshot)
        # This pilot is deliberately limited to the BDDL-identified task0 target.
        target='akita_black_bowl_1'
        if meta['task_id']!=0 or target not in env.obj_of_interest:raise RuntimeError('Explicit task0 target required')
        body=env.env.obj_body_id[target]
        # Validate replay at this actual mid-episode snapshot, not only at reset.
        repeats=[];images=[]
        for _ in range(2):
            replay_obs=restore(env,snapshot)
            images.append(replay_obs['agentview_image'].copy())
            states=[]
            for _ in range(3):
                env.step([0.]*6+[-1.]);states.append(env.get_sim_state().copy())
            repeats.append(np.stack(states))
        replay_error=float(np.max(np.abs(repeats[0]-repeats[1])))
        if replay_error>1e-9 or not np.array_equal(images[0],images[1]):
            raise RuntimeError(f'Mid-episode replay not reproducible: {replay_error}')
        obs=restore(env,snapshot)
        gap=float(np.linalg.norm(obs['robot0_eef_pos']-env.sim.data.body_xpos[body]))
        if gap<.08:raise RuntimeError('Snapshot too close to contact for this free-space displacement pilot')
        joint=env.env.objects_dict[target].joints[0];addr=env.sim.model.get_joint_qpos_addr(joint)
        if not isinstance(addr,tuple) or addr[1]-addr[0]!=7:raise RuntimeError('Expected free joint')
        robot_qpos=env.robots[0]._ref_joint_pos_indexes
        initial_robot=env.sim.data.qpos[robot_qpos].copy()
        summaries=[]
        save_json(output/'manifest.json',{'baseline_output':str(baseline_output),'snapshot_step':meta['snapshot_step'],
            'target':target,'requested_displacement_m':delta,'replan_steps':replan_steps,'max_steps':max_steps,
            'simulation_pauses_during_inference':True,'same_noise_schedule_all_branches':True,
            'task':task.language,'initial_eef_target_distance_m':gap,'mid_snapshot_replay_max_error':replay_error,
            'scope':'Exploratory three-branch pilot; no mechanistic or generalization conclusion.'})
        for label,shift in [('nominal',0.),('minus_x',-delta),('plus_x',delta)]:
            branch=output/label;branch.mkdir()
            obs=restore(env,snapshot)
            base_pos=env.sim.data.body_xpos[body].copy()
            q=env.sim.data.get_joint_qpos(joint).copy();q[0]+=shift
            env.sim.data.set_joint_qpos(joint,q);env.sim.forward();env._update_observables(force=True)
            obs=env.env._get_observations()
            actual=env.sim.data.body_xpos[body].copy()-base_pos
            assert np.allclose(actual,[shift,0,0],atol=1e-10,rtol=0)
            assert np.array_equal(env.sim.data.qpos[robot_qpos],initial_robot)
            # Reject introduced interpenetration with any other movable object.
            movable=set(env.env.obj_body_id[n] for n in env.env.objects_dict if n!=target)
            invalid=[]
            for c in env.sim.data.contact[:env.sim.data.ncon]:
                a=int(env.sim.model.geom_bodyid[c.geom1]);b=int(env.sim.model.geom_bodyid[c.geom2])
                if ((a==body and b in movable) or (b==body and a in movable)) and c.dist < -1e-4:
                    invalid.append(float(c.dist))
            if invalid:
                record={'branch':label,'status':'invalid_initial_interpenetration','penetration_distances':invalid}
                save_json(branch/'result.json',record);summaries.append(record);continue
            actions=[];positions=[];target_positions=[];done=False
            started=time.monotonic();writer=imageio.get_writer(str(branch/'rollout.mp4'),fps=20)
            try:
                for k in range(max_steps):
                    writer.append_data(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]))
                    positions.append(obs['robot0_eef_pos'].copy());target_positions.append(env.sim.data.body_xpos[body].copy())
                    # Identical declared execution horizon and noise schedule across branches.
                    if k % replan_steps == 0:
                        chunk=policy.infer(obs,task.language,meta['seed']+meta['snapshot_step']+k)
                    action=np.clip(chunk[k % replan_steps],-1,1);actions.append(action)
                    obs,_,done,_=env.step(action.tolist())
                    if k%20==0 or done:print(f'PAIR {label} step={k+1} success={bool(env.check_success())}',flush=True)
                    if done:break
                record={'branch':label,'status':'complete','success':bool(env.check_success()),'steps':len(actions),
                        'actual_shift_m':actual.tolist(),'elapsed_seconds':time.monotonic()-started}
                save_json(branch/'result.json',record)
                np.savez_compressed(branch/'trajectory.npz',actions=actions,eef_xyz=positions,target_xyz=target_positions)
                summaries.append(record)
            finally:writer.close()
            save_json(output/'summary.json',{'status':'running','branches':summaries})
        save_json(output/'summary.json',{'status':'complete','branches':summaries,
            'note':f'One snapshot; paired replan{replan_steps}, baseline replan{meta["replan_steps"]}. Paired branches start a fresh chunk at the snapshot; compare branches within this run.'})
        return summaries
    finally:env.close()
