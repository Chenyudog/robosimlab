"""Check paired object displacement and replay repeatability; does not call a policy."""
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ROOT=Path(__file__).resolve().parents[1]


def capture(env):
    controllers=[]
    for robot in env.robots:
        ctrl=robot.controller
        values={k:copy.deepcopy(v) for k,v in vars(ctrl).items()
                if isinstance(v,(np.ndarray,int,float,bool,str,type(None)))}
        controllers.append((values,copy.deepcopy(robot.gripper.current_action)))
    return {'sim':env.get_sim_state().copy(),'controllers':controllers,
            'clock':{k:copy.deepcopy(getattr(env.env,k)) for k in ['timestep','cur_time','done'] if hasattr(env.env,k)}}


def restore(env,snapshot):
    env.set_state(snapshot['sim'])
    env.sim.forward()
    for robot,(values,grip) in zip(env.robots,snapshot['controllers'],strict=True):
        for k,v in values.items():setattr(robot.controller,k,copy.deepcopy(v))
        robot.gripper.current_action=copy.deepcopy(grip)
    for k,v in snapshot['clock'].items():setattr(env.env,k,copy.deepcopy(v))
    env._post_process();env._update_observables(force=True)
    return env.env._get_observations()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--object-name',default='akita_black_bowl_1',help='Task0 BDDL goal object; evaluation-only label')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    suite=benchmark.get_benchmark_dict()['libero_spatial']();task=suite.get_task(0)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                           camera_heights=128,camera_widths=128)
    try:
        env.seed(7);env.reset()
        states=torch.load(Path(get_libero_path('init_states'))/task.problem_folder/Path(task.init_states_file).name,weights_only=False)
        env.set_init_state(states[0])
        for _ in range(10):env.step([0.]*6+[-1.])
        snap=capture(env)
        inventory={name:{'joints':list(obj.joints)} for name,obj in env.env.objects_dict.items()}
        name=args.object_name
        if name not in inventory or name not in env.obj_of_interest:
            raise RuntimeError(f'Invalid task object: {name}; {inventory}')
        joint=inventory[name]['joints'][0]
        addr=env.sim.model.get_joint_qpos_addr(joint)
        if not isinstance(addr,tuple) or addr[1]-addr[0]!=7:raise RuntimeError('Expected free joint')
        start,end=addr
        body=env.env.obj_body_id[name]
        results={};base_qpos=None;base_pos=None
        for label,delta in [('nominal',0.),('minus',-.02),('plus',.02)]:
            obs=restore(env,snap)
            q=env.sim.data.qpos.copy();q[start]+=delta
            env.sim.data.qpos[:]=q;env.sim.forward()
            env._update_observables(force=True);obs=env.env._get_observations()
            xyz=env.sim.data.body_xpos[body].copy()
            if label=='nominal':base_qpos=q.copy();base_pos=xyz.copy()
            mask=np.ones(len(q),bool);mask[start]=False
            assert np.allclose(q[mask],base_qpos[mask],atol=0,rtol=0)
            assert np.allclose(xyz-base_pos,[delta,0,0],atol=1e-10,rtol=0)
            img=np.ascontiguousarray(obs['agentview_image'][::-1,::-1])
            Image.fromarray(img).save(args.output/f'{label}.png')
            results[label]={'requested_dx_m':delta,'body_xyz':xyz.tolist()}
        tracks=[];renderings=[]
        for repeat in range(2):
            obs=restore(env,snap)
            renderings.append(obs['agentview_image'].copy())
            trajectory=[]
            for _ in range(5):
                env.step([0.]*6+[-1.]);trajectory.append(env.get_sim_state().copy())
            tracks.append(np.stack(trajectory))
        maxerr=float(np.max(np.abs(tracks[0]-tracks[1])))
        pixelerr=int(np.max(np.abs(renderings[0].astype(int)-renderings[1].astype(int))))
        passed=maxerr<1e-9 and pixelerr==0
        report={'status':'ok' if passed else 'failed','task':task.language,'target':name,'object_inventory':inventory,
                'branches':results,'repeated_5_step_state_max_abs_error':maxerr,'repeated_initial_image_max_error':pixelerr,
                'scope':'Environment-only replay check at one initial snapshot; not policy feedback or general snapshot correctness.'}
        (args.output/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        np.savez_compressed(args.output/'replay.npz',first=tracks[0],second=tracks[1],snapshot=snap['sim'])
        print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
        if not passed:raise RuntimeError('Replay repeatability check failed')
    finally:env.close()

if __name__=='__main__':main()
