"""Small offline inference audit on actual demonstration observations."""
from pathlib import Path
import json,time
import io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
from run_baseline import Policy,ROOT,save_json


def aa_quat(v):
    v=np.asarray(v);angle=np.linalg.norm(v)
    return np.r_[v/angle*np.sin(angle/2),np.cos(angle/2)] if angle>1e-10 else np.array([0,0,0,1.])


def main():
    torch.set_num_threads(4)
    policy=Policy(ROOT/'models/pi0_libero')
    d=ROOT/'data/libero_pi'
    episode=1272
    table=pq.read_table(d/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet')
    text='pick up the black bowl between the plate and the ramekin and place it on the plate'
    rows=[]
    with torch.inference_mode():
        for index in [0,20,40,60]:
            frame=table.slice(index,1).to_pylist()[0]
            s=np.array(frame['state'])
            obs={'robot0_eef_pos':s[:3],'robot0_eef_quat':aa_quat(s[3:6]),'robot0_gripper_qpos':s[6:8]}
            for name,key in [('agentview_image','image'),('robot0_eye_in_hand_image','wrist_image')]:
                obs[name]=np.array(Image.open(io.BytesIO(frame[key]['bytes'])))[::-1,::-1].copy()
            pred=policy.infer(obs,text,7+index)
            target=np.array(table.slice(index,min(5,len(table)-index))['actions'].to_pylist())
            error=np.mean(np.abs(pred[:len(target)]-target),axis=0)
            row={'frame':index,'prediction_first':pred[0].tolist(),'demonstration_first':target[0].tolist(),'mean_abs_error_first5':error.tolist()}
            rows.append(row);print(json.dumps(row),flush=True)
    save_json(ROOT/'outputs/offline_inference_audit.json',{'episode':episode,'task':text,'frames':rows,
        'note':'Stochastic inference vs a single demonstration is diagnostic, not a success benchmark.'})

if __name__=='__main__':main()
