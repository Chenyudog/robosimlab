"""Descriptive matched-branch plots; no claim of internal causal mechanism."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    root=args.directory
    summary=json.loads((root/'summary.json').read_text());meta=json.loads((root/'manifest.json').read_text())
    if summary['status']!='complete':raise RuntimeError('Paired run incomplete')
    valid={};records=[]
    for item in summary['branches']:
        if item['status']!='complete':continue
        branch=item['branch'];a=np.load(root/branch/'trajectory.npz')
        distance=np.linalg.norm(a['eef_xyz']-a['target_xyz'],axis=1)
        valid[branch]={k:a[k] for k in a.files};valid[branch]['distance']=distance
        records.append({'branch':branch,'success':item['success'],'initial_eef_target_distance_m':float(distance[0]),
            'minimum_eef_target_distance_m':float(distance.min()),'final_eef_target_distance_m':float(distance[-1])})
    metrics={'branches':records,'interpretation':'EEF-target distance is descriptive; it is not a task-independent progress or feedback correctness metric.'}
    if {'plus_x','minus_x'}.issubset(valid):
        p=valid['plus_x'];m=valid['minus_x'];delta=meta['requested_displacement_m']
        metrics['first_action_difference_per_metre_of_object_displacement']=((p['actions'][0]-m['actions'][0])/(2*delta)).tolist()
        if len(p['eef_xyz'])>1 and len(m['eef_xyz'])>1:
            metrics['first_step_eef_response_per_metre']=((p['eef_xyz'][1]-p['eef_xyz'][0]-m['eef_xyz'][1]+m['eef_xyz'][0])/(2*delta)).tolist()
    (root/'response_metrics.json').write_text(json.dumps(metrics,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for name,a in valid.items():
        axs[0].plot(a['distance'],label=name)
        axs[1].plot(a['actions'][:,0],label=name)
        line=axs[2].plot(a['eef_xyz'][:,0],label=f'{name}: EEF')[0]
        axs[2].plot(a['target_xyz'][:,0],linestyle='--',color=line.get_color(),alpha=.7,label=f'{name}: target')
    for ax in axs:ax.set_xlabel('Control step after intervention');ax.grid(alpha=.2)
    axs[0].set_ylabel('EEF-target distance (m)');axs[0].legend()
    axs[1].set_ylabel('x action (controller input units)')
    axs[2].set_ylabel('World x position (m)');axs[2].legend(fontsize=7)
    fig.suptitle(f'Exploratory paired perturbation pilot: one snapshot, replan every {meta["replan_steps"]} control steps')
    fig.savefig(root/'response_comparison.png',dpi=180)
    plt.close(fig)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
