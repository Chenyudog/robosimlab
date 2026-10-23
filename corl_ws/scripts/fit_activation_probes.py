"""Trajectory-held-out ridge probes, with state/constant/shuffled-label controls."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/probe_pilot_v1'

def main():
    (OUT/'status.json').write_text(json.dumps({'stage':'fitting_probes'}))
    protocol=json.loads((OUT/'protocol.json').read_text())
    episodes={i:np.load(OUT/f'episode_{i:03d}.npz') for i in range(20)}
    split=protocol['split'];ys={s:np.concatenate([episodes[i]['target'] for i in ids]) for s,ids in split.items()}
    keys=[k for k in episodes[0].files if k.startswith(('prefix_','expert_'))]+['state']
    def mse(a,b):return float(np.mean((a-b)**2))
    constant=np.broadcast_to(ys['train'].mean(0),ys['test'].shape)
    reports=[]
    for key in keys:
        xs={s:np.concatenate([episodes[i][key] for i in ids]).astype(np.float64) for s,ids in split.items()}
        mean=xs['train'].mean(0);std=xs['train'].std(0);std[std<1e-6]=1
        xs={s:(x-mean)/std for s,x in xs.items()}
        ymean=ys['train'].mean(0);y=ys['train']-ymean;x=xs['train']
        gram=x@x.T;alphas=[.1,1.,10.,100.,1000.]
        fits=[]
        for alpha in alphas:
            dual=np.linalg.solve(gram+alpha*np.eye(len(x)),y)
            fits.append((mse(xs['validation']@x.T@dual+ymean,ys['validation']),alpha,dual))
        val,alpha,dual=min(fits,key=lambda a:a[0])
        pred=xs['test']@x.T@dual+ymean
        # Shuffle whole training-trajectory target sequences where their lengths match.
        # Simpler independent training-row permutation is only a weak negative control.
        shuffled=np.random.default_rng(7).permutation(y)
        shuffled_dual=np.linalg.solve(gram+alpha*np.eye(len(x)),shuffled)
        test_mse=mse(pred,ys['test']);base=mse(constant,ys['test'])
        reports.append({'feature':key,'alpha':alpha,'validation_mse_m2':val,'test_mse_m2':test_mse,
            'test_rmse_m':float(np.sqrt(test_mse)),'test_axis_rmse_m':np.sqrt(np.mean((pred-ys['test'])**2,0)).tolist(),
            'test_r2_vs_training_mean':1-test_mse/base,
            'shuffled_train_rows_test_mse_m2':mse(xs['test']@x.T@shuffled_dual+ymean,ys['test'])})
    d={'split':split,'samples':{k:len(v) for k,v in ys.items()},'constant_test_mse_m2':mse(constant,ys['test']),
       'probes':reports,'limitations':['One task and four held-out trajectories; no significance claim.',
       'Layer selection by validation only; test is descriptive, not for subsequent tuning.',
       'Decodability is not causal use; state baseline may explain phase correlations.',
       'Pooled activations can miss spatial information; shuffled rows are a weak negative control.']}
    (OUT/'probe_results.json').write_text(json.dumps(d,indent=2))
    selected=min((r for r in reports if r['feature']!='state'),key=lambda r:r['validation_mse_m2'])
    text='# Activation probe pilot\n\n'+f'Samples: {d["samples"]}. Validation-selected activation: {selected["feature"]}.\n\n'
    text+='| Feature | Validation MSE | Test RMSE (m) | Test R2 vs train mean |\n|---|---:|---:|---:|\n'
    for r in reports:text+=f'| {r["feature"]} | {r["validation_mse_m2"]:.6f} | {r["test_rmse_m"]:.4f} | {r["test_r2_vs_training_mean"]:.3f} |\n'
    text+='\n'+'\n'.join('- '+x for x in d['limitations'])+'\n'
    (OUT/'PROBE_RESULTS.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    ax.bar([r['feature'] for r in reports],[r['test_rmse_m'] for r in reports])
    ax.axhline(np.sqrt(d['constant_test_mse_m2']),color='red',linestyle='--',label='Training-mean baseline')
    ax.set_ylabel('Held-out coordinate RMSE (m)');ax.tick_params(axis='x',rotation=65)
    ax.set_title('Exploratory position probes: four held-out trajectories');ax.legend()
    fig.savefig(OUT/'probe_comparison.png',dpi=160);plt.close(fig)
    (OUT/'status.json').write_text(json.dumps({'stage':'complete','episodes':20,'samples':d['samples']}))
    print(text,flush=True)
if __name__=='__main__':main()
