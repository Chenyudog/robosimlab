"""Local LIBERO pi0 baseline. No robot hardware; saves full action/state/video trace."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT / 'runtime/libero_config'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
LEROBOT_SOURCE=Path(os.environ.get('VLA_LEROBOT_SRC','/home/cyd/lerobot/src'))
sys.path.insert(0,str(LEROBOT_SOURCE))
if os.environ.get('VLA_LEGACY_PI0')=='1':
    legacy_site=str(ROOT/'runtime/legacy_pi0/lib/python3.12/site-packages')
    shared_site=str(ROOT/'runtime/.venv/lib/python3.12/site-packages')
    sys.path.insert(sys.path.index(legacy_site)+1,shared_site)
import numpy as np
import torch
import sentencepiece
from accelerate import init_empty_weights
from safetensors import safe_open
from PIL import Image
import imageio.v2 as imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch
from check_counterfactual import capture


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def load_model(path, vision_fp32=False):
    with init_empty_weights():
        model = PI0Pytorch(PI0Config(dtype='bfloat16', device='cuda'))
    model.paligemma_with_expert.gemma_expert.lm_head = torch.nn.Identity()
    model.paligemma_with_expert.paligemma.lm_head = torch.nn.Identity()
    expected = dict(model.named_parameters())
    # Bounded host-memory streaming avoids retaining an entire checkpoint mmap.
    with path.open('rb', buffering=0) as stream:
        header_size=struct.unpack('<Q',stream.read(8))[0]
        header=json.loads(stream.read(header_size))
        data_start=8+header_size
        actual={k.removeprefix('model.'):k for k in header if k!='__metadata__'}
        used=set()
        alias={'paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight':
               'paligemma_with_expert.paligemma.lm_head.weight'}
        dtype_map={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16}
        # Load in physical file order to avoid random seeks on the local HDD.
        def storage_offset(item):
            name,_=item
            key=name if name in actual else alias.get(name,name)
            if key not in actual:raise RuntimeError(f'Missing parameter {name}')
            return header[actual[key]]['data_offsets'][0]
        ordered=sorted(expected.items(),key=storage_offset)
        for i,(name,param) in enumerate(ordered):
            key=name if name in actual else alias.get(name,name)
            if key not in actual:raise RuntimeError(f'Missing parameter {name}')
            meta=header[actual[key]]
            if list(param.shape)!=meta['shape']:raise RuntimeError(f'Shape mismatch: {name}')
            dtype=torch.float32 if any(x in name for x in ['input_layernorm','post_attention_layernorm','.norm.']) or (vision_fp32 and any(x in name for x in ['vision_tower','multi_modal_projector'])) else torch.bfloat16
            source_dtype=dtype_map[meta['dtype']]
            item_size=torch.empty((),dtype=source_dtype).element_size()
            begin,end=meta['data_offsets']
            # Avoid a transient ~1GB GPU embedding allocation in FP32-vision mode.
            cpu_vision=vision_fp32 and os.environ.get('VLA_VISION_CPU')=='1'
            storage_device='cpu' if vision_fp32 and (name.endswith('language_model.embed_tokens.weight') or (cpu_vision and any(v in name for v in ['vision_tower','multi_modal_projector']))) else 'cuda'
            value=torch.empty(tuple(param.shape),device=storage_device,dtype=dtype)
            flat=value.view(-1)
            offset=begin
            while offset<end:
                count=min(16*1024*1024,end-offset)
                stream.seek(data_start+offset)
                data=stream.read(count)
                if len(data)!=count:raise RuntimeError('Truncated checkpoint tensor')
                cpu=torch.frombuffer(bytearray(data),dtype=source_dtype)
                index=(offset-begin)//item_size
                flat[index:index+cpu.numel()].copy_(cpu)
                del cpu,data
                offset+=count
            parent,_,leaf=name.rpartition('.')
            setattr(model.get_submodule(parent),leaf,torch.nn.Parameter(value,requires_grad=False))
            del flat,value
            used.add(key)
            if i%150==0:print(f'Loading weights {i}/{len(expected)}',flush=True)
        normalization_buffers={prefix+'.'+stat for prefix in ['normalize_inputs.buffer_observation_state','normalize_targets.buffer_action','unnormalize_outputs.buffer_action'] for stat in ['mean','std']}
        if normalization_buffers.intersection(actual) and not normalization_buffers.issubset(actual):raise RuntimeError('Incomplete embedded normalization buffers')
        unexpected=set(actual)-used-normalization_buffers-{'paligemma_with_expert.gemma_expert.lm_head.weight'}
        if unexpected:raise RuntimeError(f'Unexpected tensors {unexpected}')
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError('Unloaded parameters')
    # Parameters are already placed; move nonpersistent rotary/other buffers only.
    for module_path,module in model.named_modules():
        buffer_device='cpu' if vision_fp32 and os.environ.get('VLA_VISION_CPU')=='1' and any(v in module_path for v in ['vision_tower','multi_modal_projector']) else 'cuda'
        for name,buffer in module._buffers.items():
            if buffer is not None:module._buffers[name]=buffer.to(buffer_device)
    return model.eval()


def quat_axis(q):
    q = np.array(q, dtype=np.float64, copy=True)
    w = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - w*w)
    return np.zeros(3) if np.isclose(den, 0) else q[:3] * (2*np.arccos(w)/den)


class Policy:
    def __init__(self, model_dir, vision_fp32=False, openpi_delta=False):
        self.model = load_model(model_dir / 'model.safetensors',vision_fp32=vision_fp32)
        if vision_fp32:
            path=self.model.paligemma_with_expert
            original_embed=path.embed_image
            def fp32_embed(image):
                with torch.autocast('cuda',enabled=False):
                    if os.environ.get('VLA_VISION_CPU')=='1':
                        return original_embed(image.float().cpu()).to(image.device)
                    return original_embed(image.float())
            path.embed_image=fp32_embed
            # Keep the large vocabulary embedding on CPU for this local precision audit.
            # Only the short gathered token embedding crosses to CUDA.
            embedding=path.paligemma.model.language_model.get_input_embeddings().to('cpu')
            def cpu_language_embed(tokens):
                return embedding(tokens.cpu()).to(tokens.device)
            path.embed_language_tokens=cpu_language_embed
            torch.cuda.empty_cache()
        self.openpi_delta=openpi_delta
        stats_name='norm_stats.json' if openpi_delta else 'checkpoint_norm_stats.json'
        self.stats = json.loads((model_dir / stats_name).read_text())['norm_stats']
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / 'paligemma_tokenizer.model'))

    def infer(self, obs, text, seed):
        # Official OpenPI pi0 uses mean/std; pi0.5 uses quantiles.
        s = np.concatenate([obs['robot0_eef_pos'], quat_axis(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']]).astype(np.float32)
        raw_state=s.copy()
        st = self.stats['state']
        s = (s-np.array(st['mean'][:8], np.float32))/(np.array(st['std'][:8], np.float32)+1e-8)
        state = torch.tensor(np.pad(s, (0, 24))[None], device='cuda')
        images = []
        for key in ['agentview_image', 'robot0_eye_in_hand_image']:
            arr = np.ascontiguousarray(obs[key][::-1, ::-1])
            arr = np.asarray(Image.fromarray(arr).resize((224,224), Image.Resampling.BILINEAR)).copy()
            images.append(torch.tensor(arr, device='cuda', dtype=torch.float32).permute(2,0,1)[None]/127.5-1)
        masks = [torch.ones(1, dtype=torch.bool, device='cuda') for _ in images]
        # Fully masked third-camera padding is omitted; it cannot influence valid tokens.
        ids = (self.sp.encode(text.strip(), add_bos=True)+self.sp.encode('\n'))[:48]
        tokens = torch.tensor([ids+[0]*(48-len(ids))], device='cuda')
        mask = torch.tensor([[True]*len(ids)+[False]*(48-len(ids))], device='cuda')
        noise = torch.randn((1,50,32), device='cuda', generator=torch.Generator(device='cuda').manual_seed(seed))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = self.model.sample_actions(images,masks,tokens,mask,state,noise=noise,num_steps=10)
        pred = pred[0,:,:7].float().cpu().numpy()
        ac = self.stats['actions']
        actions = pred * (np.asarray(ac['std'][:7])+1e-8) + np.asarray(ac['mean'][:7])
        if self.openpi_delta:
            actions[:,:6]+=raw_state[None,:6]
        if not np.isfinite(actions).all():
            raise RuntimeError('Nonfinite actions')
        return actions.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir',type=Path,default=ROOT/'models/pi0_libero')
    ap.add_argument('--suite',default='libero_spatial')
    ap.add_argument('--task-id',type=int,default=0)
    ap.add_argument('--episode-index',type=int,default=0)
    ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--replan-steps',type=int,default=5)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--snapshot-step',type=int,default=40)
    ap.add_argument('--vision-fp32',action='store_true')
    ap.add_argument('--openpi-delta',action='store_true',help='Compatibility audit: use OpenPI checkpoint stats and restore extra_delta_transform')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not (args.model_dir / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Checkpoint download not complete: {args.model_dir}')
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    if not 1 <= args.replan_steps <= 50:
        raise ValueError('replan-steps must be 1..50')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    suite=benchmark.get_benchmark_dict()[args.suite]()
    task=suite.get_task(args.task_id)
    manifest={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest.update(task=task.language,protocol='normal closed-loop baseline; no perturbation',
        normalization=('OpenPI pi0_libero mean/std plus inverse extra_delta_transform' if args.openpi_delta else 'Mean/std from this checkpoints paired normalization assets'),
        precision=('FP32 vision/projector, BF16 language/expert, CPU vocabulary embedding, FP32 normalization' if args.vision_fp32 else 'BF16 weights, FP32 normalization; eager attention'),
        image_preprocessing='rotate180, PIL bilinear resize to224; two valid cameras',
        checkpoint_source=json.loads((args.model_dir/'source.json').read_text()),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        lerobot_source=str(LEROBOT_SOURCE),
        lerobot_commit=(json.loads((LEROBOT_SOURCE.parent/'source.json').read_text())['revision'] if (LEROBOT_SOURCE.parent/'source.json').exists() else subprocess.check_output(['git','-C',str(LEROBOT_SOURCE.parent),'rev-parse','HEAD'],text=True).strip()),
        backend='legacy_pi0' if os.environ.get('VLA_LEGACY_PI0')=='1' else 'modern_pi0',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assets=json.loads((args.model_dir/'assets_sources.json').read_text()))
    save_json(args.output/'manifest.json',manifest)
    try:
        policy=Policy(args.model_dir,vision_fp32=args.vision_fp32,openpi_delta=args.openpi_delta)
    except BaseException as exc:
        save_json(args.output/'error.json', {'status':'error','stage':'model_load','type':type(exc).__name__,'message':str(exc)})
        raise
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    actions=[];states=[];timings=[];queue=deque();done=False
    writer=imageio.get_writer(str(args.output/'rollout.mp4'),fps=20)
    started=time.monotonic()
    try:
        env.seed(args.seed);env.reset()
        init_path=Path(get_libero_path('init_states'))/task.problem_folder/Path(task.init_states_file).name
        init=torch.load(init_path,weights_only=False)
        obs=env.set_init_state(init[args.episode_index])
        for _ in range(10):
            obs,_,_,_=env.step([0.]*6+[-1.])
        for step in range(args.max_steps):
            writer.append_data(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]))
            states.append(np.concatenate([obs['robot0_eef_pos'],obs['robot0_eef_quat'],obs['robot0_gripper_qpos']]))
            if step == args.snapshot_step:
                snapshot=capture(env)
                snapshot['pending_action_queue']=list(queue)
                snapshot['next_inference_seed']=args.seed+step
                torch.save(snapshot,args.output/'snapshot.pt')
                save_json(args.output/'snapshot_metadata.json',{'step':step,'task':task.language,'pending_actions':len(queue),
                    'note':'Internal trusted pickle artifact; contains simulator/controller state, not policy KV cache.'})
            if not queue:
                tick=time.monotonic()
                chunk=policy.infer(obs,task.language,args.seed+step)
                timings.append({'step':step,'seconds':time.monotonic()-tick})
                queue.extend(chunk[:args.replan_steps])
            raw=queue.popleft()
            action=np.clip(raw,-1,1)
            actions.append(action)
            obs,reward,done,info=env.step(action.tolist())
            if step % 10 == 0 or done:
                print(f'step={step+1} success={bool(env.check_success())} elapsed={time.monotonic()-started:.1f}s',flush=True)
            if done:
                break
        success=bool(env.check_success())
        np.savez_compressed(args.output/'trajectory.npz',actions=np.array(actions),states=np.array(states),final_sim_state=env.get_sim_state())
        save_json(args.output/'result.json',{'status':'complete','success':success,'steps':len(actions),
            'elapsed_seconds':time.monotonic()-started,'inference':timings,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'note':'One episode is a smoke test, not a benchmark success-rate estimate.'})
        print(f'COMPLETE success={success} steps={len(actions)} output={args.output}',flush=True)
    except BaseException as exc:
        save_json(args.output/'error.json',{'status':'error','type':type(exc).__name__,'message':str(exc),'steps':len(actions)})
        raise
    finally:
        writer.close();env.close()

if __name__=='__main__':
    main()
