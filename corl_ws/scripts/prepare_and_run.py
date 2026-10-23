"""Resumable ranged checkpoint download, SHA256 verification, then baseline."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/pi0_libero'
TOTAL=7011549448
SHA='b78433b33b7d9f9da964852e5b88987f43a0ee52552cbb6ab83aeeabd8c27e75'
URL='https://huggingface.co/lerobot/pi0_libero_base/resolve/98ac51b805725b9f6cc90ad0b92b8b8ecbefa9eb/model.safetensors'
BLOCK=64*1024*1024
CHUNKS=MODEL/'download_chunks'
CHUNKS.mkdir(exist_ok=True)


def fetch(i):
    start=i*BLOCK;end=min(TOTAL,start+BLOCK)-1
    dst=CHUNKS/f'{i:04d}.part'
    if dst.exists() and dst.stat().st_size==end-start+1:
        return i
    tmp=dst.with_suffix('.tmp')
    for attempt in range(12):
        try:
            # Distinct query avoids caching a different byte range under the same URL.
            with requests.get(URL+f'?download=true&chunk={i}',headers={'Range':f'bytes={start}-{end}'},
                              timeout=(30,120),stream=True) as r:
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{TOTAL}':
                    raise RuntimeError('Server did not honor exact requested range')
                with tmp.open('wb') as f:
                    for b in r.iter_content(1024*1024):
                        f.write(b)
            if tmp.stat().st_size!=end-start+1:
                raise RuntimeError('Incomplete chunk')
            tmp.replace(dst)
            return i
        except Exception as e:
            print(f'chunk={i} attempt={attempt+1} error={type(e).__name__}',flush=True)
            if attempt==11:raise
            time.sleep(min(10,attempt+1))


def main():
    checkpoint=MODEL/'model.safetensors'
    if not checkpoint.exists():
        sequential=MODEL/'model.safetensors.part'
        if sequential.exists():
            with sequential.open('rb') as f:
                for i in range(sequential.stat().st_size//BLOCK):
                    dst=CHUNKS/f'{i:04d}.part'
                    b=f.read(BLOCK)
                    if not dst.exists():dst.write_bytes(b)
        n=(TOTAL+BLOCK-1)//BLOCK
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(fetch,i) for i in range(n)]
            for completed,future in enumerate(as_completed(futures),1):
                future.result()
                if completed%5==0 or completed==n:
                    print(f'DOWNLOAD {completed}/{n} chunks',flush=True)
        tmp=MODEL/'model.assembled.tmp'
        h=hashlib.sha256()
        with tmp.open('wb') as out:
            for i in range(n):
                with (CHUNKS/f'{i:04d}.part').open('rb') as inp:
                    for b in iter(lambda:inp.read(8*1024*1024),b''):
                        h.update(b);out.write(b)
        if tmp.stat().st_size!=TOTAL or h.hexdigest()!=SHA:
            raise RuntimeError('Checkpoint SHA256 mismatch')
        tmp.replace(checkpoint)
        for p in CHUNKS.iterdir():p.unlink()
        CHUNKS.rmdir()
        if sequential.exists():sequential.unlink()
        (MODEL/'verification.json').write_text(json.dumps({'bytes':TOTAL,'sha256':SHA,'verified':True},indent=2))
        print('Checkpoint SHA256 verified',flush=True)
    output=ROOT/'outputs/baseline_spatial0_seed7_run2'
    if output.exists():
        raise FileExistsError(f'Will not overwrite existing run: {output}')
    with (ROOT/'logs/baseline_spatial0_seed7_run2.log').open('w') as log:
        subprocess.run([str(ROOT/'run_baseline.sh'),'--output',str(output),'--suite','libero_spatial',
                        '--task-id','0','--seed','7','--max-steps','220','--replan-steps','5'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Baseline finished; inspect result.json',flush=True)

if __name__=='__main__':
    main()
