"""Resumable exact-range download directly into one file, then SHA256 verify."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,os,time
from pathlib import Path
import requests


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--revision',required=True);ap.add_argument('--directory',type=Path,required=True)
    args=ap.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    api=f'https://huggingface.co/api/models/{args.repo}/tree/{args.revision}?expand=true'
    r=requests.get(api,timeout=30);r.raise_for_status();files=r.json()
    info=next(x for x in files if x['path']=='model.safetensors')
    total=info['size'];sha=info['lfs']['oid']
    spec={'repo':args.repo,'revision':args.revision,'bytes':total,'sha256':sha}
    source=args.directory/'source.json'
    if source.exists() and json.loads(source.read_text())!=spec:raise RuntimeError('Model directory provenance differs')
    source.write_text(json.dumps(spec,indent=2))
    url=f'https://huggingface.co/{args.repo}/resolve/{args.revision}/'
    for info in files:
        name=info['path']
        if name=='model.safetensors' or info['type']!='file':continue
        if name.endswith(('.json','.md','.safetensors')):
            res=requests.get(url+name,timeout=30);res.raise_for_status();(args.directory/name).write_bytes(res.content)
    dst=args.directory/'model.safetensors';part=args.directory/'model.download.part';marks=args.directory/'download_progress'
    marks.mkdir(exist_ok=True)
    if not dst.exists():
        fd=os.open(part,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,total)
        block=64*1024*1024;n=(total+block-1)//block
        def fetch(i):
            marker=marks/f'{i:04d}.done'
            if marker.exists():return
            start=i*block;end=min(total,start+block)-1
            for attempt in range(12):
                try:
                    with requests.get(url+f'model.safetensors?download=true&part={i}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,90)) as response:
                        response.raise_for_status()
                        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':raise RuntimeError('Range mismatch')
                        offset=start
                        for data in response.iter_content(1024*1024):
                            written=0
                            while written<len(data):written+=os.pwrite(fd,data[written:],offset+written)
                            offset+=len(data)
                    if offset!=end+1:raise RuntimeError('Incomplete range')
                    marker.write_text(str(end-start+1));return
                except Exception as exc:
                    print(f'chunk={i} retry={attempt+1} {type(exc).__name__}',flush=True)
                    if attempt==11:raise
                    time.sleep(min(attempt+1,10))
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                for count,future in enumerate(as_completed([pool.submit(fetch,i) for i in range(n)]),1):
                    future.result()
                    if count%5==0 or count==n:print(f'DOWNLOAD {count}/{n}',flush=True)
        finally:os.close(fd)
        h=hashlib.sha256()
        with part.open('rb') as inp:
            for data in iter(lambda:inp.read(16*1024*1024),b''):h.update(data)
        if h.hexdigest()!=sha:raise RuntimeError('SHA256 mismatch; do not use this checkpoint')
        part.replace(dst)
        (args.directory/'verification.json').write_text(json.dumps({**spec,'verified':True},indent=2))
        for p in marks.iterdir():p.unlink()
        marks.rmdir()
        print('VERIFIED',dst,flush=True)

if __name__=='__main__':main()
