"""Recover failed 4x crop arm; keep existing baseline/native observations unchanged."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
from dynamic_camera_recovery import ROOT,detect,run_fit

def repair(camera):
    cv2.setNumThreads(1);source=ROOT/f'detections_camera{camera}.json';rows=json.loads(source.read_text());manifest=json.loads((ROOT/'dynamic-source-manifest.json').read_text());entry=next(e for e in manifest if e['name'].endswith(f'_{camera}.mp4'));cap=cv2.VideoCapture(str(ROOT/entry['name']));code_hash=hashlib.sha256(Path('dynamic_camera_recovery.py').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    for f,row in rows.items():
        anchor=row['methods']['source720_subpix5']['points']
        if anchor is None:row['methods']['roi2_border_sb']={'points':None,'info':{'failure':'no same-image source anchor'}};continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,im=cap.read()
        if not ok:raise ValueError(f'Cannot read camera{camera} frame{f}')
        start=time.perf_counter();uv,info=detect(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),'roi2_border_sb',np.array(anchor));row['methods']['roi2_border_sb']={'points':None if uv is None else uv.tolist(),'info':info,'seconds':time.perf_counter()-start,'implementation_sha256':code_hash}
    cap.release();(ROOT/f'recovered-detections-camera{camera}.json').write_text(json.dumps(rows,indent=2));print('recovered camera',camera,'ROI2',sum(v['methods']['roi2_border_sb']['points'] is not None for v in rows.values()),'/',len(rows),flush=True);return rows

if __name__=='__main__':
    split=ROOT/'frozen-fit-eval-split.json';original=ROOT/'roi4-failed-fit-eval-split.json'
    if split.exists() and not original.exists():original.write_bytes(split.read_bytes())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs={i:pool.submit(repair,i) for i in [1,4]};d={i:f.result() for i,f in fs.items()}
    run_fit(d[1],d[4],methods=['source720_subpix5','native_sb','roi2_border_sb'])
