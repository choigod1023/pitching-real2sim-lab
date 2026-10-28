"""Download public static targets selected from the recorded Drive listing."""
import argparse, concurrent.futures, hashlib, json
from pathlib import Path
import cv2, requests

ROOT=Path('artifacts/edge-calibration')
def fetch(entry):
    dest=ROOT/entry['name']
    if not dest.exists():
        r=requests.get('https://drive.usercontent.google.com/download',params={'id':entry['id'],'export':'download'},timeout=(10,60));r.raise_for_status()
        if b'ftyp' not in r.content[:100]:raise ValueError('Response is not MP4')
        dest.write_bytes(r.content)
    if dest.stat().st_size != int(entry['size']):raise ValueError(f'Size mismatch: {dest.name}')
    observed_hash=hashlib.sha256(dest.read_bytes()).hexdigest()
    if entry.get('sha256') and observed_hash != entry['sha256']:raise ValueError(f'Hash mismatch: {dest.name}')
    cap=cv2.VideoCapture(str(dest));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    for frame in [0,120,240]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,min(frame,count-1));ok,im=cap.read()
        if not ok:raise ValueError(f'Cannot read frame {frame}')
        cv2.imwrite(str(ROOT/f'{dest.stem}_f{frame}.png'),im)
    cap.release()
    return dict(entry,sha256=observed_hash,frame_count=count,stored_playback_fps=fps)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',default='checkerboard',choices=['checkerboard','cube','groundplane']);a=p.parse_args()
    listing=json.loads(Path('edge_source_inventory.json').read_text());entries=[e for e in listing if f'edgertronic_static_{a.kind}_' in e['name']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:manifest=list(pool.map(fetch,entries))
    (ROOT/f'{a.kind}-source-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps([{k:e[k] for k in ['name','frame_count','stored_playback_fps']} for e in manifest]))
