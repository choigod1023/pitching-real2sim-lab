"""Reproducible raw release-window extraction; downloads only with explicit flag."""
from pathlib import Path
import argparse
import hashlib
import json
import cv2
from PIL import Image,ImageChops

IDS={15:'1u4-xPnQ3DCG_ocUZn-bTeieqCGm_W4Ka',16:'1WHEr9MXDQBv0nrEB1ahEfK6Dd-ObP2JL',
     17:'1b37Da2tNjqytG3-rRqkgawoeQJktXVsS',18:'1Err5ZMBa_M5npiO437LuLV4HFjE3b-_U',
     19:'1sa2wrKhEHXsEvM59PQdixIDVa_e5QEDI',20:'1xJDj0xlDX7FHfqPYkfBK3hiJnWhPWsLO',
     21:'1vwbuVrvkbo_wFLTEEYFm2mWRH3RshiWV',22:'1nE1uRyTCxTriwIQMYXC9DjXI5xKRpOuZ'}


def run(download_missing=False):
    root=Path(__file__).resolve().parent;lab=root.parent;out=root/'artifacts/release-window'
    out.mkdir(parents=True,exist_ok=True);rows=[]
    for camera,fid in IDS.items():
        existing=lab/('speed-journey/artifacts/release-observation' if camera in [20,21]
                     else 'ball-coordinate-audit/artifacts/video-audit')/f'camera{camera}.mp4'
        path=existing if existing.exists() else out/f'camera{camera}.mp4'
        if not path.exists():
            if not download_missing:raise FileNotFoundError(f'{path}; pass --download-missing for public files')
            import requests
            r=requests.get('https://drive.usercontent.google.com/download',
                           params={'id':fid,'export':'download'},timeout=(10,40))
            r.raise_for_status()
            if len(r.content)<1000000 or b'ftyp' not in r.content[:100]:
                raise ValueError('Expected public video; no quota/auth bypass attempted')
            path.write_bytes(r.content)
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise ValueError(f'Unable to decode {path}')
        row=dict(camera=camera,path=str(path),drive_id=fid,bytes=path.stat().st_size,
                 observed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 nominal_fps=cap.get(cv2.CAP_PROP_FPS),frames=cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES,960)
        if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))!=960:raise ValueError('Seek mismatch')
        hashes=[]
        for frame in range(960,991):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Truncated video')
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));p=out/f'camera{camera}_{frame}.png';im.save(p)
            hashes.append(dict(frame=frame,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        cap.release();row['extracted_frames']=hashes
        if camera in [19,20,21]:
            old=(existing.parent/(f'camera{camera}_974.png' if camera in [20,21]
                                  else f'camera{camera}_frame974.png'))
            row['decoded_974_matches_previous_pixels']=ImageChops.difference(
                Image.open(old).convert('RGB'),Image.open(out/f'camera{camera}_974.png').convert('RGB')).getbbox() is None
            if not row['decoded_974_matches_previous_pixels']:raise ValueError('Existing frame labeling mismatch')
        rows.append(row)
    manifest=dict(scope='Observed file hashes, not published trusted checksums or exposure synchronization proof',
                  zero_based_decoded_indices=True,videos=rows)
    (out/'raw_source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'{len(rows)} cameras, {sum(len(r["extracted_frames"]) for r in rows)} raw frames; old pixel checks passed')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download-missing',action='store_true');a=p.parse_args();run(a.download_missing)
