"""Acquire explicitly listed public assets; no recursive or private Drive access."""
from pathlib import Path
import argparse
import hashlib
import json
import requests

VIDEOS={
 'edge67_cam1.mp4':('edgertronic_dynamic_baseballthrow67mph_1.mp4','1oV_iplHmHo6mtMCEiKCCvPCySqCkWofW'),
 'edgertronic_dynamic_baseballthrow67mph_2.mp4':(None,'1saH7Hz2VGHNSofg0ygpTvvYDonsEmghP'),
 'edgertronic_dynamic_baseballthrow67mph_3.mp4':(None,'1jwLCD-qoH12TE02gN69tyVXDB7Q6RARN'),
 'edgertronic_dynamic_baseballthrow67mph_4.mp4':(None,'1DmX5JcP7o1iglJyLOox97pXFnaEi7JFC'),
 'edge70_cam1.mp4':('edgertronic_dynamic_baseballthrow70mph_1.mp4','1kNtu_YtjZkmfpc3_PHc70VOvnsazM3ti'),
 'edgertronic_dynamic_baseballthrow70mph_2.mp4':(None,'17OUVi9lmL9JcyWGsMN9tSLRa_IzhZbMH'),
 'edgertronic_dynamic_baseballthrow70mph_3.mp4':(None,'1tOAiVkc-nr80mA4cWu9Nkc_S-Rh2Dldc'),
 'edgertronic_dynamic_baseballthrow70mph_4.mp4':(None,'1crx_g3_EJZHHNJL75Fw3StnX0Otpnk70')}


def run(allow_download=False):
    out=Path(__file__).resolve().parent/'artifacts/paired-source';out.mkdir(parents=True,exist_ok=True)
    assets={name:dict(source_name=original or name,url=f'https://drive.usercontent.google.com/download?id={fid}&export=download') for name,(original,fid) in VIDEOS.items()}
    assets['obp-cv-source-index.xlsx']=dict(source_name='OBP-CV official source index',url='https://docs.google.com/spreadsheets/d/1NhpF8DnfBdio_xsU7B44KNuHghtePDp-d3juvCYNm9Q/export?format=xlsx')
    assets['cube-dimension-photo.jpg']=dict(source_name='calibration_cube_27.5inches_per_side_to_red_joint_centers.jpg',url='https://drive.usercontent.google.com/download?id=1wcXLZDoKF3C7lVH1F2BP8wJ1LXfxQLVl&export=download')
    assets['cs200-technical-drawing.png']=dict(source_name='cs-200-technical-drawing-ground-plane.png',url='https://drive.usercontent.google.com/download?id=1BSVylrKh0x8tZA-JsJWCM4fzTkrY9nUA&export=download')
    for i in [1,2]:
        assets[f'yamaguchi_supp{i}.docx']=dict(source_name=f'41598_2025_93632_MOESM{i}_ESM.docx',url=f'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41598-025-93632-y/MediaObjects/41598_2025_93632_MOESM{i}_ESM.docx')
    rows=[]
    for name,metadata in assets.items():
        p=out/name
        if not p.exists():
            if not allow_download:raise FileNotFoundError(f'{p}; --download permits fetching explicitly listed public files')
            r=requests.get(metadata['url'],timeout=(10,40));r.raise_for_status();content=r.content
            if name.endswith('.mp4') and (len(content)<1000000 or b'ftyp' not in content[:100]):raise ValueError('Not video; no auth/quota bypass')
            if name.endswith(('.xlsx','.docx')) and not content.startswith(b'PK'):raise ValueError('Not an Office archive')
            p.write_bytes(content)
        rows.append(dict(**metadata,local_path=str(p),bytes=p.stat().st_size,observed_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (out/'paired_asset_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(f'{len(rows)} public assets verified locally; no private access or external contact');return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args();run(a.download)
