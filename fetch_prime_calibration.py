"""Read explicitly public Prime calibration assets, retaining upstream hashes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
OUT=ROOT/'artifacts/prime-camera'
IDS={17:'1mQkWUrm2Hqj3NddB641LkkPJYW4hGzqm',20:'1kkMI-hwAJvxI8Q0kBPVgtS5ZGJdCtjTo',21:'1holuNR3Z5czh4AXXEettbHWEqcb22pi-'}


def confirmation(html, file_id):
    """Follow the publisher's public large-file confirmation, never an auth/quota page."""
    class Form(HTMLParser):
        def __init__(self):super().__init__();self.action=None;self.params={};self.active=False
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='form' and a.get('id')=='download-form':self.active=True;self.action=a.get('action')
            if tag=='input' and self.active and a.get('type')=='hidden':self.params[a.get('name')]=a.get('value')
        def handle_endtag(self,tag):
            if tag=='form':self.active=False
    p=Form();p.feed(html)
    if 'Virus scan warning' not in html or p.action!='https://drive.usercontent.google.com/download':
        raise ValueError('Not the public large-file confirmation')
    if p.params.get('id')!=file_id or p.params.get('export')!='download' or p.params.get('confirm')!='t':
        raise ValueError('Unexpected confirmation file or operation')
    if any(k not in {'id','export','confirm','uuid'} for k in p.params):raise ValueError('Unexpected confirmation parameter')
    return p.params


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    rig=json.loads((LAB/'openbiomechanics/computer_vision/calibration/results/optitrack_rig_provisional.json').read_text())
    inventory_path=OUT/'prime_folder_inventory.json'
    if inventory_path.exists():inventory=json.loads(inventory_path.read_text())
    else:
        provenance=json.loads((ROOT/'prime_camera_input_provenance.json').read_text())
        inventory={'items':{r['file_id']:r['original_name'] for r in provenance['download_sources']}}
    def one(camera):
        fid=IDS[camera];assert f'Camera {camera} (' in inventory['items'][fid]
        name=f'opti_dynamic_checkerboard_cam{camera}.mp4';expected=rig['source_input_provenance'][name]
        url=f'https://drive.usercontent.google.com/download?id={fid}&export=download';path=OUT/name
        row={'camera':camera,'file_id':fid,'original_name':inventory['items'][fid],'url':url,'local_path':str(path),'expected_sha256':expected['sha256']}
        try:
            if not path.exists():
                response=requests.get(url,stream=True,timeout=(10,45));response.raise_for_status()
                if 'text/html' in response.headers.get('Content-Type',''):
                    html=response.text;params=confirmation(html,fid)
                    (OUT/(name+'.download_warning.html')).write_text(html,encoding='utf-8')
                    response=requests.get('https://drive.usercontent.google.com/download',params=params,stream=True,timeout=(10,45));response.raise_for_status()
                    row['public_large_file_confirmation_followed']=True
                part=path.with_suffix('.mp4.part');size=0;first=b''
                if part.exists():
                    old=part.read_bytes()
                    if b'Virus scan warning' in old:part.replace(path.with_suffix('.initial_warning.html'))
                    else:raise FileExistsError('Existing unverified partial download is preserved')
                with part.open('xb') as f:
                    for block in response.iter_content(1024*1024):
                        if not first:first=block[:100]
                        f.write(block);size+=len(block)
                if b'ftyp' not in first or size<1000000:raise ValueError('Response is not the expected video; preserved part for audit')
                digest=sha(part)
                if digest!=expected['sha256']:raise ValueError('Upstream hash mismatch; preserved part for audit')
                part.replace(path)
            row.update(bytes=path.stat().st_size,observed_sha256=sha(path),hash_matches_upstream=sha(path)==expected['sha256'])
        except Exception as e:row.update(error=repr(e),hash_matches_upstream=False)
        print(json.dumps(row),flush=True);return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,IDS))
    (OUT/'download_manifest.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    return rows


if __name__=='__main__':run()
