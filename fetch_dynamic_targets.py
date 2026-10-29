"""Public dynamic checkerboard acquisition, isolated from prior experiments."""
import concurrent.futures,hashlib,json
from html.parser import HTMLParser
from pathlib import Path
import cv2,requests

ROOT=Path('artifacts/dynamic-recovery')
class ConfirmationForm(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.fields={};self.inside=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':self.inside=True;self.action=a.get('action')
        if self.inside and tag=='input' and a.get('name') in ['id','export','confirm','uuid']:self.fields[a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.inside=False

def confirmation(html,expected_id):
    """Follow only Google's observed public large-file confirmation form."""
    f=ConfirmationForm();f.feed(html)
    if f.action!='https://drive.usercontent.google.com/download' or f.fields.get('id')!=expected_id or f.fields.get('export')!='download' or not f.fields.get('confirm'):
        raise ValueError('Not the expected public download confirmation; no alternate access attempted')
    return f.action,f.fields

def fetch(entry):
    ROOT.mkdir(parents=True,exist_ok=True);dest=ROOT/entry['name']
    if not dest.exists():
        r=requests.get('https://drive.usercontent.google.com/download',params={'id':entry['id'],'export':'download'},timeout=(10,60));r.raise_for_status()
        if b'ftyp' not in r.content[:100] and 'Google Drive - Virus scan warning' in r.text:
            action,fields=confirmation(r.text,entry['id']);r=requests.get(action,params=fields,timeout=(10,60));r.raise_for_status()
        if b'ftyp' not in r.content[:100]:raise ValueError('Public response is not MP4; access not bypassed')
        dest.write_bytes(r.content)
    if dest.stat().st_size!=int(entry['size']):raise ValueError(f'Size mismatch {dest.name}')
    digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    if entry.get('sha256') and entry['sha256']!=digest:raise ValueError(f'Hash mismatch {dest.name}')
    cap=cv2.VideoCapture(str(dest));info={'frame_count':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'playback_fps':cap.get(cv2.CAP_PROP_FPS),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))};cap.release()
    return dict(entry,sha256=digest,**info)
if __name__=='__main__':
    entries=json.loads(Path('dynamic_source_inventory.json').read_text(encoding='utf-8'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,entries))
    (ROOT/'dynamic-source-manifest.json').write_text(json.dumps(results,indent=2));print(json.dumps([{k:r[k] for k in ['name','frame_count','playback_fps']} for r in results]))
