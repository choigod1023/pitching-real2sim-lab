"""Reproduce supplied relative timing fit; do not adopt its best shift."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from joint_video import load_joints
up=root.parent/'openbiomechanics/computer_vision/4d_reconstruction'
c3d=root.parent/'tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d'
paths=[up/'offset_study.py',up/'release_keypoints.csv',up/'theia_alignment.json',c3d]
result=subprocess.run([sys.executable,str(paths[0]),'--c3d',str(c3d)],capture_output=True,text=True,check=True)
out=root/'artifacts/gap-observation'
out.mkdir(parents=True,exist_ok=True)
(out/'prime_offset_log.txt').write_text(result.stdout,encoding='utf-8')
align=json.loads(paths[2].read_text())
xyz,names,rate=load_joints(c3d)
q=xyz[978]
rig=align['scale']*(q@np.array(align['rotation']).T)+align['translation']
cal=json.loads((up.parent/'calibration/results/calibration_results.json').read_text())['calibrations']['optitrack_19']
uv=np.column_stack((cal['fx_px']*rig[:,0]/rig[:,2]+cal['cx_px'],cal['fy_px']*rig[:,1]/rig[:,2]+cal['cy_px']))
image=Image.open(root.parent/'speed-journey/artifacts/release-observation/four_views_977.jpg').crop((0,0,640,360)).resize((1280,720))
d=ImageDraw.Draw(image)
for name,color in [('l_hand','yellow'),('r_hand','cyan'),('l_larm','yellow'),('r_larm','cyan')]:
    x,y=uv[names.index(name)]
    d.ellipse((x-8,y-8,x+8,y+8),outline=color,width=3)
    d.text((x+10,y),name,fill=color)
image.save(out/'prime_segment_labels_977.png')
(out/'prime_alignment_manifest.json').write_text(json.dumps({
 'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 'exit_code':result.returncode,'alignment':align,
 'camera19_frame977_projected_segment_origins':{n:uv[names.index(n)].tolist() for n in ['l_hand','r_hand','l_larm','r_larm']},
 'scope':'Reproduction of contributor in-sample relative alignment, not independent synchronization or anatomical hand-origin verification. No parameters adopted.'},indent=2),encoding='utf-8')
print(result.stdout)
