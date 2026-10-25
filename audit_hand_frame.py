import hashlib,json
from pathlib import Path
import ezc3d
import numpy as np

root=Path(__file__).resolve().parent
p=root.parent/'tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d'
c=ezc3d.c3d(str(p))
labels=c['parameters']['ROTATION']['LABELS']['value']
A=c['data']['rotations']
out={'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
     'theia_parameters':{k:(v['value'].tolist() if isinstance(v['value'],np.ndarray) else v['value']) for k,v in c['parameters']['THEIA3D'].items() if k!='__METADATA__'},
     'units':c['parameters']['POINT']['UNITS']['value'],'labels':labels,'shape':list(A.shape),'segments':{}}
for side in ['l','r']:
 hand=A[:,:,labels.index(side+'_hand_4X4'),:].transpose(2,0,1)
 arm=A[:,:,labels.index(side+'_larm_4X4'),:].transpose(2,0,1)
 R=hand[:,:3,:3]
 d=hand[:,:3,3]-arm[:,:3,3]
 local=np.einsum('fji,fj->fi',arm[:,:3,:3],d)
 out['segments'][side]={'orthogonality_max':float(np.abs(R.transpose(0,2,1)@R-np.eye(3)).max()),'det_range':[float(np.linalg.det(R).min()),float(np.linalg.det(R).max())],'bottom_row_max_deviation':float(np.abs(hand[:,3,:]-[0,0,0,1]).max()),'forearm_to_hand_origin_local_median_mm':np.median(local,axis=0).tolist(),'forearm_to_hand_origin_local_range_mm':[local.min(0).tolist(),local.max(0).tolist()]}
(root/'artifacts/hand-frame-sources').mkdir(parents=True,exist_ok=True)
(root/'artifacts/hand-frame-sources/metadata_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out['segments'],indent=2))
