"""Per-image planar consistency; no metric target length or speed selection."""
import json
from pathlib import Path
import cv2,numpy as np
from dense_camera_link import ROOT
from dynamic_camera_recovery import OBJ

def run():
    out=[]
    for camera in [1,3]:
        for frame in range(540,613):
            row=json.loads((ROOT/'coverage'/f'cam{camera}_frame{frame}.json').read_text())
            for method in ['native_sb','source720_subpix5']:
                uv=row['methods'][method]['points']
                if uv is None:continue
                uv=np.array(uv);h,_=cv2.findHomography(OBJ[:,:2],uv,0);prediction=cv2.perspectiveTransform(OBJ[:,:2].reshape(-1,1,2),h).reshape(-1,2);e=np.linalg.norm(prediction-uv,axis=1)
                out.append({'camera':camera,'frame':frame,'method':method,'homography_rms_px':float(np.sqrt(np.mean(e**2))),'maximum_residual_px':float(max(e))})
    (ROOT/'per-view-planarity-diagnostic.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':run()
