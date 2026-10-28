"""Exploratory 2D bright-core extraction. Manual ROIs, no connected components.
Requires extracted native frame PNGs in this directory. No 3D or speed claim.
Every threshold result incl empty mask retained; thresholds do not imply confidence.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("frames_dir", type=Path)
root = parser.parse_args().frames_dir.resolve()
rows=[]
for camera, thresholds in [('camera19',[100,130,160]),('camera22',[70,90,110])]:
 for frame in range(978,986):
  a=np.asarray(Image.open(root/f'{camera}_frame{frame}.png'))
  if camera=='camera19': cx=785-20*(frame-978); cy=519+5.6*(frame-978)
  else: cx=715+17*(frame-980); cy=298-0.8*(frame-980)
  x0=int(cx-24); y0=int(cy-16); roi=a[y0:y0+33,x0:x0+49]
  for threshold in thresholds:
   yy,xx=np.where(roi.min(axis=2)>threshold)
   row=dict(camera=camera,frame_zero_based=frame,time_s=frame/360,threshold=threshold,roi_xyxy=[x0,y0,x0+49,y0+33],bright_pixel_count=len(xx))
   if len(xx):row.update(x_px=float(xx.mean()+x0),y_px=float(yy.mean()+y0),bbox_xyxy=[int(xx.min()+x0),int(yy.min()+y0),int(xx.max()+x0),int(yy.max()+y0)])
   else:row['status']='empty_threshold_mask'
   rows.append(row)
(root/'exploratory_2d_candidates.json').write_text(json.dumps({'scope':'Raw RGB threshold centroid within manually set ROI. No connected-component segmentation or validation. Camera19 later ROIs include possible background contamination; preserve all failures. Frame metadata time only, not validated synchronization.','rows':rows},indent=2))
for camera in ['camera19','camera22']:
 selected=[r for r in rows if r['camera']==camera]; print(camera,'empty masks',sum(not r['bright_pixel_count'] for r in selected),'/',len(selected))
 for frame in range(978,986):
  subset=[r for r in selected if r['frame_zero_based']==frame and r['bright_pixel_count']]
  if subset:print(frame,'centers',[(r['threshold'],round(r['x_px'],2),round(r['y_px'],2)) for r in subset])
