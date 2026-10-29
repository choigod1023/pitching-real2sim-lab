"""Raw-only red component observations; no camera projection or model fit."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from fetch_prime_calibration import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/prime-camera'
ROIS={17:[520,260,790,535],20:[450,240,770,555],21:[450,260,780,625]}


def red_centres(image,roi):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=(((hsv[:,:,0]<12)|(hsv[:,:,0]>168))&(hsv[:,:,1]>80)&(hsv[:,:,2]>35)).astype(np.uint8)
    x0,y0,x1,y1=roi;allowed=np.zeros_like(mask);allowed[y0:y1,x0:x1]=1;mask*=allowed
    n,labels,stats,centres=cv2.connectedComponentsWithStats(mask,8)
    rows=[]
    for i in range(1,n):
        if 2<=stats[i,cv2.CC_STAT_AREA]<=500:
            rows.append({'pixel':centres[i].tolist(),'area_px':int(stats[i,cv2.CC_STAT_AREA])})
    rows.sort(key=lambda r:(r['pixel'][1],r['pixel'][0]))
    return rows


def run():
    result={}
    for c,roi in ROIS.items():
        p=OUT/f'opti_cube_cam{c}.png';im=cv2.imread(str(p));rows=red_centres(im,roi)
        result[str(c)]={'source_image':str(p),'source_sha256':sha(p),'roi':roi,'markers':rows}
        native=Image.open(p).convert('RGB');d=ImageDraw.Draw(native)
        for i,r in enumerate(rows):
            x,y=r['pixel'];d.ellipse((x-4,y-4,x+4,y+4),outline='#55e0ff',width=1)
            d.text((x+5,y-12),str(i),font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13),fill='yellow')
        native.crop(tuple(roi)).resize(((roi[2]-roi[0])*3,(roi[3]-roi[1])*3)).save(OUT/f'cube{c}_raw_markers.png')
        print(c,len(rows))
    (OUT/'cube_red_observations.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':run()
