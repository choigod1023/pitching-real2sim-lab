"""Compare immutable visible regions to provided hand alignment, no refitting."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from joint_video import load_joints
from fixed_alignment_audit import similarity,project


def region_distance(point,box):
    point,box=np.asarray(point,float),np.asarray(box,float)
    if point.shape!=(2,) or box.shape!=(4,) or not np.isfinite(np.r_[point,box]).all() or np.any(box[2:]<=box[:2]):
        raise ValueError('Finite point and increasing rectangle required')
    delta=np.maximum(np.maximum(box[:2]-point,point-box[2:]),0)
    return float(np.linalg.norm(delta)),float(np.linalg.norm(point-(box[:2]+box[2:])/2))


def frame_path(lab,camera,frame):
    return lab/('speed-journey/artifacts/release-observation' if camera in (20,21) else 'ball-coordinate-audit/artifacts/video-audit')/(f'camera{camera}_{frame}.png' if camera in (20,21) else f'camera{camera}_frame{frame}.png')


def run():
    root=Path(__file__).resolve().parent;lab=root.parent;out=root/'artifacts/hand-image';out.mkdir(parents=True,exist_ok=True)
    paths=dict(protocol=root/'HAND_IMAGE_PROTOCOL.md',regions=root/'hand_image_regions.json',alignment=lab/'openbiomechanics/computer_vision/4d_reconstruction/theia_alignment.json',
        body=lab/'tracking-calibration/artifacts/joint-video/OBP_movement_BaseballThrow_001.c3d',
        calibration=lab/'openbiomechanics/computer_vision/calibration/results/calibration_results.json',rig=lab/'openbiomechanics/computer_vision/calibration/results/optitrack_rig_provisional.json',
        ball=lab/'speed-journey/artifacts/release-observation/candidate_audit.json')
    regions=json.loads(paths['regions'].read_text());align=json.loads(paths['alignment'].read_text());xyz,names,rate=load_joints(paths['body'])
    intr=json.loads(paths['calibration'].read_text())['calibrations'];rig=json.loads(paths['rig'].read_text())['camera_poses'];ball=json.loads(paths['ball'].read_text())
    records=[]
    for row in regions['regions']:
        for offset in (0,1,2):
            prediction=similarity(xyz[row['frame']+offset,names.index(row['segment'])],align)
            uv=project(prediction,rig[str(row['camera'])],intr[f"optitrack_{row['camera']}"])
            dist,center=region_distance(uv,row['box'])
            records.append(dict(**row,offset=offset,pixel=uv.tolist(),outside_distance_px=dist,center_distance_px=center,inside=dist==0))
    summaries=[]
    for offset in (0,1,2):
        selected=[r for r in records if r['offset']==offset]
        summaries.append(dict(offset=offset,inside=sum(r['inside'] for r in selected),n=len(selected),outside_mean_px=float(np.mean([r['outside_distance_px'] for r in selected])),
            outside_max_px=max(r['outside_distance_px'] for r in selected),by_camera={str(c):dict(inside=sum(r['inside'] for r in selected if r['camera']==c),mean_outside_px=float(np.mean([r['outside_distance_px'] for r in selected if r['camera']==c]))) for c in (19,20,21)}))
    # Same raw images with frozen regions and all offsets. Ball detections are separate.
    for frame in (974,977,979):
        sheet=Image.new('RGB',(1280,780),'#111824');d=ImageDraw.Draw(sheet)
        for index,camera in enumerate((19,20,21,22)):
            path=frame_path(lab,camera,frame);im=Image.open(path).convert('RGB');draw=ImageDraw.Draw(im)
            for row in [r for r in records if r['camera']==camera and r['frame']==frame]:
                if row['offset']==1:draw.rectangle(row['box'],outline='yellow',width=2)
                x,y=row['pixel'];color={0:'cyan',1:'lime',2:'magenta'}[row['offset']]
                draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
            for det in ball['detections']:
                if det['camera']==camera and det['frame']==frame and det['threshold']==80 and det.get('pixel') is not None:
                    x,y=det['pixel'];draw.ellipse((x-7,y-7,x+7,y+7),outline='red',width=2)
            draw.text((10,10),f'camera{camera} frame{frame}: CYAN=offset0 GREEN=1 MAGENTA=2 RED=ball candidate',fill='yellow')
            if camera==22:draw.text((10,40),'THROWING ARM OBSCURED: no anatomical annotation',fill='yellow')
            sheet.paste(im.resize((640,360)),((index%2)*640,(index//2)*360+40))
        d.text((10,10),'RAW VIDEO + FIXED MODEL PROJECTIONS | yellow regions are approximate, NOT ground truth | no correction applied',fill='white')
        sheet.save(out/f'audit-{frame}.png')
    paths.update({f'image_{c}_{f}':frame_path(lab,c,f) for c in (19,20,21,22) for f in (974,977,979)})
    wrist=[r for r in regions['regions'] if r['segment']=='l_hand'];timing=[]
    for region in wrist:
        same=[r for r in records if r['camera']==region['camera'] and r['frame']==region['frame'] and r['segment']=='l_hand']
        timing.append(dict(camera=region['camera'],frame=region['frame'],
            compatible_offsets=[r['offset'] for r in same if r['inside']],
            adjacent_model_motion_px=[float(np.linalg.norm(np.array(same[j+1]['pixel'])-same[j]['pixel'])) for j in (0,1)],
            region_size_px=(np.array(region['box'][2:])-region['box'][:2]).tolist()))
    report=dict(records=records,summaries=summaries,timing_resolution=timing,unobservable=regions['unobservable'],
        source_sha256={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},
        scope=regions['scope'],refit_performed=False,adopted_offset=1,
        interpretation='Inside a broad region does not establish anatomical accuracy; all cameras were used in contributor alignment. Ball detections may overlap hands.')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summaries,indent=2));return report


if __name__=='__main__':run()
