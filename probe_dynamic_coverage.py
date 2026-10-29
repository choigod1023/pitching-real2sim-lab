"""Checkpointed four-camera board visibility probe; permits bounded resume batches."""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import cv2
from dynamic_camera_recovery import ROOT,detect

def frame_schedule(frame_count,start,stop,stride):
    if stride<=0 or start<0 or stop<start:raise ValueError('Invalid frame interval')
    return list(range(start,min(stop+1,frame_count),stride))

def run(max_new=None,camera_ids=(1,2,3,4),start=0,stop=1320,stride=24,input_root=None):
    if not camera_ids or any(i not in [1,2,3,4] for i in camera_ids) or len(set(camera_ids))!=len(camera_ids):raise ValueError('Invalid or duplicate camera IDs')
    cv2.setNumThreads(1);folder=ROOT/'coverage';folder.mkdir(exist_ok=True);manifest=json.loads((ROOT/'dynamic-source-manifest.json').read_text());created=0;code=hashlib.sha256(Path(__file__).read_bytes()+Path('dynamic_camera_recovery.py').read_bytes()).hexdigest()
    # Old observations remain source data, with their original implementation hash retained.
    previous={}
    for i in [1,4]:
        f=ROOT/f'detections_camera{i}.json'
        if f.exists():previous[i]=json.loads(f.read_text())
    attempted={};detections={}
    for i in camera_ids:
        entry=next(e for e in manifest if e['name'].endswith(f'_{i}.mp4'));cap=cv2.VideoCapture(str((Path(input_root) if input_root else ROOT)/entry['name']));attempted[i]=[];detections[i]=[]
        for f in frame_schedule(entry['frame_count'],start,stop,stride):
            path=folder/f'cam{i}_frame{f}.json'
            if path.exists():r=json.loads(path.read_text())
            elif str(f) in previous.get(i,{}):
                old=previous[i][str(f)];r={'camera':i,'frame':f,'video_sha256':entry['sha256'],'implementation_sha256':old['implementation_sha256'],'reused_recorded_observation':True,'methods':{m:old['methods'][m] for m in ['source720_subpix5','native_sb']}};path.write_text(json.dumps(r))
            else:
                if max_new is not None and created>=max_new:continue
                cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read()
                if not ok:raise ValueError(f'Decode failed cam{i} frame{f}')
                g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);r={'camera':i,'frame':f,'video_sha256':entry['sha256'],'implementation_sha256':code,'methods':{}}
                for m in ['source720_subpix5','native_sb']:
                    t=time.perf_counter();q,info=detect(g,m);r['methods'][m]={'points':None if q is None else q.tolist(),'info':info,'seconds':time.perf_counter()-t}
                path.write_text(json.dumps(r));created+=1
            if r['video_sha256']!=entry['sha256']:raise ValueError('Source changed; coverage record cannot be reused')
            attempted[i].append(f)
            if any(v['points'] is not None for v in r['methods'].values()):detections[i].append(f)
        cap.release()
    pairs={f'{i}-{j}':sorted(set(detections[i])&set(detections[j])) for i,j in itertools.combinations(camera_ids,2)}
    expected=sum(len(frame_schedule(e['frame_count'],start,stop,stride)) for e in manifest if any(e['name'].endswith(f'_{i}.mp4') for i in camera_ids));done=sum(len(v) for v in attempted.values());result={'complete':done==expected,'configuration':{'cameras':list(camera_ids),'start':start,'stop':stop,'stride':stride},'attempted':attempted,'detected_by_any_method':detections,'same_index_pair_overlap':pairs,'completed_camera_frames':done,'expected_camera_frames':expected,'new_decodes_this_run':created,'note':'Same-index overlap is not verified physical synchronization. Partial work is not absence of overlap.'}
    filename='coverage-progress.json' if tuple(camera_ids)==(1,2,3,4) and (start,stop,stride)==(0,1320,24) else f"coverage-{'-'.join(map(str,camera_ids))}-{start}-{stop}-s{stride}.json"
    (ROOT/filename).write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['complete','completed_camera_frames','expected_camera_frames','new_decodes_this_run']}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-new',type=int);p.add_argument('--cameras',type=int,nargs='+',default=[1,2,3,4]);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=1320);p.add_argument('--stride',type=int,default=24);p.add_argument('--input-root',type=Path);a=p.parse_args();run(a.max_new,a.cameras,a.start,a.stop,a.stride,a.input_root)
