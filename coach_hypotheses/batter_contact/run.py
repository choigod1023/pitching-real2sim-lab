"""Run observed response integration and a separate pitching-scene geometry demo."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

from ..temporal_quality import model as m
from .response import BatterResponse
from .scene import load_scene,run_geometry
from .render import export,success_bundle


def emit_video(path):
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    print('![투수–타자 시뮬레이션]('+str(Path(path).resolve()).replace('\\','/')+')',flush=True)


def check_response(source,previous,output):
    source,previous,output=map(Path,(source,previous,output))
    train=m.read_rows(source/'dev_train.json','2025-06-01','2025-07-15')[0]
    train+=m.read_rows(source/'dev_validation.json','2025-07-16','2025-07-31')[0]
    raw=m.read_rows(source/'final_test.json','2025-08-01','2025-08-31')[0]
    test,audit=m.eligible_test(m.swings(train),raw)
    old_predictions=json.loads((previous/'artifacts/temporal_quality/predictions.json').read_text(encoding='utf-8'))
    old_index={k:{(r['gamePk'],r['atBatIndex'],r['pitchIndex']):r for r in rr} for k,rr in old_predictions.items()}
    results={};predictions={};maximum=0.
    for name in ('GB0','GB1'):
        model=BatterResponse(previous/'coach_hypotheses/temporal_quality/results/summary.json',
                             previous/'artifacts/temporal_quality/fits',name)
        predicted=model.predict(test,data_source='mlb_observed');predictions[name]=predicted
        compared=0
        for r in predicted:
            key=tuple(r['key'])
            if key in old_index['swing']:
                maximum=max(maximum,abs(r['swing_probability']-old_index['swing'][key][name]));compared+=1
            for endpoint in m.TARGETS:
                if key in old_index[endpoint]:
                    maximum=max(maximum,abs(r[endpoint]['labelled_miss_given_swing']-old_index[endpoint][key][name]));compared+=1
            for endpoint in m.TARGETS:
                if abs(sum(r[endpoint]['branches'].values())-1)>1e-12:raise ValueError('joint_probability_incoherent')
        results[name]=dict(comparisons=compared)
    if maximum>1e-12:raise ValueError('restored_prediction_mismatch')
    diagnostics={}
    for endpoint in m.TARGETS:
        before=predictions['GB0'];after=predictions['GB1'];y=m.target(test,endpoint)
        rows=[dict(pitcherId=r['pitcherId'],y=float(yy),before=a[endpoint]['branches']['swing_labelled_miss'],
                   after=b[endpoint]['branches']['swing_labelled_miss']) for r,yy,a,b in zip(test,y,before,after)]
        metric,pp=m.metrics(rows,('before','after'))
        diagnostics[endpoint]=dict(metrics=metric,comparison=m.contrast('before','after',pp))
    result=dict(audit=audit,restored_model_comparisons=results,max_abs_reproduction_difference=maximum,
        joint_branch_sums_verified=True,diagnostics=diagnostics,
        scope='integration_diagnostic_on_previously_used_August_not_new_heldout',
        example=predictions['GB1'][0],physical_contact_probability=None,body_mapping_validated=False,
        video_dataset_is_same_MLB_pitch=False)
    m.save(output/'response_integration.json',result)
    return result


def run(obp_dir,external,previous,mlb_source,output,media):
    output,media=Path(output),Path(media)
    response=check_response(mlb_source,previous,output)
    scene,audit=load_scene(obp_dir,external,output/'source_checks')
    geometry=run_geometry(scene)
    video=export(scene,geometry,output/'pitcher_batter_overlay.mp4')
    # Only engineering integration is successful; no athlete-performance success.
    bundle=success_bundle(True,'engineering',[video])
    media.mkdir(parents=True,exist_ok=True)
    shared=media/'pitcher_batter_overlay.mp4';shutil.copy2(video['path'],shared)
    manifest=dict(source_audit=audit,geometry=geometry,response=response,video=video,success_bundle=bundle,
        shared_video=dict(path=str(shared.resolve()),sha256=m.digest(shared)),
        same_pitch_prediction_with_body_video=False,actual_athlete_improvement=False)
    m.save(Path(__file__).with_name('results')/'integration.json',manifest)
    m.save(Path(__file__).with_name('results')/'response_integration.json',response)
    emit_video(shared)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[2]
    parser.add_argument('--obp-dir',type=Path,default=root.parent/'openbiomechanics')
    parser.add_argument('--external',type=Path,default=root/'artifacts/batter_contact/external')
    parser.add_argument('--previous',type=Path,default=root.parent/'temporal-pitch-quality-20261030')
    parser.add_argument('--mlb-source',type=Path,default=root.parent/'temporal-collection-agent-20261030/artifacts/temporal_quality_games')
    parser.add_argument('--output',type=Path,default=root/'artifacts/batter_contact')
    parser.add_argument('--media',type=Path,default=Path(__file__).with_name('media'))
    args=parser.parse_args()
    run(args.obp_dir,args.external,args.previous,args.mlb_source,args.output,args.media)


if __name__=='__main__':main()
