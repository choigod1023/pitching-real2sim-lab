"""Reproduce the target audit; downloads optional, throwing video render separate."""
import argparse,hashlib,json,subprocess,sys,platform
import cv2,numpy,scipy
from pathlib import Path

def main(fetch=False):
    root=Path(__file__).resolve().parent;p=root/'artifacts/edge-calibration';p.mkdir(parents=True,exist_ok=True)
    def execute(label,args):
        r=subprocess.run([sys.executable,*args],cwd=root,capture_output=True,text=True)
        (p/f'{label}-execution.log').write_text(r.stdout+r.stderr,encoding='utf-8');r.check_returncode()
    if fetch:
        for kind in ['cube','groundplane','checkerboard']:execute('fetch-'+kind,['fetch_edge_targets.py','--kind',kind])
    execute('round1',['edge_calibration.py']);(p/'calibration-results-round1.json').write_bytes((p/'calibration-results.json').read_bytes())
    execute('round2-refine8',['refine_cube_axes.py','--window','8'])
    execute('round2-fit8',['edge_calibration.py','--annotations','edge_target_annotations_refined.json']);(p/'calibration-results-round2-window8.json').write_bytes((p/'calibration-results.json').read_bytes())
    execute('round3',['edge_calibration.py','--annotations','edge_target_annotations_plane_round3.json'])
    execute('checkerboard',['checkerboard_rig_audit.py']);execute('published-stereo',['published_stereo_audit.py'])
    reports={name:json.loads((p/file).read_text()) for name,file in {'cube_original':'calibration-results-round1.json','cube_refinement':'calibration-results-round2-window8.json','cube_terminal_review':'calibration-results.json','planar_board':'checkerboard-rig-results.json','published_stereo':'published-stereo-heldout-results.json'}.items()}
    s=reports['published_stereo'];b=reports['planar_board']
    summary={'stage':'data','scientific_stage_passed':False,'accepted_hand_ball_3d':False,'demonstrated_speed_error_reduction_kmh':None,
      'software':{'python':platform.python_version(),'opencv':cv2.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__},
      'inputs':{'static_video_count':12,'inventory':'edge_source_inventory.json','intrinsics':'edge_intrinsics_frozen.json'},
      'cube_original':reports['cube_original']['pose_checks'],'cube_image_refinement':reports['cube_refinement']['pose_checks'],
      'cube_terminal_review':{'plane_all_views':reports['cube_terminal_review']['plane_checks'][-1],'engineering_pass':reports['cube_terminal_review']['engineering_pass']},
      'planar_board_heldout_projection':b['checkerboard_projection'],
      'published_stereo':{'cube_edges':s['cube_edges'],'cube_reprojection':s['cube_reprojection'],'plane_lengths_mm':s['plane_lengths_mm'],'plane_reprojection_rms_px':s['plane_reprojection_rms_px'],'engineering_pass':s['engineering_pass'],'plane_metric_status':s['plane_metric_status']},
      'evidence':[{ 'path':str(f.relative_to(root)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(p.glob('*results*.json'))],
      'interpretation':'Conditional source/annotation/calibration inconsistency; not instrument error, not a speed confidence interval. Methods fail the frozen metric checks. No candidate posture recommendation.'}
    (root/'edge_calibration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'scientific_stage_passed':False,'cube_edge_range_mm':[min(e['mm'] for e in s['cube_edges']),max(e['mm'] for e in s['cube_edges'])],'heldout_board_rms_px':{i:v['heldout_rms_px'] for i,v in b['checkerboard_projection'].items()}}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fetch',action='store_true');main(p.parse_args().fetch)
