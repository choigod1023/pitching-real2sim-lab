from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ezc3d
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline

from .audit import center_axis_frames,fixed_wrist_position,load_source,evaluate_source,run,main
from speed_path import retarget_velocity


def write_synthetic_data(root):
    data=Path(root)/'data';(data/'c3d').mkdir(parents=True);(data/'full_sig').mkdir();(data/'poi').mkdir()
    model=Path(root)/'code/v3d/model';model.mkdir(parents=True);(model/'v6_model_hybrid_lm.mdh').write_text('synthetic_test_model',encoding='utf-8')
    t=np.arange(721)/360;shoulder=np.column_stack([.2*t,np.zeros(len(t)),np.full(len(t),1.4)])
    elbow=shoulder+[.3,0,-.1];wrist=elbow+[.25,0,-.15]
    x=np.column_stack([np.cos(t),np.sin(t),np.zeros(len(t))]);y=np.column_stack([-np.sin(t),np.cos(t),np.zeros(len(t))])
    hand=wrist+.08*x;markers=[wrist+.09*x,wrist+.025*y,wrist-.025*y]
    metadata=[];tables=[];poi=[]
    for i in range(1,6):
        pitch=f'3034_{i}';filename=f'000002_003034_73_207_{i:03d}_FF_{800+i}.c3d'
        c=ezc3d.c3d();c.add_parameter('POINT','LABELS',['RFIN','RWRA','RWRB']);c.add_parameter('POINT','RATE',360.);c.add_parameter('POINT','UNITS',['m'])
        points=np.ones((4,3,len(t)))
        for j,mark in enumerate(markers):points[:3,j,:]=mark.T
        c['data']['points']=points;c.write(str(data/'c3d'/filename))
        metadata.append({'session_pitch':pitch,'filename_new':filename,'pitch_speed_mph':80+i/10})
        poi.append({'session_pitch':pitch,'pitch_speed_mph':80+i/10})
        row={'session_pitch':pitch,'time':np.round(t,4),'fp_10_time':1.,'BR_time':1.6}
        for name,point in zip(['shoulder','elbow','wrist','hand'],[shoulder,elbow,wrist,hand]):
            row.update({f'{name}_jc_{axis}':point[:,j] for j,axis in enumerate('xyz')})
        tables.append(pd.DataFrame(row))
    pd.DataFrame(metadata).to_csv(data/'metadata.csv',index=False);pd.DataFrame(poi).to_csv(data/'poi/poi_metrics.csv',index=False)
    pd.concat(tables).to_csv(data/'full_sig/landmarks.csv',index=False)
    return data


class GapRepairTests(unittest.TestCase):
    def test_axes_are_proper_and_reject_degenerate_inputs(self):
        frames=center_axis_frames(np.array([[1.,0,0],[0,1.,0]]),np.array([[0,1.,0],[0,0,1.]]))
        np.testing.assert_allclose(np.linalg.det(frames),1.)
        with self.assertRaisesRegex(ValueError,'degenerate'):center_axis_frames(np.zeros((2,3)),np.ones((2,3)))
        with self.assertRaises(ValueError):center_axis_frames(np.ones((2,3)),np.full((2,3),np.nan))

    def test_fixed_wrist_position_derivative_matches_analytic_velocity(self):
        def points(t):return np.array([[.2*t,0,1.],[.2*t+np.cos(t),np.sin(t),1.],[.2*t+np.cos(t),np.sin(t),.5]])
        t=.3;eps=1e-5;p=points(t);v=(points(t+eps)-points(t-eps))/(2*eps);lengths=np.array([.8,.4])
        numerical=(fixed_wrist_position(points(t+eps),lengths)-fixed_wrist_position(points(t-eps),lengths))/(2*eps)
        np.testing.assert_allclose(numerical,retarget_velocity(p,v,lengths),atol=1e-8)
        with self.assertRaises(ValueError):fixed_wrist_position(np.zeros((3,3)),lengths)

    def test_loader_never_requires_radar_and_exact_path_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=write_synthetic_data(tmp);metadata=pd.read_csv(data/'metadata.csv').drop(columns='pitch_speed_mph');metadata.to_csv(data/'metadata.csv',index=False)
            source=load_source(data,'3034_1');rows,geometry=evaluate_source(source)
            self.assertEqual(len(rows),6);self.assertEqual(source['br'],1.6)
            self.assertLess(max(max(row['position_derivative_difference_mps']) for row in rows),1e-5)
            self.assertLess(geometry['axis_angle_difference_deg'],1.)

    def test_global_coordinates_rotation_does_not_change_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=load_source(write_synthetic_data(tmp),'3034_1');rows,_=evaluate_source(source)
            q=Rotation.from_euler('xyz',[20,10,30],degrees=True).as_matrix();t=source['time'];old=source['curve'](t)
            source['curve']=CubicSpline(t,np.einsum('ij,nkj->nki',q,old)+[1.,2.,3.])
            source['mid']=CubicSpline(t,source['mid'](t)@q.T+[1.,2.,3.]);source['fin']=CubicSpline(t,source['fin'](t)@q.T+[1.,2.,3.])
            for frame in ['surface','center_axis']:source[frame]=RotationSpline(t,Rotation.from_matrix(np.einsum('ij,njk->nik',q,source[frame](t).as_matrix())))
            after,_=evaluate_source(source)
            np.testing.assert_allclose([r['assumed_ball_kmh'] for r in rows],[r['assumed_ball_kmh'] for r in after],atol=1e-7)

    def test_full_csv_pipeline_preserves_source_and_detects_identity_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=write_synthetic_data(tmp);source=(data/'metadata.csv').read_bytes();out=Path(tmp)/'results';result=run(data,out)
            self.assertEqual(len(result['summary']),6);self.assertFalse(result['new_core_adopted']);self.assertEqual(source,(data/'metadata.csv').read_bytes())
            self.assertTrue((out/'paths.csv').is_file());self.assertLess(max(r['speed_difference_kmh'] for r in result['regression']),1e-7)
            poi=pd.read_csv(data/'poi/poi_metrics.csv');poi.loc[0,'pitch_speed_mph']+=1;poi.to_csv(data/'poi/poi_metrics.csv',index=False)
            with self.assertRaisesRegex(ValueError,'radar_key_mismatch'):run(data,Path(tmp)/'failed')

    def test_cli_dispatch_and_output_guard(self):
        with patch('sys.argv',['audit','--data','example']),patch('coach_hypotheses.large_gap_repair.audit.run',return_value={'summary':[]}) as mocked,patch('builtins.print'):
            main();mocked.assert_called_once()
        with patch('sys.argv',['audit','--data','example','--output',str(Path.home())]):
            with self.assertRaises(SystemExit):main()


if __name__=='__main__':unittest.main()
