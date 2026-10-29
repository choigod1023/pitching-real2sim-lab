import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ezc3d
import numpy as np
from scipy.spatial.transform import Rotation

from .pilot import (load_c3d,rigid_transform,calibrate_wrist,dynamic_hand,fit_offset,
                    predict_marker,evaluate,run,main)


def source_fixture():
    n=401;time=(np.arange(n)+1)/200;translation=np.column_stack([.3*time,np.zeros((n,2))])
    locations={'RFR1':[-.05,0,1],'RFR2':[-.03,.04,1],'RFR3':[-.1,.01,1.03],
               'RFR4':[-.08,-.02,.98],'RWRL':[0,-.02,1],'RWRM':[0,.02,1],
               'R3MC':[.08,0,1],'R5MC':[.06,.03,1]}
    markers={label:np.array(point)+translation for label,point in locations.items()}
    markers['Ball']=translation+np.column_stack([.12+.15*(time-1.5),np.full(n,.02),np.ones(n)])
    return {'markers':markers,'valid':{label:np.ones(n,bool) for label in markers},'time':time,
            'rate':200.,'first':1,'last':n,'events':[{'label':'Event','time':1.5,'source_frame':300,'array_index':299}]}


def write_fixture(path):
    data=source_fixture();source=ezc3d.c3d();labels=list(data['markers'])
    source.add_parameter('POINT','LABELS',labels);source.add_parameter('POINT','RATE',200.)
    source.add_parameter('POINT','UNITS',['mm']);source.add_parameter('EVENT','LABELS',['Event'])
    source.add_parameter('EVENT','TIMES',np.array([[0.],[1.5]]))
    points=np.ones((4,len(labels),len(data['time'])))
    for i,label in enumerate(labels):points[:3,i,:]=data['markers'][label].T*1000
    source['data']['points']=points;source.write(str(path))


class ObservationTests(unittest.TestCase):
    def test_load_units_and_on_disk_event_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trial.c3d';write_fixture(path);source=load_c3d(path)
            self.assertEqual(source['events'][0]['array_index'],300-source['first'])
            np.testing.assert_allclose(source['markers']['Ball'],source_fixture()['markers']['Ball'],atol=1e-6)
            self.assertEqual(source['rate'],200.)
            with path.open('r+b') as f:f.seek(1);f.write(b'X')
            with self.assertRaises(ValueError):load_c3d(path)

    def test_rigid_fit_and_degenerate_cluster(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(4,3));q=Rotation.from_euler('xyz',[20,30,5],degrees=True).as_matrix();t=np.array([1.,2.,3.])
        rotation,translation,error=rigid_transform(x,(q@x.T).T+t)
        np.testing.assert_allclose(rotation,q,atol=1e-12);np.testing.assert_allclose(translation,t,atol=1e-12)
        self.assertLess(error,1e-12)
        with self.assertRaisesRegex(ValueError,'degenerate'):rigid_transform(np.zeros((4,3)),np.zeros((4,3)))

    def test_static_holdout_does_not_change_calibration(self):
        source=source_fixture();first=calibrate_wrist(source,'R');self.assertLess(first['held_wrist_rmse_mm'],1e-8)
        changed=copy.deepcopy(source)
        for label in ['RWRL','RWRM']:changed['markers'][label][first['fit_frames']:]+=np.array([.02,0,0])
        second=calibrate_wrist(changed,'R')
        np.testing.assert_allclose(first['wrist_local'],second['wrist_local'])
        self.assertAlmostEqual(second['held_wrist_rmse_mm'],20.)

    def test_dynamic_hand_frame_and_observed_relative_motion(self):
        source=source_fixture();cal=calibrate_wrist(source,'R');w,r,valid,residual=dynamic_hand(source,cal,'R')
        self.assertTrue(valid.all());np.testing.assert_allclose(np.linalg.det(r),1.,atol=1e-12)
        indices=np.arange(280,290);time=source['time']-1.5
        coefficient=fit_offset(time,w,r,source['markers']['Ball'],indices,True)
        np.testing.assert_allclose(coefficient[1],[.15,0,0],atol=1e-10)
        prediction=predict_marker(time,w,r,coefficient)
        np.testing.assert_allclose(prediction,source['markers']['Ball'],atol=1e-10)

    def test_linear_surface_marker_recovered_and_no_held_target_leak(self):
        source=source_fixture();cal=calibrate_wrist(source,'R');metrics,predictions=evaluate(source,cal,'R')
        self.assertGreater(metrics[0]['marker_position_rmse_mm'],1.)
        self.assertLess(metrics[1]['marker_position_rmse_mm'],1e-8)
        self.assertLess(metrics[1]['marker_speed_mae_kmh'],1e-8)
        changed=copy.deepcopy(source);t=changed['time']-1.5;changed['markers']['Ball'][(t>-.041)&(t<-.001)]+=1.
        _,after=evaluate(changed,cal,'R')
        for a,b in zip(predictions,after):
            for key in ['predicted_x_m','predicted_y_m','predicted_z_m']:self.assertEqual(a[key],b[key])

    def test_global_rigid_transform_preserves_errors(self):
        source=source_fixture();q=Rotation.from_euler('xyz',[20,40,10],degrees=True).as_matrix()
        transformed=copy.deepcopy(source)
        transformed['markers']={label:(q@points.T).T+[2.,1.,-3.] for label,points in source['markers'].items()}
        first,_=evaluate(source,calibrate_wrist(source,'R'),'R')
        second,_=evaluate(transformed,calibrate_wrist(transformed,'R'),'R')
        for a,b in zip(first,second):
            for key in ['marker_position_rmse_mm','marker_speed_mae_kmh','marker_velocity_vector_rmse_kmh']:self.assertAlmostEqual(a[key],b[key],places=8)

    def test_missing_observation_rejected_and_shift_kept_separate(self):
        source=source_fixture();cal=calibrate_wrist(source,'R');shifted,_=evaluate(source,cal,'R',1)
        self.assertEqual(shifted[0]['event_shift_frames'],1)
        source['valid']['Ball'][295]=False
        with self.assertRaisesRegex(ValueError,'missing_observations'):evaluate(source,cal,'R')

    def test_real_file_orchestration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_fixture(p/'trial.c3d');write_fixture(p/'static.c3d')
            with patch('coach_hypotheses.observed_release_pilot.pilot.CASES',[('trial.c3d','static.c3d','R')]):result=run(p,p/'out')
            self.assertEqual(len(result['rows']),4);self.assertEqual(result['exclusions'],[])
            self.assertFalse(result['radar_comparison_performed']);self.assertTrue((p/'out/metrics.csv').is_file())

    def test_cli_dispatch_and_path_guard(self):
        with patch('sys.argv',['pilot','--data','example']),patch('coach_hypotheses.observed_release_pilot.pilot.run',return_value={'rows':[],'exclusions':[]}) as mocked,patch('builtins.print'):
            main();mocked.assert_called_once()
        with patch('sys.argv',['pilot','--data','example','--output',str(Path.home())]):
            with self.assertRaises(SystemExit):main()


if __name__=='__main__':unittest.main()
