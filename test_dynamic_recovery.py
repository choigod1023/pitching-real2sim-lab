import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import dynamic_camera_recovery as recovery
from fetch_dynamic_targets import confirmation
from probe_dynamic_coverage import frame_schedule

class RecoveryTests(unittest.TestCase):
    def test_good_reprojection_cannot_hide_bad_heldout_length(self):
        good={'reprojection_rms_px':.01,'spacing_max_abs_error_percent':.2,'min_depth_mm':3000.};self.assertTrue(recovery.board_screen([good]));self.assertFalse(recovery.board_screen([dict(good,spacing_max_abs_error_percent=2.)]));self.assertFalse(recovery.board_screen([]))
    def test_frame_schedule_never_reads_frame_count_as_an_index(self):
        frames=frame_schedule(1320,0,1320,1);self.assertEqual(len(frames),1320);self.assertEqual(frames[-1],1319)
        for args in [(1320,0,10,0),(1320,-1,10,1),(1320,20,10,1)]:
            with self.assertRaises(ValueError):frame_schedule(*args)
    def test_public_confirmation_requires_same_file_and_host(self):
        html='<form id="download-form" action="https://drive.usercontent.google.com/download"><input name="id" value="public-file"><input name="export" value="download"><input name="confirm" value="t"></form>'
        url,fields=confirmation(html,'public-file');self.assertEqual(fields['id'],'public-file')
        for wrong in [html.replace('google.com','other.example'),html.replace('public-file','another-file')]:
            with self.assertRaises(ValueError):confirmation(wrong,'public-file')
    def test_grid_order_is_matched_in_the_same_image(self):
        anchor=np.array([[100+x*30.,80+y*24.] for y in range(4) for x in range(7)])
        for name,q in recovery.grid_variants(anchor).items():
            aligned,variant,error=recovery.align_same_image(q,anchor);np.testing.assert_allclose(aligned,anchor);self.assertEqual(error,0.)
    def test_insufficient_matched_observations_do_not_become_pass(self):
        rows={'0':{'methods':{m:{'points':None} for m in ['source720_subpix5','native_sb','roi4_sb']}}}
        with tempfile.TemporaryDirectory() as td,patch.object(recovery,'ROOT',Path(td)):
            r=recovery.run_fit(rows,rows);self.assertFalse(r['scientific_stage_passed']);self.assertEqual(r['status'],'insufficient_matched_detections');self.assertEqual(r['split']['common_frames'],[])
    def test_synthetic_pair_fit_predicts_heldout_board_metric(self):
        k=np.array([[1200.,0,640],[0,1200,512],[0,0,1]]);d=np.zeros(5);rotation=np.eye(3);t=np.array([-500.,0,0]);da={};db={}
        import cv2
        for i in range(10):
            rv=np.array([.02*i,.015*i,.005*i]);rr=cv2.Rodrigues(rv)[0];world=recovery.OBJ@rr.T+np.array([i*5.,i*3.,3000+i*20.]);da[i]=cv2.projectPoints(world,np.zeros(3),np.zeros(3),k,d)[0].reshape(-1,2);db[i]=cv2.projectPoints(world,np.zeros(3),t,k,d)[0].reshape(-1,2)
        rms,r,tt=recovery.stereo(list(range(8)),da,db,k,d,k,d);self.assertLess(rms,1e-3);np.testing.assert_allclose(tt,t,atol=.01)
        cams=[{'K':k,'dist':d,'R':np.eye(3),'t':np.zeros(3)},{'K':k,'dist':d,'R':r,'t':tt}];metrics=recovery.board_metrics([8,9],da,db,cams);self.assertLess(max(x['spacing_mae_mm'] for x in metrics),.001)
if __name__=='__main__':unittest.main()
