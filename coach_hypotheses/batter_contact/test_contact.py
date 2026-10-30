import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageFont

from . import geometry as g,response as r,scene as s,render as v,run as app
from ..temporal_quality import model as m
from ..temporal_quality.test_model import rows


def track(moving=True):
    time=np.linspace(0,1,101)
    sweet=np.column_stack([np.zeros_like(time),20*(time-.5) if moving else np.zeros_like(time),np.ones_like(time)])
    return dict(time=time,sweet=sweet,grip=sweet-[0,.8,0],contact=.5)


def tiny_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
    for n in range(3):writer.write(np.full((64,64,3),n*50,dtype=np.uint8))
    writer.release()


class GeometryTests(unittest.TestCase):
    def test_segment_analytic_endpoints_zero_and_rigid_invariance(self):
        p=np.array([[0,1,0],[3,0,0]]);a=np.array([0,0,0]);b=np.array([2,0,0])
        np.testing.assert_allclose(g.point_segment_distance(p,a,b),[1,1])
        self.assertAlmostEqual(g.point_segment_distance([0,3,4],a,a),5)
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1]]);offset=[4,5,6]
        np.testing.assert_allclose(g.point_segment_distance(p@rot.T+offset,a@rot.T+offset,b@rot.T+offset),[1,1])
        with self.assertRaises(ValueError):g.point_segment_distance([np.nan,0,0],a,b)
        with self.assertRaises(ValueError):g.point_segment_distance([0,0],a,b)

    def test_projectile_fixed_speed_target_and_unreachable(self):
        velocity,t=g.ballistic_initial([18,0,2],[0,0,1],40)
        self.assertAlmostEqual(np.linalg.norm(velocity),40)
        np.testing.assert_allclose(g.ball_positions([18,0,2],velocity,t),[0,0,1],atol=1e-12)
        with self.assertRaises(ValueError):g.ballistic_initial([0,0,0],[100,0,100],1)
        with self.assertRaises(ValueError):g.ballistic_initial([0,0,0],[0,0,0],40)
        with self.assertRaises(ValueError):g.ball_positions([0,0,0],velocity,-1)
        with self.assertRaises(ValueError):g.ball_positions([np.nan,0,0],velocity,1)

    def test_positive_delay_means_earlier_measured_swing_state(self):
        tr=track();_,ss,_=g.bat_state(tr,.2,.2,12)
        self.assertAlmostEqual(ss[1],-.24)
        with self.assertRaises(ValueError):g.bat_state(tr,-1,.2,0)
        broken=track();broken['grip']=broken['sweet'].copy()
        with self.assertRaises(ValueError):g.bat_state(broken,.2,.2,0)

    def test_conditional_geometry_and_resolution_not_real_whiff(self):
        tr=track();origin=[2,0,1];velocity,t=g.ballistic_initial(origin,[0,0,1],20)
        baseline=g.proximity(tr,origin,velocity,t,0);delayed=g.proximity(tr,origin,velocity,t,12)
        self.assertTrue(baseline['sweet_zone_near']);self.assertFalse(delayed['sweet_zone_near'])
        self.assertFalse(baseline['measured_contact_or_miss'])
        fine=g.proximity(tr,origin,velocity,t,12,step=.00005)
        self.assertLess(abs(delayed['minimum_clearance_mm']-fine['minimum_clearance_mm']),.1)
        with self.assertRaises(ValueError):g.proximity(tr,origin,velocity,t,0,step=0)
        with self.assertRaises(ValueError):g.proximity(tr,origin,velocity,t,2000)

    def test_sweep_boundaries_and_all_engineering_conditions(self):
        tr=track(False);origin=[2,0,1];velocity,t=g.ballistic_initial(origin,[0,0,1],20)
        sweep=g.sweep(tr,origin,velocity,t)
        self.assertEqual(sweep['grid_near_intervals_ms'],[[-20,20]]);self.assertTrue(sweep['touches_sweep_boundary'])
        result=s.run_geometry(dict(bat=tr,release=origin,velocity=velocity,arrival=t))
        self.assertEqual(len(result['convergence']),3);self.assertEqual(len(result['engineering_sensitivity']),9)
        self.assertFalse(result['matched_player_validation'])

    def test_source_selection_canonical_grid_and_archive_guard(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.csv';df=pd.DataFrame(dict(id=['a','a','b'],time=[0,round(1/360,4),0],x=[1,2,3]))
            df.to_csv(path,index=False);selected=s.selected_table(path,'id','a')
            self.assertEqual(len(selected),2);np.testing.assert_allclose(s.canonical_time(selected),[0,1/360])
            z=Path(d)/'data.zip'
            with zipfile.ZipFile(z,'w') as zz:zz.write(path,'landmarks.csv')
            self.assertEqual(len(s.selected_table(z,'id','a',True)),2)
            with self.assertRaises(ValueError):s.selected_table(path,'id','missing')
            with self.assertRaises(ValueError):s.canonical_time(pd.DataFrame(dict(time=[0,.1])))
            ext=Path(d)/'external';ext.mkdir();(ext/'hitting_landmarks.zip').write_bytes(b'bad')
            with self.assertRaises(ValueError):s.load_scene(d,ext,Path(d)/'out')

    def test_hitter_selection_uses_numeric_identity_not_file_order(self):
        metadata=pd.DataFrame(dict(user=['000005','000004','000003'],session=['000006','000103','000001'],
            session_swing=['6_1','103_1','1_1'],hitter_side=['R','R','L'],bat_length_in=[34,34,34]))
        self.assertEqual(s.choose_hitter(metadata).session_swing,'103_1')
        with self.assertRaises(ValueError):s.choose_hitter(metadata.iloc[2:])


class ResponseTests(unittest.TestCase):
    def test_joint_branches_coherent_and_not_named_contact(self):
        branches=r.joint_branches([0,.5,1],[.8,.2,.4])
        np.testing.assert_allclose(sum(branches.values()),1)
        self.assertNotIn('contact',branches);self.assertAlmostEqual(branches['swing_labelled_miss'][1],.1)
        with self.assertRaises(ValueError):r.joint_branches(1.1,.5)

    def test_saved_model_restore_and_cross_dataset_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);summary=dict(analyses={});rr=rows(n=150);original={}
            for endpoint in ('swing',)+m.TARGETS:
                train=rr if endpoint=='swing' else m.swings(rr)
                fitted=m.Logistic().fit(train,endpoint,'GB1',10);original[endpoint]=fitted
                path=root/'fits'/endpoint/f'{endpoint}_GB1_10.json';m.save(path,fitted.audit)
                summary['analyses'][endpoint]=dict(fits=[dict(model='GB1',success=True,fit_audit_sha256=m.digest(path))])
            source=root/'summary.json';m.save(source,summary)
            restored=r.BatterResponse(source,root/'fits')
            predictions=restored.predict(rr[:7],data_source='mlb_observed')
            np.testing.assert_allclose([p['swing_probability'] for p in predictions],original['swing'].predict(rr[:7])[0],atol=1e-15)
            for endpoint in m.TARGETS:
                np.testing.assert_allclose([p[endpoint]['labelled_miss_given_swing'] for p in predictions],original[endpoint].predict(rr[:7])[0],atol=1e-15)
            self.assertIsNone(predictions[0]['physical_contact_probability'])
            with self.assertRaises(ValueError):restored.predict(rr[:1],data_source='obp')
            with self.assertRaises(ValueError):restored.predict([],data_source='mlb_observed')
            with self.assertRaises(ValueError):restored.predict([dict(rr[0],pre_count={})],data_source='mlb_observed')
            with self.assertRaises(ValueError):r.BatterResponse(source,root/'fits','unsupported')
            path=root/'fits/swing/swing_GB1_10.json';path.write_text('{}')
            with self.assertRaises(ValueError):r.restore(path,'wrong')

    def test_response_integration_matches_pitch_keys_and_rejects_changed_prediction(self):
        train=rows(n=180);valid=rows('2025-07-20',100,n=80);test=rows('2025-08-10',200,n=80)
        predicted=[dict(key=[r['gamePk'],r['atBatIndex'],r['pitchIndex']],swing_probability=.5,
            **{k:dict(labelled_miss_given_swing=.25,branches=dict(take=.5,swing_labelled_miss=.125,swing_other=.375))
               for k in m.TARGETS}) for r in test]
        reference={k:[dict(gamePk=r['gamePk'],atBatIndex=r['atBatIndex'],pitchIndex=r['pitchIndex'],
                    GB0=.5 if k=='swing' else .25,GB1=.5 if k=='swing' else .25) for r in test]
                   for k in ('swing',)+m.TARGETS}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'previous/artifacts/temporal_quality/predictions.json';m.save(path,reference)
            for corrupt in (False,True):
                if corrupt:reference[m.TARGETS[0]][0]['GB1']=.75;m.save(path,reference)
                with patch.object(app.m,'read_rows',side_effect=[(train,{}),(valid,{}),(test,{})]),\
                     patch.object(app,'BatterResponse',return_value=SimpleNamespace(predict=lambda rr,**kw:predicted)):
                    if corrupt:
                        with self.assertRaises(ValueError):app.check_response(root,root/'previous',root/'out')
                    else:
                        result=app.check_response(root,root/'previous',root/'out')
                        self.assertEqual(result['max_abs_reproduction_difference'],0)
                        self.assertFalse(result['body_mapping_validated'])


class MediaTests(unittest.TestCase):
    def test_projector_and_pitching_frame(self):
        for view in ('field','pitcher','hitter','contact'):
            self.assertEqual(v.project([[0,0,0],[1,1,1]],view).shape,(2,2))
        with self.assertRaises(ValueError):v.project([0,0,0],'unknown')
        tr=track();velocity,arrival=g.ballistic_initial([2,0,1],[0,0,1],20)
        scene=dict(br=.5,pitch_time=np.array([0,1,2]),pitch_xyz=np.zeros((3,len(s.JOINTS),3)),
             hitter_time=tr['time'],hitter_xyz=np.zeros((101,len(s.HITTER_JOINTS),3)),bat=tr,
             release=[2,0,1],velocity=velocity,arrival=arrival,speed_kmh=72.,hitter_id='fixture',swing_id='fixture')
        fonts={size:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',size) for size in (17,18,20,22,23,30)}
        geometry=dict(nominal=g.sweep(tr,[2,0,1],velocity,arrival))
        image=v.frame(scene,arrival,geometry,fonts);self.assertEqual(image.size,v.SIZE)

    def test_full_decode_and_success_video_policy(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tiny.mp4';tiny_video(path)
            info=v.validate_video(path,Path(d)/'frames');self.assertEqual(info['decoded_frames'],3)
            self.assertEqual(len(info['representative_frames']),3)
            with self.assertRaises(ValueError):v.success_bundle(True,'engineering',[])
            with self.assertRaises(ValueError):v.success_bundle(True,'engineering',[info])
            info['contains_pitcher_scene']=True
            result=v.success_bundle(True,'engineering',[info]);self.assertIn(str(path).replace('\\','/'),result['inline_markdown'][0])
            with self.assertRaises(ValueError):v.success_bundle(True,'athlete_candidate',[info],athlete_evidence={'same_force':True})
            with self.assertRaises(ValueError):v.success_bundle(True,'athlete_candidate',[info],athlete_evidence={
                k:True for k in ('same_pitch_baseline','heldout_validation','same_force','same_work','body_change_validated')})
            with self.assertRaises(ValueError):v.success_bundle(True,'unknown',[info])
            with self.assertRaises(ValueError):v.validate_video(Path(d)/'missing.mp4')

    def test_export_stream_and_main_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            tr=track();scene=dict(arrival=.2,bat=tr,hitter_time=tr['time'])
            with patch.object(v,'SIZE',(64,64)),patch.object(v,'FPS',5),patch.object(v,'frame',return_value=Image.new('RGB',(64,64))):
                info=v.export(scene,{},Path(d)/'export.mp4')
            self.assertGreater(info['decoded_frames'],1);self.assertTrue(info['contains_pitcher_scene'])
        with patch('sys.argv',['run']),patch.object(app,'run') as called:
            app.main();self.assertEqual(len(called.call_args[0]),6)
            self.assertFalse(called.call_args.kwargs['show_demo'])
        with patch('sys.argv',['run','--show-demo']),patch.object(app,'run') as called:
            app.main();self.assertTrue(called.call_args.kwargs['show_demo'])

    def test_video_markdown_supports_unicode_console(self):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):app.emit_video('example.mp4')
        self.assertIn('example.mp4',stream.getvalue());self.assertIn('![',stream.getvalue())
        buffer=io.BytesIO();encoded=io.TextIOWrapper(buffer,encoding='cp949')
        with contextlib.redirect_stdout(encoded):app.emit_video('example.mp4')
        encoded.flush();self.assertEqual(encoded.encoding,'utf-8')
        self.assertIn('투수–타자',buffer.getvalue().decode('utf-8'));encoded.close()

    def test_run_preserves_scope_and_does_not_claim_athlete_success(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);path=root/'source.mp4';tiny_video(path)
            info=dict(path=str(path),contains_pitcher_scene=True)
            with patch.object(app,'check_response',return_value={}),patch.object(app,'load_scene',return_value=({},{})),\
                 patch.object(app,'run_geometry',return_value={}),patch.object(app,'export',return_value=info),\
                 patch.object(app.m,'save'):
                result=app.run(root,root,root,root,root/'out',root/'media',show_demo=True)
            self.assertFalse(result['actual_athlete_improvement']);self.assertFalse(result['same_pitch_prediction_with_body_video'])

    def test_default_does_not_substitute_demo_for_pitcher_improvement(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout,patch.object(app.m,'save') as save,\
             patch.object(app,'check_response') as response,patch.object(app,'load_scene') as scene,\
             patch.object(app,'export') as export,patch.object(app,'emit_video') as emit:
            result=app.run('unused','unused','unused','unused','unused','unused')
        self.assertIsNone(result['pitcher_performance_video'])
        self.assertFalse(result['pitcher_performance_improvement_verified'])
        self.assertFalse(result['statistical_prediction_gain_is_pitcher_improvement'])
        response.assert_not_called();scene.assert_not_called();export.assert_not_called();emit.assert_not_called()
        self.assertNotIn('![',stdout.getvalue());save.assert_called_once()


if __name__=='__main__':unittest.main()
