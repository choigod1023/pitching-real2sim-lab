import unittest
import numpy as np
import mujoco
from coupled_payload import model_for
from baseline_calibration import scale_mass,fit_attachment,summarize

class TestBaselineCalibration(unittest.TestCase):
    def test_mass_scaling_preserves_ball_and_inertia_ratio(self):
        m=model_for([.3,.25],.001);ball=m.body('ball_payload').id;b=float(m.body_mass[ball]);old=m.body_mass.copy();oldi=m.body_inertia.copy()
        ratio=scale_mass(m,8.)
        self.assertAlmostEqual(m.body_mass.sum()-b,8.);self.assertEqual(m.body_mass[ball],b)
        for i in range(1,m.nbody):
            if i!=ball:
                self.assertAlmostEqual(m.body_mass[i]/old[i],ratio)
                np.testing.assert_allclose(m.body_inertia[i],oldi[i]*ratio)
    def test_attachment_fit_recovers_fixed_local_offsets(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="torso"><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>')
        q=np.array([[0,0,1,1,0,0,0],[1,2,3,np.cos(.4),0,0,np.sin(.4)]])
        local={'shoulder_jc':np.array([.2,.1,.5]),'glove_shoulder_jc':np.array([-.2,0,.5])};obs={n:[] for n in local};d=mujoco.MjData(m)
        for x in q:
            d.qpos[:]=x;mujoco.mj_forward(m,d)
            for n,p in local.items():obs[n].append(d.xpos[1]+d.xmat[1].reshape(3,3)@p)
        result=fit_attachment(m,q,obs)
        for n in local:np.testing.assert_allclose(result[n],local[n],atol=1e-12)

    def test_absolute_fit_does_not_hide_change_prediction_failure(self):
        rows=[]
        for mode in ['nominal','mass_only','frozen_geometry_mass']:
            for split,actual,pred in [('calibration',100,90),('reused_test',102,95)]:
                rows.append(dict(mode=mode,hz=2880,split=split,measured_speed_kmh=actual,simulated_speed_kmh=pred,absolute_speed_error_kmh=abs(actual-pred)))
        s=summarize(rows)['nominal']
        self.assertEqual(s['absolute_mae_kmh'],7);self.assertEqual(s['change_from_first_pitch_mae_kmh'],3);self.assertEqual(s['no_change_mae_kmh'],2)

if __name__=='__main__':unittest.main()
