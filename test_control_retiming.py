import copy
import json
import unittest
from pathlib import Path
import mujoco
import numpy as np
from control_retiming import retime, groups, budget, mechanics, OUT
from trace_error_audit import integrate_inputs


class TestControlRetiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><freejoint/><geom type="sphere" size=".1" mass="1"/>
            <body><joint name="torso_rotation" type="ball"/><geom type="sphere" size=".08" mass="1"/>
              <body><joint name="lead_knee"/><geom type="sphere" size=".06" mass="1"/>
                <body><joint name="rear_knee"/><geom type="sphere" size=".04" mass="1"/>
                  <site name="ball_center" pos="0 0 .15"/>
                </body>
              </body>
            </body>
          </body></worldbody></mujoco>''')
        cls.source=np.linspace(-.02,.14,81)
        cls.loads=np.array([2+cls.source*(i+1) for i in range(cls.m.nv)]).T
        cls.times=np.linspace(0,.12,61)

    def test_zero_factors_preserve_all_controls(self):
        loads=retime(self.m,self.source,self.loads,self.times,(0,0,0))
        np.testing.assert_allclose(loads,2+self.times[:,None]*np.arange(1,self.m.nv+1),atol=1e-14)

    def test_initial_control_and_unshifted_root_are_preserved(self):
        a=retime(self.m,self.source,self.loads,self.times,(0,0,0))
        b=retime(self.m,self.source,self.loads,self.times,(1,1,1))
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_array_equal(a[:,:6],b[:,:6])

    def test_advancing_positive_ramp_samples_later_input(self):
        a=retime(self.m,self.source,self.loads,self.times,(0,0,0))
        b=retime(self.m,self.source,self.loads,self.times,(1,0,1))
        self.assertTrue(np.all(b[-1,groups(self.m)['lead_knee']]>a[-1,groups(self.m)['lead_knee']]))
        self.assertTrue(np.all(b[-1,groups(self.m)['rear_knee']]<a[-1,groups(self.m)['rear_knee']]))

    def test_interpolation_stays_within_source_amplitude_bounds(self):
        a=retime(self.m,self.source,self.loads,self.times,(-1,1,-1))
        self.assertTrue(np.all(a<=self.loads.max(0)+1e-14))
        self.assertTrue(np.all(a>=self.loads.min(0)-1e-14))

    def test_extrapolation_and_invalid_inputs_fail(self):
        with self.assertRaises(ValueError): retime(self.m,self.source,self.loads,np.array([.2]),(1,0,0))
        with self.assertRaises(ValueError): retime(self.m,self.source,self.loads,self.times,(1,))
        with self.assertRaises(ValueError): retime(self.m,self.source[::-1],self.loads,self.times,(1,0,0))

    def test_same_force_does_not_mean_same_work(self):
        a={'positive_j':2.,'negative_j':1.,'costs':{'external_force':{'positive_j':2.,'negative_j':1.,'peak':10.}}}
        b=copy.deepcopy(a);b['costs']['external_force']['positive_j']+=.1
        self.assertTrue(budget(a,b,True))

    def test_redistribution_does_not_relax_arm_work(self):
        a={'positive_j':2.,'negative_j':1.,'costs':{'pitch_shoulder':{'positive_j':2.,'negative_j':1.,'peak':10.}}}
        b=copy.deepcopy(a);b['costs']['pitch_shoulder']['positive_j']+=.1
        self.assertTrue(budget(a,b,True))

    def test_leg_redistribution_requires_total_cap(self):
        a={'positive_j':4.,'negative_j':1.,'costs':{'lead_knee':{'positive_j':2.,'negative_j':.5,'peak':10.},'rear_knee':{'positive_j':2.,'negative_j':.5,'peak':10.}}}
        b=copy.deepcopy(a);b['costs']['lead_knee']['positive_j']+=1;b['costs']['rear_knee']['positive_j']-=1
        self.assertTrue(budget(a,b));self.assertFalse(budget(a,b,True))
        b['positive_j']+=.1;self.assertTrue(budget(a,b,True))

    def test_forward_rest_gravity_energy_and_actual_velocity(self):
        dt=.00025;n=41;q0=self.m.qpos0.copy();q0[2]=1
        loads=np.zeros((n,self.m.nv));v0=np.zeros(self.m.nv)
        q,v,info=integrate_inputs(self.m,q0,v0,loads,dt,policy='foh')
        metrics=mechanics(self.m,q,v,loads,dt)
        self.assertAlmostEqual(metrics['positive_j'],0)
        self.assertLess(abs(metrics['energy_residual_j']),1e-10)
        self.assertAlmostEqual(v[-1,2],-9.81*(n-1)*dt,places=10)
        self.assertEqual(info['state_resets'],0)


class TestSavedExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=OUT/'summary.json'
        if not path.exists():raise unittest.SkipTest('Run control_retiming.py for full integration checks')
        cls.report=json.loads(path.read_text(encoding='utf-8'))

    def test_complete_condition_grid_and_no_state_resets(self):
        for resolution in self.report['resolutions'].values():
            runs=resolution['runs'];self.assertEqual(len(runs),27)
            self.assertEqual(len({r['id'] for r in runs}),27)
            self.assertFalse(any('failure' in r for r in runs))
            self.assertTrue(all(r['state_resets']==0 for r in runs))

    def test_zero_condition_is_exact_baseline(self):
        for resolution in self.report['resolutions'].values():
            zero=next(r for r in resolution['runs'] if r['id']=='0,0,0')
            self.assertEqual(zero['gain_kmh'],0.)
            self.assertFalse(zero['strict_violations'])
            self.assertEqual(zero['max_foot_change_mm'],0.)

    def test_selected_failure_is_not_reported_as_improvement(self):
        r=self.report
        self.assertFalse(r['actual_athlete_improvement_demonstrated'])
        self.assertIsNone(r['spin_or_command_improvement'])
        if r['selected']['redistribution_violations']:
            self.assertFalse(r['selected']['redistribution_resolved_candidate'])

    def test_saved_states_have_identical_initial_conditions_and_arm_inputs(self):
        z=np.load(OUT/'selected.npz')
        for k in ('q','v'):
            np.testing.assert_array_equal(z['baseline_'+k][0],z['candidate_'+k][0])
        m=mujoco.MjModel.from_xml_path(str(OUT/'source/model.xml'))
        for name in ('external_force','external_torque','pitch_shoulder','elbow','marker_roll','wrist_marker_rotation'):
            ix=groups(m)[name]
            np.testing.assert_array_equal(z['baseline_loads'][:,ix],z['candidate_loads'][:,ix])

    def test_all_energy_balances_small(self):
        for resolution in self.report['resolutions'].values():
            self.assertLess(abs(resolution['baseline']['energy_residual_j']),.1)
            for r in resolution['runs']:self.assertLess(abs(r['energy_residual_j']),.1)


if __name__=='__main__':unittest.main()
