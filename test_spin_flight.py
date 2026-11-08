import unittest
import numpy as np
from scipy.integrate import solve_ivp

from spin_flight import (MASS,INERTIA,G,release_energy,energy_speed,velocities,spin_vectors,
                         lift_coefficient,accelerations,flight_batch,aim)
from pitch_performance_frontier import states,E0,improvement,pareto_mask,choose_speed_then_movement
import pandas as pd


class SpinFlightTests(unittest.TestCase):
    def test_energy_reallocation_preserves_budget(self):
        speed=energy_speed(release_energy(130,2400,1.8),2000,1.8)
        self.assertGreater(speed,130)
        self.assertAlmostEqual(release_energy(speed,2000,1.8),release_energy(130,2400,1.8),places=11)
        with self.assertRaises(ValueError):energy_speed(1,20000,1.8)

    def test_spin_efficiency_and_rpm_are_not_double_counted(self):
        v=np.array([[0.,40.,0.]])
        for eff in (0,.6,.75,1):
            w=spin_vectors(v,2400,eff,0)
            self.assertAlmostEqual(np.linalg.norm(w),2400*2*np.pi/60,places=10)
            fraction=np.linalg.norm(np.cross(w[0],v[0]))/(np.linalg.norm(w)*40)
            self.assertAlmostEqual(fraction,eff,places=10)

    def test_lift_is_orthogonal_and_vanishes_for_gyro_without_gravity(self):
        v=np.array([[3.,40.,-2.]])
        w=np.array([[220.,10.,20.]])
        a=accelerations(v,w,cd=0,gravity=0)
        self.assertAlmostEqual(float(np.sum(a*v)),0,places=10)
        np.testing.assert_allclose(accelerations(v,v*3,cd=0,gravity=0),0,atol=1e-12)
        self.assertAlmostEqual(float(lift_coefficient(.1)),.15)

    def test_vacuum_and_independent_ode_solver(self):
        p=np.array([[0.,0.,1.8]]);v=np.array([[.2,40.,0.]]);w=np.zeros((1,3))
        r=flight_batch(p,v,w,cd=0,lift_scale=0,dt=.001)
        self.assertTrue(r['valid'][0]);t=16.8/40
        np.testing.assert_allclose(r['state'][0,:3],[.2*t,16.8,1.8-G*t*t/2],atol=2e-6)
        v=np.array([[0.,130/3.6,-.5]]);w=spin_vectors(v,2400,.75,0)
        def rhs(t,s):return np.r_[s[3:],accelerations(s[None,3:],w)[0]]
        def event(t,s):return s[1]-16.8
        event.terminal=True
        sol=solve_ivp(rhs,[0,1],np.r_[p[0],v[0]],events=event,rtol=1e-10,atol=1e-11)
        r=flight_batch(p,v,w,dt=.001)
        np.testing.assert_allclose(r['state'][0],sol.y[:,-1],atol=1e-5)

    def test_reaim_changes_direction_not_energy_or_command_variance(self):
        p=np.array([[0,0,1.8]]);angles=aim(p,np.array([130.]),[2400],[.75],[0])
        v=velocities([130],angles[:,0],angles[:,1]);w=spin_vectors(v,[2400],[.75],[0])
        r=flight_batch(p,v,w)
        np.testing.assert_allclose(r['state'][0,[0,2]],[0,.9],atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(v)*3.6,130,places=10)

    def test_common_noise_conserves_paired_energy_even_when_rpm_changes(self):
        rows=pd.DataFrame([dict(rpm=2400,efficiency=.75,tilt_deg=0,height_m=1.8,forward_m=0),
                           dict(rpm=2000,efficiency=1.,tilt_deg=0,height_m=1.8,forward_m=0)])
        noise=np.random.default_rng(42).normal(size=(16,10))
        p,v,w,e=states(rows,np.zeros((2,2)),noise)
        actual=(.5*MASS*np.sum(v*v,axis=1)+.5*INERTIA*np.sum(w*w,axis=1)+MASS*G*p[:,2]).reshape(2,16)
        np.testing.assert_allclose(actual,e,atol=1e-12)
        np.testing.assert_allclose(actual[0],actual[1],atol=1e-12)

    def test_joint_gate_rejects_single_metric_wins(self):
        base=dict(speed_kmh=130,ivb_proxy_cm=50,rms_target_cm=7,flight_ms=490)
        bad=dict(speed_kmh=131,ivb_proxy_cm=49,rms_target_cm=6,flight_ms=485)
        self.assertFalse(improvement(bad,base)['all_four_nonworse'])
        self.assertFalse(improvement(base,base)['all_three_strict'])
        np.testing.assert_array_equal(pareto_mask([[1,1],[2,0],[0,0],[1,1]]),[True,True,False,True])
        tied=pd.DataFrame(dict(speed_kmh=[130.+1e-12,130.],ivb_proxy_cm=[50,55],rms_target_cm=[6,6]))
        self.assertEqual(choose_speed_then_movement(tied),1)

    def test_bad_spin_and_nonforward_flight_rejected(self):
        with self.assertRaises(ValueError):spin_vectors([[0,40,0]],2400,1.1,0)
        with self.assertRaises(ValueError):flight_batch([[0,0,1.8]],[[0,-10,0]],[[0,0,0]])


if __name__=='__main__':unittest.main()
