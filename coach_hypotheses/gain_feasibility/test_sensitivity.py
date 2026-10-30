"""Synthetic kinematic checks, independent of original experiment outputs."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from .sensitivity import interpolate_state, paired_sensitivity, paired_summary, run, site_velocity


def rigid_model():
    # The diagnostic site lives in a translated/rotated fixed child, so these
    # checks also detect confusing child-local and hand-local grip coordinates.
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="hand"><freejoint/><geom type="sphere" size=".03" mass="1"/>
        <body name="payload" pos=".02 .01 0" quat=".7071067811865476 0 0 .7071067811865475">
          <site name="ball_center" pos=".01 -.13 0"/>
        </body>
      </body></worldbody></mujoco>''')


def rigid_states(model, n=3, dt=.01, scale=1.):
    q = np.repeat(model.qpos0[None], n, axis=0)
    v = np.zeros((n, model.nv))
    for i, t in enumerate(np.arange(n) * dt):
        angle = scale * (t + t * t)
        mujoco.mj_integratePos(model, q[i], np.array([0., 0., 0., 0., 0., angle]), 1.)
        v[i, 5] = scale * (1. + 2. * t)
    return q, v


class InterpolationTests(unittest.TestCase):
    def test_quaternion_midpoint_and_linear_velocity_without_rounding(self):
        model = rigid_model()
        q = np.repeat(model.qpos0[None], 2, axis=0)
        mujoco.mj_integratePos(model, q[1], np.array([1., 0., 0., 0., 0., 2. * np.pi / 3.]), 1.)
        # Negating a unit quaternion must not change its physical interpolation.
        q[1, 3:] *= -1.
        v = np.array([[0., 0., 0., 0., 0., 1.], [2., 0., 0., 0., 0., 3.]])
        midpoint, velocity = interpolate_state(model, q, v, .01, .0025)
        difference = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, difference, 1., q[0], midpoint)
        np.testing.assert_allclose(difference, [0.25, 0., 0., 0., 0., np.pi / 6.], atol=1e-12)
        np.testing.assert_allclose(velocity, [.5, 0., 0., 0., 0., 1.5], atol=1e-12)

    def test_closed_boundaries_no_extrapolation_and_no_source_mutation(self):
        model = rigid_model()
        q, v = rigid_states(model)
        for t, index in [(0., 0), (.02, 2)]:
            qi, vi = interpolate_state(model, q, v, .01, t)
            np.testing.assert_array_equal(qi, q[index])
            np.testing.assert_array_equal(vi, v[index])
            qi[:] = 99
            self.assertFalse(np.any(q[index] == 99))
        for t in [-1e-12, np.nextafter(.02, np.inf)]:
            with self.assertRaisesRegex(ValueError, 'outside'):
                interpolate_state(model, q, v, .01, t)
        for bad_dt in [0., np.nan]:
            with self.assertRaises(ValueError):
                interpolate_state(model, q, v, bad_dt, 0.)
        invalid_q = q.copy()
        invalid_q[0, 3:] = 0.
        with self.assertRaisesRegex(ValueError, 'unit norm'):
            interpolate_state(model, invalid_q, v, .01, 0.)


class JacobianTests(unittest.TestCase):
    def test_known_rigid_rotation_and_input_model_unchanged(self):
        model = rigid_model()
        original_site = model.site_pos.copy()
        q = model.qpos0.copy()
        v = np.array([1., 0., 0., 0., 0., 2.])
        for grip in [.14, .15, .16]:
            np.testing.assert_allclose(site_velocity(model, q, v, grip), [1., 2. * grip, 0.], atol=1e-12)
        np.testing.assert_array_equal(model.site_pos, original_site)

    def test_jacobian_matches_independent_central_position_difference(self):
        model = rigid_model()
        # Model's original fixed-child site corresponds to hand-local (.15,.02,0),
        # so set its child-local position to the desired hand-local (.15,0,0).
        model.site_pos[0] = [-.01, -.13, 0.]
        q = model.qpos0.copy()
        mujoco.mj_integratePos(model, q, np.array([.1, -.2, .3, .2, -.3, .6]), 1.)
        v = np.array([.2, -.1, .5, -.7, .4, 1.3])
        positions = []
        epsilon = 1e-6
        for sign in [-1., 1.]:
            qi = q.copy()
            mujoco.mj_integratePos(model, qi, v, sign * epsilon)
            data = mujoco.MjData(model)
            data.qpos[:] = qi
            mujoco.mj_forward(model, data)
            positions.append(data.site_xpos[0].copy())
        finite_difference = (positions[1] - positions[0]) / (2. * epsilon)
        np.testing.assert_allclose(site_velocity(model, q, v, .15), finite_difference, atol=1e-9)

    def test_rejects_site_on_moving_child_and_bad_state(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="hand">
          <freejoint/><geom type="sphere" size=".03" mass="1"/>
          <body><joint/><geom type="sphere" size=".02" mass="1"/><site name="ball_center"/></body>
          </body></worldbody></mujoco>''')
        with self.assertRaisesRegex(ValueError, 'fixed'):
            site_velocity(model, model.qpos0, np.zeros(model.nv), .15)
        with self.assertRaises(ValueError):
            site_velocity(model, model.qpos0, np.zeros(1), .15)


class PairedTests(unittest.TestCase):
    def test_shared_scenario_zero_difference_and_positive_offsets_excluded(self):
        model = rigid_model()
        q, v = rigid_states(model)
        report = paired_sensitivity(model, q, v, q.copy(), v.copy(), .01)
        self.assertEqual(len(report['scenarios']), 9)
        self.assertEqual([x['frame_offset'] for x in report['excluded_times']], [1, 2])
        self.assertEqual(report['summary']['paired_delta_range_kmh'], [0., 0.])
        self.assertEqual(report['summary']['sign_stability'], 'zero')
        self.assertIn('not a measured ball-release', report['time_scope'])

    def test_fractional_source_frame_time_and_known_paired_gain(self):
        model = rigid_model()
        bq, bv = rigid_states(model)
        cq, cv = rigid_states(model, scale=2.)
        report = paired_sensitivity(model, bq, bv, cq, cv, .01)
        for row in report['scenarios']:
            expected_time = .02 + row['frame_offset'] / 360.
            self.assertEqual(row['time_s'], expected_time)
            self.assertAlmostEqual(row['paired_delta_kmh'], 3.6 * row['grip_length_m'] * (1. + 2. * expected_time), places=12)
        self.assertEqual(report['summary']['sign_stability'], 'positive')
        # A saved endpoint inside the trajectory allows both positive offsets.
        extended = paired_sensitivity(model, bq, bv, cq, cv, .01, endpoint_time=.01)
        self.assertEqual(len(extended['scenarios']), 15)
        self.assertEqual(extended['excluded_times'], [])

    def test_pair_covariance_prevents_independent_variance_inflation(self):
        baseline = np.array([30., 40., 60.])
        summary = paired_summary(baseline, baseline + 2.)
        self.assertEqual(summary['descriptive_paired_delta_variance_kmh2'], 0.)
        self.assertGreater(summary['descriptive_baseline_variance_kmh2'], 0.)
        self.assertAlmostEqual(summary['variance_identity_rhs_kmh2'], 0.)
        self.assertEqual(paired_summary([1., 2., 3.], [2., 2., 2.])['sign_stability'], 'mixed_or_zero')
        self.assertEqual(paired_summary([2., 3.], [1., 2.])['sign_stability'], 'negative')
        with self.assertRaises(ValueError):
            paired_summary([1., 2.], [1.])

    def test_candidate_saved_interval_is_checked_independently(self):
        model = rigid_model()
        q, v = rigid_states(model)
        report = paired_sensitivity(model, q, v, q[:2], v[:2], .01, frame_offsets=(0,))
        self.assertIsNone(report['summary'])
        self.assertEqual(report['scenarios'], [])
        self.assertEqual(report['excluded_times'][0]['reasons'], ['outside_candidate_saved_interval'])

    def test_invalid_trajectories_and_duplicate_scenarios_rejected(self):
        model = rigid_model()
        q, v = rigid_states(model)
        for kwargs in [{'frame_offsets': [0, .5]}, {'grip_lengths': [.15, .15]}, {'source_frame_dt': 0.}]:
            with self.assertRaises(ValueError):
                paired_sensitivity(model, q, v, q, v, .01, **kwargs)
        broken = v.copy()
        broken[0, 0] = np.nan
        with self.assertRaises(ValueError):
            paired_sensitivity(model, q, broken, q, v, .01)

    def test_npz_report_is_separate_and_provenance_tagged(self):
        from coupled_payload import model_for
        model = model_for([.3, .25], .01)
        q = np.repeat(model.qpos0[None], 3, axis=0)
        v = np.zeros((3, model.nv))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'states.npz'
            out = Path(directory) / 'report.json'
            np.savez(source, qpos=q, qvel=v, dt=.01, lengths=[.3, .25])
            original = source.read_bytes()
            run(source, source, out)
            saved = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(saved['summary']['paired_delta_range_kmh'], [0., 0.])
            self.assertEqual(len(saved['sources']['baseline']['sha256']), 64)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                run(source, source, source)
            self.assertEqual(source.read_bytes(), original)
            candidate = Path(directory) / 'candidate.npz'
            np.savez(candidate, qpos=q, qvel=v, dt=.02, lengths=[.3, .25])
            with self.assertRaisesRegex(ValueError, 'identical dt'):
                run(source, candidate, out)


if __name__ == '__main__':
    unittest.main()
