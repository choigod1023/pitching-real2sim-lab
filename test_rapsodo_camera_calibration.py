import unittest

import numpy as np

from experiments.rapsodo_camera_calibration import study


class CameraCalibrationTests(unittest.TestCase):
    def test_basis_and_independent_line(self):
        scene = {"plate_and_device_px": [[960, 600], [960, 800]],
                 "wall_primary_px": [[1000, 580], [1100, 680]],
                 "vertical_primary_px": [[1100, 400], [1100, 600]]}
        basis, vp = study.camera_basis(scene, 2000., study.PRINCIPAL)
        np.testing.assert_allclose(vp, study.PRINCIPAL)
        np.testing.assert_allclose(basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(study.direction_residual([[500, 400], [500, 800]], basis[2],
                                                         2000., study.PRINCIPAL), 0.)

    def test_sphere_direction_and_cross_pitch_focal(self):
        focal, fps = 3500., 500
        velocity = np.array([-.8, 36., .35])
        frames = np.arange(30)
        points = np.array([-.15, 4., -.1]) + frames[:, None] * velocity / fps
        x, y, z = points.T
        trace = np.column_stack((frames, 960 + focal*x/y, 540 - focal*z/y, focal*study.RADIUS_M/y))
        np.testing.assert_allclose(study.positions(trace, focal, study.PRINCIPAL), points, atol=1e-12)
        np.testing.assert_allclose(study.initial_velocity(trace, focal, study.PRINCIPAL, fps), velocity, atol=1e-12)
        self.assertAlmostEqual(study.cross_focal(trace, np.linalg.norm(velocity), fps), focal)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            study.initial_velocity(np.delete(trace, 10, axis=0), focal, study.PRINCIPAL, fps)

    def test_spin_fraction_and_transverse_transport(self):
        recorded = {"horizontal_angle_deg": 3., "vertical_angle_deg": -1.}
        optical = study.angular_pitch(recorded, [0., 1., 0.])
        axis = [.8, -.6, 0.]
        active, transported, transverse = study.revised_inputs(recorded, optical, axis)
        self.assertAlmostEqual(active, .8)
        np.testing.assert_allclose(transverse, [1., 0., 0.])
        forward, right, _ = study.launch_basis(recorded)
        np.testing.assert_allclose(transported, right)
        self.assertAlmostEqual(transported @ forward, 0.)

    def test_camera_pose_cancels_for_spin_efficiency(self):
        pitch = {"horizontal_angle_deg": 2., "vertical_angle_deg": 1.}
        optical_forward = study.unit([.03, 1., .06])
        axis = study.unit([.72, .6, -.34])
        radians = .14
        basis = np.array([[1., 0., 0.], [0., np.cos(radians), -np.sin(radians)],
                          [0., np.sin(radians), np.cos(radians)]])
        observed = study.angular_pitch(pitch, basis @ optical_forward)
        active, _, _ = study.revised_inputs(pitch, observed, basis @ axis)
        self.assertAlmostEqual(active, np.linalg.norm(np.cross(axis, optical_forward)))

    def test_project_path_uses_nominal_frames_and_excludes_late_overrun(self):
        path = np.array([[0., 0., 1.], [.1, 1., 1.], [.2, 2., .9], [.21, 2.1, .88]])
        projected = study.project_path(path, [100, 101, 102, 103], 100, [0., 4., 0.],
                                       np.eye(3), 1000., 100., dt=.01)
        np.testing.assert_allclose(projected[:3], [[960., 540.], [980., 540.],
                                                   [960. + 200./6., 540. + 100./6.]])
        self.assertTrue(np.isnan(projected[-1]).all())


if __name__ == "__main__":
    unittest.main()
