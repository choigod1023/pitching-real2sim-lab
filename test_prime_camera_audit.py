import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from prime_board_audit import triangulate,projection,lengths
from prime_cube_audit import metric,VERTICES,MIDS
from prime_cube_correction import cube_chart,rigid_registration
from prime_cube_markers import red_centres
from fetch_prime_calibration import confirmation


class PrimeCameraTests(unittest.TestCase):
    def setUp(self):
        k=np.array([[1000.,0,640],[0,1000,360],[0,0,1]])
        self.c1=(k,np.eye(3),np.zeros(3));self.c2=(k,np.eye(3),np.array([-1.,0,0]))

    def test_metric_triangulation_in_metres_then_mm(self):
        points=np.array([[0,0,5],[.1,0,5],[0,.1,5],[.1,.1,5.]])
        actual,residual=triangulate(projection(points,self.c1),projection(points,self.c2),self.c1,self.c2)
        np.testing.assert_allclose(actual,points,atol=1e-12)
        self.assertLess(residual.max(),1e-10)
        self.assertAlmostEqual(np.linalg.norm(actual[1]-actual[0])*1000,100.)

    def test_invalid_and_nonfinite_points_rejected(self):
        for points in [[[1,2,3]],[[np.nan,2]]]:
            with self.assertRaises(ValueError):triangulate(points,points,self.c1,self.c2)

    def test_negative_depth_is_not_a_plausible_reconstruction(self):
        with self.assertRaises(ValueError):projection(np.array([[0,0,-1.]]),self.c1)

    def test_board_edges_and_diagonals_are_both_reported(self):
        x,y=np.meshgrid(np.arange(7)*.1,np.arange(4)*.1)
        points=np.stack([x,y,np.ones_like(x)*5.],axis=-1).reshape(-1,3)
        result=lengths(points)
        self.assertEqual(result['count'],45);self.assertEqual(result['diagonal_count'],36)
        self.assertLess(result['mae_from100mm'],1e-10);self.assertLess(result['diagonal_mae_mm'],1e-10)

    def test_rigid_registration_does_not_absorb_scale_error(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]])
        r=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix();t=np.array([1,2,3.])
        fitted,shift=rigid_registration(p,p@r.T+t)
        np.testing.assert_allclose(p@fitted.T+shift,p@r.T+t,atol=1e-12)
        fitted,shift=rigid_registration(p,1.2*p@r.T+t)
        self.assertGreater(np.linalg.norm(p@fitted.T+shift-(1.2*p@r.T+t)),.1)
        self.assertAlmostEqual(np.linalg.det(fitted),1.)

    def test_cube_edges_and_unused_midpoints_have_expected_geometry(self):
        p=cube_chart();m=metric(np.array([p[k] for k in VERTICES+MIDS]),np.zeros(24),VERTICES+MIDS)
        self.assertLess(m['outer_edge_mae_mm'],1e-10)
        self.assertLess(m['half_edge_mae_mm'],1e-10)
        self.assertLess(max(m['vertical_midpoint_displacement_mm']),1e-10)

    def test_red_detection_applies_roi_and_rejects_single_pixel_noise(self):
        im=np.zeros((64,64,3),np.uint8);im[12:16,12:16]=[0,0,255];im[40:45,40:45]=[0,0,255];im[4,4]=[0,0,255]
        found=red_centres(im,[0,0,32,32]);self.assertEqual(len(found),1)
        np.testing.assert_allclose(found[0]['pixel'],[13.5,13.5]);self.assertEqual(found[0]['area_px'],16)

    def test_large_file_confirmation_preserves_file_identity(self):
        page='<title>Virus scan warning</title><form id="download-form" action="https://drive.usercontent.google.com/download"><input type="hidden" name="id" value="public123"><input type="hidden" name="export" value="download"><input type="hidden" name="confirm" value="t"></form>'
        self.assertEqual(confirmation(page,'public123')['id'],'public123')
        with self.assertRaises(ValueError):confirmation(page,'different-file')

    def test_auth_quota_and_foreign_confirmation_are_not_followed(self):
        for page in ['<title>Quota exceeded</title>','<title>Sign in</title>','<title>Virus scan warning</title><form id="download-form" action="https://example.org/download"></form>']:
            with self.assertRaises(ValueError):confirmation(page,'public123')


if __name__=='__main__':unittest.main()
