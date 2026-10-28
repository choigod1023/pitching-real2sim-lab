import unittest
import numpy as np
from fixed_alignment_audit import similarity,sample_origins,project

class TestFixedAlignment(unittest.TestCase):
    def test_similarity_units_rotation_translation(self):
        a={'scale':0.001,'rotation':[[0,-1,0],[1,0,0],[0,0,1]],'translation':[1,2,3]}
        np.testing.assert_allclose(similarity([[1000,0,2000]],a),[[1,3,5]])
    def test_offset_is_array_index_not_c3d_label(self):
        origins=np.arange(30).reshape(10,1,3)
        for offset in [0,1,2]:
            np.testing.assert_array_equal(sample_origins(origins,[2,4],offset),origins[[2+offset,4+offset]])
        with self.assertRaises(IndexError):sample_origins(origins,[9],1)
        with self.assertRaises(IndexError):sample_origins(origins,[0],-1)
    def test_projection_translation_mm_and_depth(self):
        p={'reference_to_camera_rotation_matrix':np.eye(3).tolist(),'reference_to_camera_translation_mm':[1000,0,0]}
        k={'fx_px':100,'fy_px':200,'cx_px':10,'cy_px':20,'distortion':[0]*5}
        np.testing.assert_allclose(project([[0,1,2]],p,k),[[60,120]])
        with self.assertRaises(ValueError):project([[0,1,-2]],p,k)
    def test_noninteger_offset_rejected(self):
        with self.assertRaises(ValueError):sample_origins(np.zeros((3,1,3)),[0],0.5)
if __name__=='__main__':unittest.main()
