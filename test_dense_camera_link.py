import json,unittest
from pathlib import Path
import cv2,numpy as np
from dense_camera_link import map_cube_name,pair_cameras
from background_orientation import rotation_difference,ratio_matches
from recover_board_order import association
from dynamic_camera_recovery import grid_variants,OBJ
from edge_calibration import project

class DenseLinkTests(unittest.TestCase):
    def test_cube_symmetry_preserves_height_and_is_invertible(self):
        for label in ['A0','B0','C0','D0','A1','B1','C1','D1']:
            self.assertEqual(map_cube_name(label,4),label);self.assertEqual(map_cube_name(map_cube_name(label,1),3),label);self.assertEqual(map_cube_name(label,1)[1],label[1])
        self.assertEqual(map_cube_name('B1',1),'A1');self.assertEqual(map_cube_name('B1',3),'C1')
    def test_pair_camera_uses_camera3_intrinsics(self):
        source=json.loads(Path('edge_intrinsics_frozen.json').read_text());cc=pair_cameras(np.eye(3),np.array([-500.,0,0]),source);self.assertEqual(cc[1]['K'][0,0],source['calibrations']['edgertronic_3']['fx_px']);self.assertNotEqual(cc[1]['K'][0,0],source['calibrations']['edgertronic_4']['fx_px'])
    def test_frame_order_association_recovers_known_synthetic_correspondence(self):
        k=np.array([[1200.,0,640],[0,1200,512],[0,0,1]]);d=np.zeros(5);r=cv2.Rodrigues(np.array([.01,.04,.02]))[0];t=np.array([-500.,20,0]);camera={'R':r,'t':t,'K':k,'dist':d};world={f:OBJ+np.array([f*8.,f*3.,3000.]) for f in range(4)};truth={f:project(q,camera) for f,q in world.items()};db={f:list(grid_variants(truth[f]).values())[f] for f in world};mapped,costs=association(r,t,list(world),world,db,k,d)
        for f in world:np.testing.assert_allclose(mapped[f],truth[f]);self.assertAlmostEqual(costs[f]['prediction_rms_px'],0.)
    def test_rotation_comparison_distinguishes_half_turn(self):
        self.assertAlmostEqual(rotation_difference(np.eye(3),np.eye(3)),0.);self.assertAlmostEqual(rotation_difference(cv2.Rodrigues(np.array([0.,0,np.pi]))[0],np.eye(3)),180.,places=5)
    def test_no_texture_is_not_a_valid_feature_match(self):
        self.assertEqual(ratio_matches(None,None),{});self.assertEqual(ratio_matches(np.zeros((2,128),np.float32),np.zeros((1,128),np.float32)),{})
if __name__=='__main__':unittest.main()
