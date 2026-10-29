import unittest
import numpy as np
from counterview_consistency import feasible, minimum_padding
from prime17_hand_alignment import project


class RegionFeasibilityTests(unittest.TestCase):
    def cameras(self):
        k = np.array([[500.,0,320.],[0,500.,240.],[0,0,1.]])
        return {0:(k,np.eye(3),np.zeros(3)),1:(k,np.eye(3),np.array([-.5,0,0]))}

    def test_known_point_has_feasible_positive_depth_regions(self):
        cams = self.cameras();point=np.array([.1,.2,4.])
        rows=[]
        for c in cams:
            u=project(point,cams[c])[0];rows.append({'camera':c,'box':[u[0]-1,u[1]-1,u[0]+1,u[1]+1]})
        result=feasible(rows,cams);self.assertTrue(result['feasible'])
        for r in rows:
            u=project(result['conditional_example_point_m'],cams[r['camera']])[0]
            self.assertTrue(np.all(u>=np.array(r['box'][:2])-1e-7))
            self.assertTrue(np.all(u<=np.array(r['box'][2:])+1e-7))

    def test_disjoint_same_view_is_infeasible_and_padding_is_not_truth(self):
        cams=self.cameras();rows=[{'camera':0,'box':[310,230,320,250]},{'camera':0,'box':[324,230,330,250]}]
        self.assertFalse(feasible(rows,cams)['feasible'])
        self.assertAlmostEqual(minimum_padding(rows,cams),2.,delta=.002)

    def test_invalid_box_rejected(self):
        with self.assertRaises(ValueError):
            feasible([{'camera':0,'box':[10,10,0,0]}],self.cameras())


if __name__=='__main__':unittest.main()
