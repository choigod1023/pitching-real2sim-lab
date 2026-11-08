import unittest
import numpy as np
from marker_definition_check import measured_markers


def fixture(unit='m',factor=1.):
    points=np.ones((4,2,3));points[:3]=np.arange(18.).reshape(3,2,3)*factor
    return {'parameters':{'POINT':{'LABELS':{'value':['RFIN','RWRA']},'UNITS':{'value':[unit]}}},
            'data':{'points':points,'meta_points':{'residuals':np.zeros((1,2,3))}}}


class MarkerDefinitionsTests(unittest.TestCase):
    def test_native_units_preserve_same_physical_positions(self):
        m,valid=measured_markers(fixture(),['RFIN','RWRA'])
        for unit,factor in [('mm',1000.),('cm',100.)]:
            other,v=measured_markers(fixture(unit,factor),['RFIN','RWRA'])
            np.testing.assert_allclose(m,other);self.assertTrue(v.all())
        self.assertTrue(valid.all())

    def test_residual_gap_is_not_overridden_by_homogeneous_row(self):
        c=fixture();c['data']['meta_points']['residuals'][0,0,1]=-1
        _,v=measured_markers(c,['RFIN','RWRA'])
        self.assertFalse(v[1,0]);self.assertTrue(v[1,1])

    def test_ambiguous_label_and_unknown_units_rejected(self):
        c=fixture();c['parameters']['POINT']['LABELS']['value']=['RFIN','RFIN']
        with self.assertRaises(ValueError):measured_markers(c,['RFIN'])
        with self.assertRaises(ValueError):measured_markers(fixture('inch'),['RFIN'])


if __name__=='__main__':unittest.main()
