import unittest,json
import numpy as np
from wrist_residual_audit import project,OUT,INPUT

class ResidualAuditTests(unittest.TestCase):
    def test_projection_velocity_acceleration_and_idempotence(self):
        g=np.array([[1.,2.,0.],[2.,4.,0.]])
        v=np.array([5.,3.,2.]);rhs=np.array([3.,6.]);p=project(v,g,rhs)
        np.testing.assert_allclose(g@p,rhs,atol=1e-12)
        np.testing.assert_allclose(project(p,g,rhs),p,atol=1e-12)
        self.assertEqual(p[2],v[2])

    def test_original_reproduces_previous_run_and_decomposition(self):
        rows=json.loads((OUT/'results.json').read_text())['rows']
        original=json.loads((INPUT/'muscle_screen.json').read_text())['reports']
        for x in rows:
            if x['model']=='scaled_source' and x['mode']=='original':
                prior=next(y for y in original if y['pitch']==x['pitch'])
                self.assertAlmostEqual(x['rms']['residual'],prior['hill_residual_rms_nm'],places=7)
            for r in x['rows']:
                self.assertAlmostEqual(r['residual']**2,r['common_component']**2+r['wrist_complement']**2,places=6)
            if x['mode']!='original':
                self.assertLess(x['rms']['u_constraint'],1e-10)
                self.assertLess(x['rms']['a_constraint'],1e-9)

if __name__=='__main__':unittest.main()
