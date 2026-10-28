import json,unittest,xml.etree.ElementTree as E
import numpy as np
import opensim as o
from anatomical_shoulder_fit import arr
from anatomical_muscle_screen import ov
from applied_path_forces import unit_path_force
from local_force_comparison import require_domain
from scholz_wrapping_trial import ROOT,OUT,SOURCE

class AppliedPathTests(unittest.TestCase):
    def test_force_matrix_matches_actual_inverse_dynamics_load_change(self):
        o.Logger.setLevelString('error')
        for file,names in [(SOURCE,None),(OUT/'local_0p002_length_based_trial.osim',['Subscapularis_I'])]:
            m=o.Model(str(file));s=m.initSystem();z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz')
            if names is None:names=[m.getMuscles().get(i).getName() for i in range(m.getMuscles().getSize())]
            for j,v in enumerate(z['q'][28]):m.getCoordinateSet().get(j).setValue(s,float(v),False)
            for i in range(m.getForceSet().getSize()):m.getForceSet().get(i).setAppliesForce(s,False)
            s.setU(ov(np.linspace(-.1,.1,s.getNU())));m.realizeVelocity(s);solver=o.InverseDynamicsSolver(m);acc=ov(np.zeros(s.getNU()))
            for name in names:
                m.realizeVelocity(s);mus=m.getMuscles().get(name);column=unit_path_force(m,s,mus.getPath());baseline=arr(solver.solve(s,acc));mus.setAppliesForce(s,True);mus.overrideActuation(s,True);mus.setOverrideActuation(s,1.);m.realizeVelocity(s);loaded=arr(solver.solve(s,acc))
                np.testing.assert_allclose(baseline-loaded,column,atol=1e-9);mus.setAppliesForce(s,False)

    def test_local_geometry_and_derivative_pass_but_global_geometry_fails(self):
        local=json.loads((OUT/'local_0p002_length_based_checks.json').read_text());global_fit=json.loads((OUT/'length_based_checks.json').read_text())
        self.assertTrue(local['geometry_pass']);self.assertFalse(global_fit['geometry_pass'])
        self.assertLess(max(x for r in local['after'] for s in r['sweeps'] for x in s['error_m_s']),1e-4)
        before=E.parse(SOURCE);after=E.parse(OUT/'local_0p002_length_based_trial.osim')
        for node in before.findall('.//ForceSet/objects/*'):
            other=after.find(f".//ForceSet/objects/*[@name='{node.get('name')}']")
            self.assertEqual(node.findtext('max_isometric_force'),other.findtext('max_isometric_force'))
        for body in before.findall('.//BodySet/objects/*'):
            other=after.find(f".//BodySet/objects/*[@name='{body.get('name')}']")
            for field in ['mass','inertia']:self.assertEqual(body.findtext(field),other.findtext(field))

    def test_local_model_cannot_be_applied_to_whole_pitch(self):
        r=json.loads((OUT/'local_0p002_length_based_checks.json').read_text());z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz');n=list(z['coordinate_names']);indices=[n.index(k) for k in ['plane_elv','shoulder_elv','axial_rot']]
        require_domain(z['q'][28,indices],r['domain_low_rad'],r['domain_high_rad'])
        with self.assertRaises(ValueError):require_domain(z['q'][2,indices],r['domain_low_rad'],r['domain_high_rad'])
        with self.assertRaises(ValueError):require_domain([float('nan')]*3,r['domain_low_rad'],r['domain_high_rad'])

if __name__=='__main__':unittest.main()
