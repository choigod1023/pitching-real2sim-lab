import unittest,json,xml.etree.ElementTree as E
import numpy as np
import opensim as o
from shoulder_load_audit import ROOT,MODEL,OUT

class LoadAuditTests(unittest.TestCase):
    def test_runtime_scaling_preserves_mass_and_scales_inertia(self):
        o.Logger.setLevelString('error');m=o.Model(str(ROOT/'artifacts/anatomical-shoulder/TSM_subject_noWeight.osim'));s=m.initSystem();b=m.getBodySet().get('humerus');mass=b.getMass();inertia=np.array([b.getInertia().getMoments().get(i) for i in range(3)])
        ss=o.ScaleSet();sc=o.Scale();sc.setApply(True);sc.setSegmentName('humerus');sc.setScaleFactors(o.Vec3(2.));ss.cloneAndAppend(sc);m.scale(s,ss,True)
        self.assertAlmostEqual(b.getMass(),mass,places=12)
        np.testing.assert_allclose([b.getInertia().getMoments().get(i) for i in range(3)],inertia*4.,rtol=1e-12)

    def test_only_mass_and_inertia_changed(self):
        before=E.parse(MODEL);after=E.parse(OUT/'mass_corrected.osim');source=E.parse(ROOT/'artifacts/anatomical-shoulder/TSM_subject_noWeight.osim')
        for b in source.findall('.//BodySet/objects/*'):
            name=b.get('name');new=after.find(f".//BodySet/objects/*[@name='{name}']")
            self.assertAlmostEqual(float(b.findtext('mass')),float(new.findtext('mass')),places=12)
        # Strip intended inertial edits then compare all remaining XML nodes.
        for tree in [before,after]:
            for b in tree.findall('.//BodySet/objects/*'):
                for field in ['mass','inertia']:b.find(field).text='INERTIAL_FIELD'
        self.assertEqual(E.tostring(before.getroot()),E.tostring(after.getroot()))

    def test_torque_and_group_decomposition_and_torso_loads(self):
        for filename in ['results.json','corrected_results.json']:
            reports=json.loads((OUT/filename).read_text())['reports'];self.assertEqual(len(reports),5)
            for report in reports:
                for r in report['rows']:
                    self.assertLess(r['torque_reconstruction_error'],1e-9)
                    self.assertLess(r['group_orthogonality'],1e-10)
                    self.assertEqual(r['group_dofs'],r['internal_dofs'])
                    self.assertLess(r['torso_wrench_internal_leak'],1e-10)
                    self.assertAlmostEqual(sum(g['residual']**2 for g in r['groups'].values()),r['residual']**2,delta=1e-6)
                    self.assertLess(r['signed_unbounded'],1e-8)
                    self.assertLess(r['positive_unbounded'],1e-8)

if __name__=='__main__':unittest.main()
