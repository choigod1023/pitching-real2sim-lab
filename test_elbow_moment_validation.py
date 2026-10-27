import json,unittest,xml.etree.ElementTree as E
import numpy as np
from elbow_moment_validation import OUT,BASE,PREVIOUS

class MomentValidationTests(unittest.TestCase):
    def test_template_length_match(self):
        info=json.loads((OUT/'scale_manifest.json').read_text())
        self.assertAlmostEqual(info['source_template_distance_mm']*info['new_humerus_scale'],info['subject_joint_center_distance_mm'],places=9)
        self.assertNotAlmostEqual(info['new_humerus_scale'],info['previous_humerus_scale'],places=3)

    def test_body_inertia_and_maximum_force_unchanged(self):
        for name in ['registered_wrist','registered_plus_elbow']:
            old=E.parse(PREVIOUS/f'{name}.osim');new=E.parse(OUT/f'matched/{name}.osim')
            for b in old.findall('.//BodySet/objects/*'):
                c=new.find(f".//BodySet/objects/*[@name='{b.get('name')}']")
                for field in ['mass','mass_center','inertia']:self.assertEqual(b.findtext(field),c.findtext(field))
            for m in old.findall('.//ForceSet/objects/*'):
                if m.find('max_isometric_force') is None:continue
                n=new.find(f".//ForceSet/objects/*[@name='{m.get('name')}']")
                self.assertEqual(float(m.findtext('max_isometric_force')),float(n.findtext('max_isometric_force')))

    def test_transplanted_moment_curves_match_own_source(self):
        c=json.loads((OUT/'moment_curves.json').read_text())
        for a,b in [('source_prior_scaled','transplanted_prior'),('subject_matched','transplanted_corrected')]:
            for name in ['BRD','BRA','ECRL','PT','TRIlat']:
                np.testing.assert_allclose(c[a][name]['model_values_mm'],c[b][name]['model_values_mm'],atol=1e-4)

    def test_results_same_motion_and_finite(self):
        rows=json.loads((OUT/'results.json').read_text())['rows'];self.assertEqual(len(rows),20)
        for p in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
            r=[x for x in rows if x['pitch']==p];self.assertEqual(len(r),4)
            self.assertLess(np.ptp([x['arm_rmse_mm'] for x in r]),1e-10)
            self.assertTrue(np.isfinite([x['rms']['residual'] for x in r]).all())

if __name__=='__main__':unittest.main()
