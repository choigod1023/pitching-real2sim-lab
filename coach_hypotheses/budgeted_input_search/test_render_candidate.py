import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from .render_candidate import require_candidate,project,export,fit_view


class CandidateRenderTests(unittest.TestCase):
    def test_guard_requires_actual_strict_pair_and_resolved_gain(self):
        row={'budget_gate':{'strict_model_budget_pass':True},'pose_screen_pass':True,
             'initial_state_equal':True,'paired_gain_kmh':.6}
        report={'baseline_fallback':False,'eligible_count':1,
                'selected':{'coarse':copy.deepcopy(row),'fine':copy.deepcopy(row)}}
        self.assertIs(require_candidate(report),report['selected']['fine'])
        report['selected']['fine']['budget_gate']['strict_model_budget_pass']=False
        with self.assertRaises(ValueError):require_candidate(report)
        report['selected']['fine']['budget_gate']['strict_model_budget_pass']=True
        report['selected']['fine']['paired_gain_kmh']=.61
        with self.assertRaises(ValueError):require_candidate(report)

    def test_failed_report_cannot_export_or_create_media(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);report=p/'report.json';report.write_text(json.dumps({'baseline_fallback':True}))
            with self.assertRaises(ValueError):export(report,p/'missing',p/'data',p/'media')
            self.assertFalse((p/'media').exists())

    def test_common_projection_preserves_alignment_and_scaling(self):
        center=np.array([1.,2.,3.]);origin=np.array([100.,100.]);points=np.array([[1.,2.,3.],[2.,2.,3.]])
        a=project(points,center,100.,origin);b=project(points,center,200.,origin)
        np.testing.assert_array_equal(a[0],origin)
        np.testing.assert_allclose(b-origin,2*(a-origin),atol=1e-12)

    def test_all_frames_fit_the_same_view(self):
        rng=np.random.default_rng(1);points=rng.normal(size=(100,3))*[2,1,.5]
        center,scale,origin=fit_view(points,744,145,672,550)
        projected=project(points,center,scale,origin)
        self.assertTrue(np.all(projected[:,0]>744+20))
        self.assertTrue(np.all(projected[:,0]<744+672-20))
        self.assertTrue(np.all(projected[:,1]>145+50))
        self.assertTrue(np.all(projected[:,1]<145+550-10))


if __name__=='__main__':unittest.main()
