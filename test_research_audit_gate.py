import json,unittest
from summarize_research_audit import OUT,path_consistency_gate

class ResearchAuditGateTests(unittest.TestCase):
    def test_missing_and_nonfinite_evidence_cannot_pass(self):
        self.assertFalse(path_consistency_gate([])['passed'])
        for values in [[],[{'max_error_m_s':float('nan')}]]:
            self.assertFalse(path_consistency_gate([{'pitch':'invalid','sample':0,'epsilon_sweep':values}])['passed'])
    def test_actual_wrapped_path_fails_consistency_gate(self):
        reports=json.loads((OUT/'formula_checks.json').read_text())['reports']
        self.assertFalse(path_consistency_gate(reports)['passed'])
        control=json.loads((OUT/'wrapping_control.json').read_text())['rows'];wrapped,straight=control
        self.assertGreater(abs(wrapped['path_speed_m_s']-wrapped['finite_difference_speed_m_s']),.009)
        self.assertLess(abs(straight['path_speed_m_s']-straight['finite_difference_speed_m_s']),1e-8)
        self.assertLess(wrapped['body_station_velocity_error_m_s'],1e-9)

    def test_all_steps_must_pass_and_boundary_is_explicit(self):
        sample={'pitch':'synthetic','sample':0,'epsilon_sweep':[{'max_error_m_s':1e-8},{'max_error_m_s':.01}]}
        self.assertFalse(path_consistency_gate([sample])['passed'])
        sample['epsilon_sweep']=[{'max_error_m_s':1e-8},{'max_error_m_s':1e-4}]
        self.assertTrue(path_consistency_gate([sample])['passed'])

if __name__=='__main__':unittest.main()
