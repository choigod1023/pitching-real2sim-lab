import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from .restore import choose_start,restoration_margins,run_restore
from . import test_search as fixtures
from .search import evaluate,POLICIES


class RestoreTests(unittest.TestCase):
    def test_start_choice_has_fixed_gain_and_pose_requirements(self):
        rows=[]
        for i,(gain,excess,pose) in enumerate([(.6,.2,True),(.4,.001,True),(.7,.1,True),(.8,.01,False)]):
            rows.append({'evaluation_id':i,'paired_gain_kmh':gain,'pose_screen_pass':pose,
                         'budget_gate':{'violations':[{'metric':'positive_work_per_dof_j','excess':excess}]}})
        self.assertEqual(choose_start(rows)['evaluation_id'],2)
        with self.assertRaises(ValueError):choose_start(rows[1:2])

    def test_both_resolutions_and_floor_are_actual_constraints(self):
        fixture=fixtures.BudgetInputTests();fixture.setUp()
        b=fixture.base
        result,_=evaluate(fixture.lengths,fixture.q,fixture.controls,fixture.dt,1,b,POLICIES[0],np.zeros(33))
        pair={'coarse':copy.deepcopy(result),'fine':copy.deepcopy(result)}
        baselines={1:b,2:b}
        margins=restoration_margins(baselines,pair)
        self.assertEqual(margins[-1],-.5)
        self.assertEqual(margins[-2],-.5)
        pair['coarse']['paired_gain_kmh']=.6;pair['fine']['paired_gain_kmh']=.6
        self.assertGreaterEqual(restoration_margins(baselines,pair).min(),-1e-12)
        pair['fine']['positive_work_per_dof_j'][0]+=1.
        self.assertLess(restoration_margins(baselines,pair).min(),0)

    def test_bad_input_hash_prevents_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'source').write_bytes(b'changed')
            (p/'prior').write_text(json.dumps({'source_sha256':'bad'}))
            with patch('coach_hypotheses.budgeted_input_search.restore.minimize') as opt:
                with self.assertRaises(ValueError):run_restore(p/'source',p/'prior',p/'out')
                opt.assert_not_called()

    def test_restoration_targets_raw_cap_not_acceptance_epsilon(self):
        fixture=fixtures.BudgetInputTests();fixture.setUp()
        base=fixture.base
        r,_=evaluate(fixture.lengths,fixture.q,fixture.controls,fixture.dt,1,base,POLICIES[0],np.zeros(33))
        r['paired_gain_kmh']=.6
        r['positive_work_per_dof_j'][0]+=5e-10
        from capacity_budget_gate import gate
        self.assertTrue(gate(base[0],r)['strict_model_budget_pass'])
        pair={'coarse':r,'fine':copy.deepcopy(r)}
        self.assertLess(restoration_margins({1:base,2:base},pair).min(),0)

    def test_redundant_peak_roundoff_does_not_relax_final_peak_gate(self):
        fixture=fixtures.BudgetInputTests();fixture.setUp();base=fixture.base
        r,_=evaluate(fixture.lengths,fixture.q,fixture.controls,fixture.dt,1,base,POLICIES[0],np.zeros(33))
        r['paired_gain_kmh']=.6
        r['peak_abs_generalized_force_per_dof'][0]+=7e-15
        pair={'coarse':r,'fine':copy.deepcopy(r)}
        self.assertGreaterEqual(restoration_margins({1:base,2:base},pair).min(),-1e-12)
        from capacity_budget_gate import gate
        r['peak_abs_generalized_force_per_dof'][0]+=.001
        self.assertFalse(gate(base[0],r)['strict_model_budget_pass'])


if __name__=='__main__':unittest.main()
