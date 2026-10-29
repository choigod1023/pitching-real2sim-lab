import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ..stage2.test_uncertainty import synthetic_case, synthetic_data
from ..stage2.uncertainty import prepare_case, speeds, grouped_mc
from . import targets
from .run import run, main, PACKAGE, write_json, tracked_hashes


class TargetTests(unittest.TestCase):
    def test_conditions_complete_and_fixed_nominal(self):
        rows = targets.conditions()
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r['name'] for r in rows}), 24)
        for mask, row in enumerate(rows[:16]):
            self.assertEqual(row['fixed'], tuple(n for i,n in enumerate(targets.FACTORS) if mask & (1<<i)))

    def test_common_random_samples_and_seed(self):
        a = targets.common_samples(100, 3)
        np.testing.assert_array_equal(a, targets.common_samples(100, 3))
        self.assertFalse(np.array_equal(a, targets.common_samples(100, 4)))
        with self.assertRaises(ValueError): targets.common_samples(1, 3)

    def test_original_prior_width_reproduced(self):
        case = synthetic_case()
        u = targets.common_samples(32768, 20261029).reshape(-1,7)
        output = targets.summarize(targets.evaluate(case, u))
        prior = grouped_mc(lambda x:speeds(case,x),32768,20261029)
        self.assertAlmostEqual(output['mean_kmh'],prior['mean_kmh'])
        self.assertAlmostEqual(output['width_kmh'],prior['scenario_quantiles_025_50_975_kmh'][2]-prior['scenario_quantiles_025_50_975_kmh'][0])

    def test_distance_mm_units_and_input_unchanged(self):
        u = np.full((2,7), .5)
        u[:,0] = [0,1]
        original = u.copy()
        narrowed = targets.conditioned_samples(u, (), 1.)
        np.testing.assert_allclose(.1+.1*narrowed[:,0], [.149,.151])
        np.testing.assert_array_equal(u,original)
        case=synthetic_case()
        output=targets.evaluate(case,u,targets.FACTORS[1:],1.)
        np.testing.assert_allclose(output, np.array([.149,.151])*100*3.6)

    def test_each_fixed_factor_and_allfixed_width_zero(self):
        u = targets.common_samples(1024,5).reshape(-1,7)
        for name,index in zip(targets.FACTORS,(0,2,3,1)):
            fixed=targets.conditioned_samples(u,(name,))
            np.testing.assert_array_equal(fixed[:,index], .5)
        result=targets.summarize(targets.evaluate(synthetic_case(),u,targets.FACTORS,0.))
        self.assertEqual(result['width_kmh'],0.)
        self.assertAlmostEqual(result['mean_kmh'],54.)

    def test_zero_offset_columns_cannot_change_result(self):
        u=targets.common_samples(128,4).reshape(-1,7)
        changed=u.copy()
        changed[:,4:] = 1.-changed[:,4:]
        np.testing.assert_array_equal(targets.evaluate(synthetic_case(),u),targets.evaluate(synthetic_case(),changed))

    def test_invalid_inputs(self):
        for fixed,width in [(('unknown',),1.),(('distance','distance'),1.),((),-1.),((),51.)]:
            with self.assertRaises(ValueError): targets.conditioned_samples(np.full((1,7),.5),fixed,width)
        with self.assertRaises(ValueError): targets.conditioned_samples(np.ones((2,4)))
        with self.assertRaises(ValueError): targets.summarize([])
        with self.assertRaises(ValueError): targets.width_shapley({0:1.})

    def test_sensitivity_scalar_and_vector_differ_under_cancellation(self):
        case=synthetic_case()
        result=targets.local_distance_sensitivity(case)
        self.assertAlmostEqual(result['analytic_slope_kmh_per_mm'],.36)
        self.assertLess(result['max_analytic_fd_difference_kmh_per_mm'],1e-12)
        case['wrist_velocity'][:]=[20.,-15.,0.]
        result=targets.local_distance_sensitivity(case)
        self.assertAlmostEqual(result['analytic_slope_kmh_per_mm'],0.)
        self.assertGreater(result['vector_derivative_norm_kmh_per_mm'],.35)

    def test_width_shapley_efficiency_and_interactions(self):
        # Non-additive max-like width reduction; allocations must still sum to full reduction.
        widths={mask: float((16-mask.bit_count()**2)) for mask in range(16)}
        values=targets.width_shapley(widths)
        self.assertAlmostEqual(sum(values.values()),16.)
        for value in values.values(): self.assertAlmostEqual(value,4.)

    def test_new_goal_requires_five_pitches_both_seeds(self):
        rows=[{'pitch':f'3034_{i}','seed':s,'summary':{'width_kmh':2.}} for i in range(1,6) for s in targets.SEEDS]
        self.assertTrue(targets.scenario_goal(rows)['all_five_both_seeds_width_le_2'])
        rows[0]['summary']['width_kmh']=2.001
        self.assertFalse(targets.scenario_goal(rows)['all_five_both_seeds_width_le_2'])
        with self.assertRaises(ValueError): targets.scenario_goal(rows[:5])
        diff=targets.numerical_difference(targets.summarize([1.,2.]),targets.summarize([2.,3.]))
        self.assertAlmostEqual(diff['mean_difference_kmh'],1.)
        self.assertAlmostEqual(diff['width_difference_kmh'],0.)

    def test_no_radar_loader_and_unchanged_original_criteria(self):
        before=tracked_hashes()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            c3d=synthetic_data(path)  # Metadata contains no radar field.
            with patch('coach_hypotheses.stage2.uncertainty.ezc3d.c3d',return_value=c3d):
                case=prepare_case(path,'3034_1')
            self.assertTrue(np.isfinite(targets.evaluate(case,np.full((1,7),.5))).all())
        self.assertEqual(before,tracked_hashes())
        self.assertNotIn('recorded',inspect.getsource(targets.evaluate))

    def test_runner_freezes_before_recorded_comparison_and_boundaries(self):
        def prepared(data,pitch):
            return dict(synthetic_case(),pitch=pitch,sources={})
        def compare(data,rows):
            self.assertTrue((output/'scenario_results.json').exists())
            self.assertTrue((output/'summaries.csv').exists())
            return []
        with tempfile.TemporaryDirectory(dir=PACKAGE) as folder:
            output=Path(folder)
            with patch('coach_hypotheses.measurement_targets.run.N',32768), patch('coach_hypotheses.measurement_targets.run.prepare_case',side_effect=prepared), patch('coach_hypotheses.measurement_targets.run.recorded_comparison',side_effect=compare):
                payload=run('no_radar_metadata',output)
            self.assertEqual(len(payload['summaries']),240)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertTrue(manifest['existing_files_preserved'])
            self.assertEqual(len(manifest['scenario_frozen_before_recorded_read']),2)
            write_json(output/'roundtrip.json',{'a':1.})
            self.assertEqual(json.loads((output/'roundtrip.json').read_text()),{'a':1.})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError): run('.',Path(folder))
        with patch('sys.argv',['targets','--data','.']),patch('coach_hypotheses.measurement_targets.run.run',return_value={'summaries':[],'conditions':[],'local_conservative_distance_mm_for_1kmh_across_five':1.}):
            main()


if __name__ == '__main__': unittest.main()
