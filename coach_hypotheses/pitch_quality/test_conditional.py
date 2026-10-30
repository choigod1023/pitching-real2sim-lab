import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.optimize import check_grad
from scipy.special import expit

from . import conditional as c


def row(i=0, pitcher=1, game=101, call='S'):
    return dict(gamePk=game, gameDate='2025-06-03', atBatIndex=i, pitchIndex=0,
                pitcherId=pitcher, batterId=100, pitchType='FF', identity_status='verified_within_play',
                pitchHand='R', batSide='L', startSpeed=90+i*.01, pX=.2, pZ=2.5,
                spinRate=2300+i, breakVerticalInduced=14+i*.01, breakHorizontal=-4,
                pre_count=dict(balls=0, strikes=0), post_count=dict(balls=0, strikes=1), callCode=call, detailsCode=call,
                official_pitch_code=dict(code=call, swingStatus=True, buntAttemptStatus=False,
                                         swingMissStatus=call in ('S', 'W', 'T', 'Q')))


class ConditionalTests(unittest.TestCase):
    def test_target_strict_and_official_keep_foul_tips_and_pitchout_in_denominator(self):
        rows = [row(i, call=call) for i, call in enumerate(('S', 'W', 'T', 'Q', 'X', 'F'))]
        selected, excluded = c.select_samples(rows)
        self.assertEqual(len(selected), 6); self.assertEqual(excluded, {})
        np.testing.assert_array_equal(c.target_values(rows, c.TARGETS[0]), [1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(c.target_values(rows, c.TARGETS[1]), [1, 1, 1, 1, 0, 0])
        with self.assertRaises(ValueError): c.target_values(rows, 'whiff')

    def test_exclusions_and_duplicate_identity(self):
        updates = [dict(gameDate='2025-05-31'), dict(identity_status='uncertain_quarantined'),
                   dict(official_pitch_code={}), dict(official_pitch_code=dict(code='S', swingStatus=False,
                      buntAttemptStatus=False, swingMissStatus=False)), dict(batSide=None),
                   dict(pitchType=None), dict(spinRate=float('nan')), dict(pre_count=dict(balls=4, strikes=1))]
        rows = [dict(row(i), **update) for i, update in enumerate(updates)]
        selected, exclusions = c.select_samples(rows)
        self.assertFalse(selected); self.assertEqual(sum(exclusions.values()), 8)
        with self.assertRaises(ValueError): c.select_samples([row(), row()])

    def test_unknown_or_mismatched_lookup_not_labelled_as_zero(self):
        rows = [dict(row(1), callCode='UNKNOWN'), dict(row(2), detailsCode='X')]
        selected, reasons = c.select_samples(rows)
        self.assertFalse(selected); self.assertEqual(reasons, dict(call_lookup_or_details_code_mismatch=2))

    def test_features_raw_polynomial_then_center_and_pre_count_only(self):
        rows = [row(0), row(1)]
        rows[1]['pX'] = .6; rows[1]['pre_count'] = dict(balls=3, strikes=2)
        values, names, n = c.features(rows, 'both')
        self.assertAlmostEqual(values[1, names.index('pX_squared')], .36)
        self.assertEqual(values[1, names.index('count_3_2')], 1)
        changed = copy.deepcopy(rows); changed[1]['post_count'] = dict(balls=0, strikes=0)
        np.testing.assert_array_equal(values, c.features(changed, 'both')[0])
        self.assertEqual(n, 19); self.assertEqual(values.shape[1], 31)
        for model, expected in [('control', 12), ('spin', 13), ('movement', 18), ('both', 19)]:
            self.assertEqual(c.features(rows, model)[2], expected)
        with self.assertRaises(ValueError): c.features(rows, 'unknown')
        with self.assertRaises(ValueError): c.features([], 'both')

    def test_encoder_training_only_unknown_group_and_zero_variance(self):
        x = np.array([[0., 5., 0.], [2., 5., 1.], [10., 5., 0.], [14., 5., 1.]])
        g = np.array(['a', 'a', 'b', 'b'])
        e = c.Encoder().fit(x, g, 2)
        np.testing.assert_allclose(e.means['a'], [1, 5]); self.assertEqual(e.scale[1], 1)
        before = e.transform(x, g).toarray()
        a = e.transform(np.array([[10000., 5., 1.]]), ['a']).toarray()
        self.assertEqual(a[0, 1], 1)
        b = e.transform(np.array([[6.5, 5., 0.]]), ['new']).toarray()
        np.testing.assert_array_equal(b[0, 1:3], [0, 0]); self.assertAlmostEqual(b[0, 3], 0)
        np.testing.assert_array_equal(before, e.transform(x, g).toarray())
        with self.assertRaises(ValueError): e.transform([[1, 2]], ['a'])
        with self.assertRaises(ValueError): c.Encoder().fit([[np.nan]], ['a'], 1)

    def test_objective_gradient_and_stability(self):
        rng = np.random.default_rng(1); x = sparse.csr_matrix(rng.normal(size=(13, 4)))
        y = np.asarray([0, 1]*6+[0]); p = np.array([0, 1, 1, 1]); b = rng.normal(size=4)
        error = check_grad(lambda b: c.objective(b, x, y, p)[0],
                           lambda b: c.objective(b, x, y, p)[1], b)
        self.assertLess(error, 2e-6)
        self.assertTrue(np.isfinite(c.objective(b*1000, x, y, p)[0]))

    def test_logistic_matches_intercept_mle_and_rejects_invalid_target(self):
        design = sparse.csr_matrix(np.ones((20, 1))); y = np.array([0]*15+[1]*5)
        beta, fit = c.fit_logistic(design, y)
        self.assertTrue(fit['success']); self.assertAlmostEqual(expit(beta[0]), .25, places=6)
        with self.assertRaises(ValueError): c.fit_logistic(design, [3]*20)
        with self.assertRaises(ValueError): c.fit_logistic(design, [0]*20)

    def test_logistic_matches_independent_dense_newton_solution(self):
        rng = np.random.default_rng(27)
        x = np.column_stack([np.ones(500), rng.normal(size=(500, 8))])
        y = rng.binomial(1, expit(x @ np.array([-.7, .4, -.2, .5, 0, .1, -.5, .3, .1])))
        penalty = np.ones(x.shape[1]); penalty[0] = 0
        reference = np.zeros(x.shape[1])
        for _ in range(20):
            p = expit(x @ reference)
            gradient = x.T @ (y-p) - penalty*reference
            hessian = x.T @ ((p*(1-p))[:, None]*x) + np.diag(penalty)
            step = np.linalg.solve(hessian, gradient)
            reference += step
            if np.max(np.abs(step)) < 1e-12: break
        beta, fit = c.fit_logistic(sparse.csr_matrix(x), y)
        self.assertTrue(fit['success'])
        np.testing.assert_allclose(beta, reference, atol=2e-5, rtol=0)
        np.testing.assert_allclose(expit(x@beta), expit(x@reference), atol=2e-5, rtol=0)

    def test_training_eligibility_uses_only_training_rows(self):
        rows = [row(i, game=101+i%2) for i in range(20)]
        self.assertEqual(c.training_eligibility(rows), {'1:FF'})
        self.assertEqual(c.training_eligibility(rows[:-1]), set())
        self.assertEqual(c.training_eligibility([dict(r, gamePk=101) for r in rows]), set())
        np.testing.assert_array_equal(c.group_ids(rows[:2]), ['1:FF', '1:FF'])

    def test_baselines_laplace_and_unseen_fallback(self):
        mean, player = c.baseline_probabilities(['a', 'a', 'b'], [1, 0, 1], ['a', 'b', 'c'])
        np.testing.assert_allclose(mean, [2/3]*3)
        np.testing.assert_allclose(player, [.5, 2/3, 2/3])

    def test_losses_and_invalid_probability(self):
        brier, loss = c.losses([1, 0], [.8, .2])
        np.testing.assert_allclose(brier, [.04, .04]); np.testing.assert_allclose(loss, [-np.log(.8)]*2)
        with self.assertRaises(ValueError): c.losses([1], [float('nan')])
        with self.assertRaises(ValueError): c.losses([1], [2])

    def test_summary_equal_pitcher_weight_paired_bootstrap(self):
        rows = []
        for pitcher, count, probability in [(1, 100, .1), (2, 1, .9)]:
            for _ in range(count):
                rows.append(dict(pitcherId=pitcher, y=0, global_mean=.5, pitcher_mean=.5,
                                 control=.5, spin=.5, movement=.5, both=probability))
        summary = c.summarize_predictions(rows, bootstrap_n=200)
        self.assertAlmostEqual(summary['metrics']['both']['brier'], .41)
        self.assertAlmostEqual(summary['control_minus_both_brier'], -.16)
        self.assertEqual(summary, c.summarize_predictions(rows, bootstrap_n=200))

    def test_acceptance_all_conditions_required(self):
        summary = dict(swings=2000, pitchers=50, pitcher_cluster_bootstrap_95pct=[.001, .004],
                       metrics=dict(control=dict(brier=.2, log_loss=.5), both=dict(brier=.19, log_loss=.49),
                                    pitcher_mean=dict(brier=.21)))
        fits = [dict(success=True)]*20
        self.assertTrue(c.acceptance(summary, fits)['passed'])
        summary['pitcher_cluster_bootstrap_95pct'][0] = -.0001
        self.assertFalse(c.acceptance(summary, fits)['passed'])

    def test_between_and_within_are_distinct(self):
        rows = []
        for pitcher, baseline in [(1, 10), (2, 100)]:
            for i in range(24):
                r = row(i, pitcher=pitcher, game=100+i%3, call='S' if i%2 else 'X')
                r['spinRate'] = baseline + i%2
                rows.append(r)
        out = c.correlations(rows, c.TARGETS[0])
        self.assertIsNone(out['fields']['spinRate']['between_pitchers'])
        self.assertAlmostEqual(out['fields']['spinRate']['within_pitcher'], 1)
        self.assertIsNone(c.pearson([]))

    def test_crossfit_no_game_leakage_and_one_prediction_per_pitch(self):
        rng = np.random.default_rng(3); rows = []
        for pitcher in range(1, 4):
            for i in range(80):
                r = row(i, pitcher, 100+i%10, call='S' if rng.random() < .3 else 'X')
                r['atBatIndex'] += 100*pitcher
                r['spinRate'] += rng.normal()*20; r['pX'] = rng.normal()
                rows.append(r)
        summary, predictions = c.crossfit(rows, c.TARGETS[0])
        self.assertEqual(len(predictions), len(rows)); self.assertEqual(len(summary['fits']), 20)
        for fold in summary['folds']:
            self.assertFalse(set(fold['train_games']) & set(fold['test_games']))
        self.assertFalse(summary['preregistered_primary_checks']['passed'])
        self.assertEqual(len({(r['gamePk'],r['atBatIndex'],r['pitchIndex']) for r in predictions}), len(rows))
        self.assertTrue(summary['full_data_both_fit']['success'])

    def test_crossfit_single_class_is_not_success(self):
        rows = [row(i, game=100+i%10) for i in range(80)]
        summary, predictions = c.crossfit(rows, c.TARGETS[0])
        self.assertFalse(predictions); self.assertEqual(len(summary['fits']), 20)
        self.assertTrue(all(not f['success'] for f in summary['fits']))

    def test_run_hashes_exclusions_and_main(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src = root/'rows.json'; src.write_text(json.dumps([row()]), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                result = c.run(src, root/'results', root/'local')
            self.assertFalse(result['primary_passed']); self.assertEqual(result['raw_pitches'], 1)
            self.assertEqual(len(c.digest(src)), 64)
            self.assertTrue((root/'local'/'predictions.json').is_file())
            with patch('sys.argv', ['conditional', '--input', str(src)]), patch.object(c, 'run') as run:
                c.main(); self.assertEqual(run.call_args[0][0], src)


if __name__ == '__main__':
    unittest.main()
