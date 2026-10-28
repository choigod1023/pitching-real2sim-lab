import unittest
import numpy as np
import pandas as pd
from speed_association_audit import FEATURES, ridge_fit, evaluate


class TestAssociation(unittest.TestCase):
    def test_zero_variance_and_fixed_shrinkage(self):
        np.testing.assert_allclose(ridge_fit(np.zeros((4, 2)), np.arange(4)), 0)
        x = np.array([-2., -1., 1., 2.])
        coefficient = ridge_fit(x[:,None], 3*x)[0]
        self.assertAlmostEqual(coefficient, 3*4/14)

    def test_heldout_nonanchor_labels_cannot_change_own_predictions(self):
        rows = []
        for user in range(3):
            for trial in range(4):
                row = dict(user=user, session=user, session_pitch=f'{user}_{trial}', speed_kmh=100+user+trial)
                row.update({name: (i+1)*trial+user for i,name in enumerate(FEATURES)})
                rows.append(row)
        frame = pd.DataFrame(rows)
        a, _ = evaluate(frame)
        changed = frame.copy()
        changed.loc[(changed.user == 1) & (changed.session_pitch != '1_0'), 'speed_kmh'] += 50
        b, _ = evaluate(changed)
        np.testing.assert_allclose(a[a.user == 1].predicted_delta_kmh, b[b.user == 1].predicted_delta_kmh)
        self.assertTrue((b[b.user == 1].actual_delta_kmh > a[a.user == 1].actual_delta_kmh).all())


if __name__ == '__main__':
    unittest.main()
