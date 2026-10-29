"""Source linkage and engineering checks; fixtures never count as research evidence."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.io import savemat

from .audit import (archive_trials, column_index, digest, extract_member, inspect_c3d,
                    inspect_mat_cohort, link_trials, read_trial_book, select_new_trials)
from .audit import status_code
from .requirements import (RELEASE_REQUIRED, readiness, scale_bias_kmh,
                           slope_noise_sd_kmh, timing_budget_ms)
from . import run as run_module


def fixture_book(path):
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}"><si><t>1w2d</t></si><si><t>DP_07</t></si><si><t>O</t></si></sst>')
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="S1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData><row><c r="A1" t="s"><v>0</v></c></row><row><c r="A2" t="s"><v>1</v></c><c r="B2" t="s"><v>2</v></c><c r="C2"><v>44</v></c></row></sheetData></worksheet>')


class SourceAuditTests(unittest.TestCase):
    def test_digest_and_cell_indices(self):
        self.assertEqual(column_index('A1'), 0)
        self.assertEqual(column_index('AA19'), 26)
        with self.assertRaises(ValueError):
            column_index('A0')
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'input'; p.write_bytes(b'known')
            self.assertEqual(digest(p), hashlib.sha256(b'known').hexdigest())

    def test_workbook_preserves_raw_speed_and_unknown_unit(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'book.xlsx'; fixture_book(p)
            rows = read_trial_book(p, 'S1')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['speed_raw'], '44')
            self.assertEqual(rows[0]['speed_unit'], 'UNKNOWN')
            self.assertEqual(rows[0]['speed_cell'], 'C2')
            self.assertEqual(rows[0]['trial_number'], 7)
            with self.assertRaises(ValueError):
                read_trial_book(p, 'S2')

    def test_archive_keys_and_mismatched_subject(self):
        rows = archive_trials(['S1/1w2d/S1_DP_07_O.c3d', 'S1/1w2d/S1_static_01_D.c3d'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['trial_number'], 7)
        with self.assertRaises(ValueError):
            archive_trials(['S2/1w2d/S1_DP_07_O.c3d'])
        for name in ('../x.c3d', '/tmp/x.c3d', 'C:/tmp/x.c3d', '..\\x.c3d'):
            with self.subTest(member=name), self.assertRaises(ValueError):
                archive_trials([name])

    def test_linkage_preserves_missing_duplicate_and_status_mismatch(self):
        trials = archive_trials(['S1/1w2d/S1_DP_07_O.c3d'])
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'book.xlsx'; fixture_book(p)
            book = read_trial_book(p, 'S1')
        linked = link_trials(trials, book)
        self.assertEqual(linked[0]['workbook_matches'], 1)
        self.assertTrue(linked[0]['status_agrees'])
        self.assertTrue(linked[0]['speed_record_present'])
        self.assertEqual(link_trials(trials, book*2)[0]['workbook_matches'], 2)
        self.assertEqual(link_trials(trials, [])[0]['workbook_matches'], 0)
        book[0]['status_raw'] = 'M-XIPH'
        self.assertFalse(link_trials(trials, book)[0]['status_agrees'])

    def test_status_details_are_not_a_repaired_quality_code(self):
        self.assertEqual(status_code('WM-XIPH'), 'WM')
        self.assertEqual(status_code('M'), 'M')
        self.assertNotEqual(status_code('O'), status_code('M-XIPH'))
        self.assertIsNone(status_code(None))
        self.assertIsNone(status_code('unclear'))
        trials = archive_trials(['S1/1w2d/S1_DP_07_unclear.c3d'])
        book = [dict(trials[0], speed_raw=None)]
        self.assertIsNone(link_trials(trials, book)[0]['status_code_agrees'])

    def test_selection_uses_chronology_and_name_not_speed(self):
        names = ['S1/2w1d/S1_DP_01_O.c3d', 'S1/1w2d/S1_DP_09_O.c3d',
                 'S1/1w2d/S1_DP_08_O.c3d', 'S1/1w1d/S1_DP_01_M.c3d']
        with tempfile.TemporaryDirectory() as temp:
            choice = select_new_trials(archive_trials(names), temp, limit=2)
            self.assertEqual(choice, names[2:0:-1])
            existing = Path(temp)/names[2]; existing.parent.mkdir(parents=True); existing.write_bytes(b'old')
            self.assertEqual(select_new_trials(archive_trials(names), temp, 1), [names[1]])

    def test_extraction_never_follows_archive_paths_or_overwrites_input(self):
        payload = bytes([2, 80])+bytes(510)
        completed = subprocess.CompletedProcess([], 0, stdout=payload, stderr=b'')
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.run', return_value=completed) as execute:
            p = extract_member('archive', 'S1/1w2d/S1_DP_07_O.c3d', temp)
            self.assertEqual(p.read_bytes(), payload)
            p.write_bytes(b'preserved')
            with self.assertRaises(FileExistsError):
                extract_member('archive', 'S1/1w2d/S1_DP_07_O.c3d', temp)
            for member in ('../x.c3d', '/tmp/x.c3d', 'C:/tmp/x.c3d', '..\\x.c3d'):
                with self.subTest(member=member), self.assertRaises(ValueError):
                    extract_member('archive', member, temp)
            execute.assert_called()

    def test_c3d_marker_validity_is_not_center_or_contact_truth(self):
        names = ['Ball', 'R3MC', 'R5MC', 'RFR1', 'RFR2', 'RFR3', 'RFR4',
                 'LASI', 'RASI', 'LPSI', 'RPSI', 'C7', 'T8', 'STRN', 'XIPH']
        values = np.ones((4, len(names), 100)); values[3, 0, 50] = -1
        source = {'parameters': {'POINT': {'LABELS': {'value': names}, 'RATE': {'value': [200.]}, 'UNITS': {'value': ['mm']}},
                                  'EVENT': {'LABELS': {'value': ['Event']}, 'TIMES': {'value': np.array([[0.], [.3]])}}},
                  'data': {'points': values}}
        with tempfile.TemporaryDirectory() as temp, patch('ezc3d.c3d', return_value=source):
            path = Path(temp)/'fixture.c3d'; path.write_bytes(b'fixture')
            result = inspect_c3d(path)
            self.assertEqual(result['ball_valid_frames'], 99)
            self.assertFalse(result['candidate_windows'][0]['all_group_markers_valid']['ball_marker'])
            self.assertTrue(result['candidate_windows'][0]['all_group_markers_valid']['throwing_hand'])
            self.assertTrue(result['candidate_windows'][0]['all_group_markers_valid']['torso'])
            self.assertFalse(result['ball_center_validated'])
            self.assertFalse(result['physical_release_validated'])

    def test_mat_cohort_is_actual_motion_and_scalar_not_ball_track(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'Sub001').mkdir()
            savemat(root/'Sub001/001.mat', {'X': np.zeros((101, 15, 3)), 'Y': 80.})
            report = inspect_mat_cohort(root)
            self.assertEqual((report['subjects'], report['trials']), (1, 1))
            self.assertFalse(report['explicit_ball_trajectory_field_present'])
            savemat(root/'Sub001/002.mat', {'X': np.zeros((2, 2)), 'Y': 80.})
            with self.assertRaises(ValueError):
                inspect_mat_cohort(root)

    def test_run_and_cli_save_real_audit_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture_book(root/'Trialinfo.xlsx'); (root/'S1.rar').write_bytes(b'archive')
            manifest = root/'manifest.json'; manifest.write_text(json.dumps({'files': [
                {'filename': n, 'sha256': digest(root/n)} for n in ('S1.rar', 'Trialinfo.xlsx')]}))
            completed = subprocess.CompletedProcess([], 0, stdout='S1/1w2d/S1_DP_07_O.c3d\n', stderr='')
            with patch('subprocess.run', return_value=completed), \
                    patch.object(run_module, 'extract_member', return_value=root/'unused.c3d'), \
                    patch.object(run_module, 'inspect_c3d', return_value={'ball_center_validated': False}), \
                    patch.object(run_module, 'inspect_mat_cohort', return_value={'subjects': 1, 'trials': 1}), \
                    contextlib.redirect_stdout(io.StringIO()):
                args = ['run', '--raw', str(root), '--download-manifest', str(manifest), '--mat-root', str(root), '--output', str(root/'output')]
                with patch('sys.argv', args):
                    run_module.main()
                result = json.loads((root/'output/audit_results.json').read_text())
                self.assertEqual(result['ozkaya_summary']['unique_workbook_links'], 1)
                self.assertEqual(result['ozkaya_summary']['speed_unit'], 'UNKNOWN')
                self.assertFalse(result['speed_model_fitted'])
                (root/'S1.rar').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'changed_original'):
                    run_module.run(root, manifest, root, root/'output')


class RequirementsTests(unittest.TestCase):
    def test_unknown_or_false_cannot_pass_and_effect_is_never_approved(self):
        evidence = dict.fromkeys(RELEASE_REQUIRED, 'TRUE')
        self.assertTrue(readiness(evidence)['hand_ball_validation_ready'])
        self.assertFalse(readiness(evidence)['scientific_gate_passed'])
        evidence['ball_center_observation'] = 'UNKNOWN'
        self.assertFalse(readiness(evidence)['hand_ball_validation_ready'])
        self.assertTrue(readiness(evidence)['body_speed_observational_ready'])
        evidence['independent_speed_link_units'] = 'FALSE'
        self.assertFalse(readiness(evidence)['body_speed_observational_ready'])
        self.assertFalse(readiness({})['hand_ball_validation_ready'])
        with self.assertRaises(ValueError):
            readiness({'raw_access': True})

    def test_slope_noise_matches_monte_carlo(self):
        sigma_mm, fps, samples = .5, 1000, 11
        expected = slope_noise_sd_kmh(sigma_mm, fps, samples)
        rng = np.random.default_rng(20261029)
        t = np.arange(samples)/fps; centered = t-t.mean()
        observations = 40*t+rng.normal(0, sigma_mm/1000, size=(20000, samples))
        slopes = observations@centered/(centered@centered)
        observed = np.std((slopes-40)*3.6, ddof=1)
        self.assertAlmostEqual(observed/expected, 1., delta=.02)
        with self.assertRaises(ValueError):
            slope_noise_sd_kmh(.5, 0, 11)

    def test_scale_and_timing_budgets_are_conditional_arithmetic(self):
        self.assertEqual(scale_bias_kmh(150, .01), 1.5)
        self.assertEqual(scale_bias_kmh(150, -.002), .3)
        self.assertAlmostEqual(timing_budget_ms(.3, 1000), 1/12)
        with self.assertRaises(ValueError):
            timing_budget_ms(.3, 0)
        with self.assertRaises(ValueError):
            scale_bias_kmh(-150, .01)


if __name__ == '__main__':
    unittest.main()
