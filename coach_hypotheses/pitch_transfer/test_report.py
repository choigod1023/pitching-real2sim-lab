import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import test_study as fixtures
from .study import pair_evaluation,extra_diagnostics,transfer_pass,COHORT
from .report import compact,summarize,archive
from .render import select_pair,export
import numpy as np


class ReportTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.TransferTests();fixture.setUp()
        p,tr=pair_evaluation(fixture.q,fixture.dt,fixture.lengths,fixture.controls,fixture.bases,np.zeros(33))
        p['judgement']=transfer_pass(p,p)
        p['extra']=extra_diagnostics({'summary':fixture.bases[2][0],
            'root_peak':np.linalg.norm(tr[2]['applied_loads'][:,:3],axis=1).max()},tr[2],p['fine'])
        p['verification']={'bitwise_replay_pass':True,'accounting_pass':True}
        self.report={'pitch':'3034_2','baseline':copy.deepcopy(p),
                     'transfers':{'normalized':copy.deepcopy(p),'absolute':copy.deepcopy(p)},
                     'refit':{'selected':copy.deepcopy(p),'optimization':{'paired_evaluations':0,'messages':[],
                                                                       'baseline_fallback':True}}}

    def test_compact_keeps_costs_and_failed_status(self):
        row=compact(self.report,'normalized')
        self.assertFalse(row['transfer_pass']);self.assertTrue(row['A_pass'])
        self.assertEqual(row['gain_kmh'],0.)
        self.assertIn('net_work_change_j',row);self.assertIn('root_force_norm_peak_change_n',row)
        self.assertTrue(compact(self.report,'refit')['baseline_fallback'])

    def test_cohort_requires_all_unique_pitches(self):
        with self.assertRaises(ValueError):summarize([self.report])
        with self.assertRaises(ValueError):summarize([self.report]*9)
        reports=[]
        for p in COHORT:
            r=copy.deepcopy(self.report);r['pitch']=p;reports.append(r)
        result=summarize(reports)
        self.assertEqual(result['same_athlete_refit_count'],4)
        self.assertEqual(result['frozen_transfer_counts']['same_athlete']['normalized']['passed'],0)
        self.assertIsNone(result['video_candidate'])

    def test_deterministic_archives_preserve_all_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for p in COHORT:
                r=copy.deepcopy(self.report);r['pitch']=p
                (root/'in'/p).mkdir(parents=True)
                (root/'in'/p/'report.json').write_text(json.dumps(r),encoding='utf8')
            before=archive(root/'in',root/'a');after=archive(root/'in',root/'b')
            self.assertEqual(before['archives'],after['archives'])
            self.assertEqual((root/'a/3034_1.json.gz').read_bytes(),(root/'b/3034_1.json.gz').read_bytes())

    def test_video_rejects_zero_failed_and_source_results(self):
        with self.assertRaises(ValueError):select_pair(self.report,'refit')
        self.report['pitch']='3034_1'
        with self.assertRaises(ValueError):select_pair(self.report,'normalized')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'r.json';path.write_text(json.dumps(self.report),encoding='utf8')
            with self.assertRaises(ValueError):export(path,'normalized',temp,Path(temp)/'media')
            self.assertFalse((Path(temp)/'media').exists())


if __name__=='__main__':unittest.main()
