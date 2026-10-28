import copy
import tempfile
import unittest
from pathlib import Path
from research_gates import CHECKS,PROTOCOL,sha,verify,initialize,atomic_write,advance,read


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.data=self.root/'measurement.json'
        self.data.write_text('{"synthetic_fixture_only":true}',encoding='utf-8')
        self.state=self.root/'state.json';initialize(self.state)

    def receipt(self,stage='data'):
        return {'stage':stage,'protocol_sha256':sha(PROTOCOL),'dataset_pitch_scope':'synthetic fixture, not research pass',
                'artifacts':{'m':{'kind':'measurement','path':str(self.data),'sha256':sha(self.data)}},
                'checks':{k:{'status':'PASS','evidence':['m'],'reason':'Synthetic test of progression machinery, not scientific evidence.'} for k in CHECKS[stage]}}

    def save(self,value,name='receipt.json'):
        p=self.root/name;atomic_write(p,value);return p

    def test_unknown_check_blocks_progress_and_preserves_state(self):
        r=self.receipt();r['checks']['release_window_usable']['status']='UNKNOWN'
        result=advance(self.state,self.save(r))
        self.assertFalse(result['passed']);self.assertEqual(read(self.state)['current_stage'],'data')

    def test_code_tests_are_not_scientific_validation(self):
        r=self.receipt();r['artifacts']['m']['kind']='tests'
        self.assertTrue(any('tests_are_not_scientific' in x for x in verify('data',r,self.root)['errors']))

    def test_changed_or_missing_artifact_blocks(self):
        r=self.receipt();self.data.write_text('changed',encoding='utf-8')
        self.assertFalse(verify('data',r,self.root)['passed'])
        self.data.unlink();self.assertFalse(verify('data',r,self.root)['passed'])

    def test_missing_check_evidence_reason_and_scope_block(self):
        for field in ('evidence','reason'):
            r=self.receipt();del r['checks']['same_pitch_identity'][field]
            self.assertFalse(verify('data',r,self.root)['passed'])
        r=self.receipt();r.pop('dataset_pitch_scope');self.assertFalse(verify('data',r,self.root)['passed'])

    def test_cannot_skip_stages(self):
        self.assertFalse(advance(self.state,self.save(self.receipt('candidate')))['passed'])

    def test_editing_current_stage_cannot_skip_verified_history(self):
        state=read(self.state);state['current_stage']='candidate';atomic_write(self.state,state)
        with self.assertRaises(ValueError):advance(self.state,self.save(self.receipt('candidate')))

    def test_pass_only_moves_one_stage(self):
        result=advance(self.state,self.save(self.receipt()))
        self.assertTrue(result['passed']);self.assertEqual(result['next_stage'],'hand_ball')

    def test_prior_receipt_tampering_blocks_later_stage(self):
        prior=self.save(self.receipt());advance(self.state,prior)
        prior.write_text('{}',encoding='utf-8')
        with self.assertRaises(ValueError):advance(self.state,self.save(self.receipt('hand_ball'),'second.json'))

    def test_prior_data_tampering_blocks_later_stage(self):
        advance(self.state,self.save(self.receipt()));self.data.write_text('changed',encoding='utf-8')
        with self.assertRaises(ValueError):advance(self.state,self.save(self.receipt('hand_ball'),'second.json'))

    def test_init_cannot_reset_history(self):
        with self.assertRaises(FileExistsError):initialize(self.state)

    def test_existing_lock_blocks_duplicate_advance(self):
        self.state.with_name('state.json.lock').touch()
        with self.assertRaises(FileExistsError):advance(self.state,self.save(self.receipt()))

    def test_protocol_change_requires_review(self):
        r=self.receipt();r['protocol_sha256']='0'*64
        self.assertFalse(advance(self.state,self.save(r))['passed'])


if __name__=='__main__':unittest.main()
