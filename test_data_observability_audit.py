import hashlib
import tempfile
import unittest
from pathlib import Path
from data_observability_audit import digest,check_reference,velocity_displacement_budget,frame_overlap


class SourceAuditTests(unittest.TestCase):
    def test_hash_uses_file_contents(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data';p.write_bytes(b'known original')
            self.assertEqual(digest(p),hashlib.sha256(b'known original').hexdigest())
            record={'path':str(p),'sha256':digest(p)}
            self.assertTrue(check_reference(record)['hash_matches'])
            p.write_bytes(b'changed');self.assertFalse(check_reference(record)['hash_matches'])

    def test_missing_source_is_not_passed(self):
        with tempfile.TemporaryDirectory() as d:
            r=check_reference({'path':str(Path(d)/'missing'),'sha256':'0'*64})
            self.assertFalse(r['exists']);self.assertFalse(r['hash_matches']);self.assertIsNone(r['actual_sha256'])

    def test_no_temporal_overlap_is_preserved(self):
        self.assertEqual(frame_overlap(range(935,965),range(973,980)),[])
        self.assertEqual(frame_overlap([1,2,2,3],[3,4]),[3])

    def test_displacement_units(self):
        self.assertAlmostEqual(velocity_displacement_budget(3.6,1000),1.)
        self.assertAlmostEqual(velocity_displacement_budget(1,360),.7716049382716049)

    def test_invalid_sensitivity_inputs_fail(self):
        for target,rate in ((0,360),(1,0),(-1,360),(float('nan'),360)):
            with self.assertRaises(ValueError):velocity_displacement_budget(target,rate)


if __name__=='__main__':unittest.main()
