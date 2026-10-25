import unittest
import pandas as pd
import numpy as np
from weighted_linkage import audit,decode
from weighted_track_quality import quality


class LinkageTests(unittest.TestCase):
    def test_encodings_are_explicit_and_suffix_is_not_silently_removed(self):
        self.assertEqual(decode('711','literal'),711)
        self.assertEqual(decode('711','three_digit_tenths'),71.1)
        self.assertIsNone(decode('818y','three_digit_tenths'))
        self.assertEqual(decode('818y','three_digit_tenths_strip_y'),81.8)
        self.assertEqual(decode('89','three_digit_tenths'),89)

    def test_duplicate_table_rows_and_colliding_files_are_ambiguous(self):
        f=dict(kind='dynamic',subject=1,weight_oz=5,speed_token='80.0',archive_path='a')
        table=pd.DataFrame([dict(subject=1,ball_weight='5oz',pitch_velocity=80.)]*2)
        r=audit({'files':[f]},table,'literal')
        self.assertEqual(r['summary']['mutually_unique_candidates'],0)
        r=audit({'files':[f,dict(f,archive_path='b')]},table.iloc[:1],'literal')
        self.assertEqual(r['summary']['mutually_unique_candidates'],0)

    def test_order_and_small_float_noise_do_not_change_candidate_identity(self):
        f=dict(kind='dynamic',subject=1,weight_oz=5,speed_token='80.0',archive_path='a')
        table=pd.DataFrame([dict(subject=1,ball_weight='5oz',pitch_velocity=80.000003),dict(subject=2,ball_weight='5oz',pitch_velocity=80.)])
        a=audit({'files':[f]},table,'literal');b=audit({'files':[f]},table.iloc[::-1],'literal')
        self.assertEqual(a['files'],b['files']);self.assertTrue(a['files'][0]['mutually_unique_candidate'])
        self.assertFalse(a['files'][0]['confirmed_identity'])

    def test_quality_excludes_residual_gap_and_degenerate_frame(self):
        points=np.zeros((4,3,4));points[:3]=np.array([[.1,0,0],[0,.03,0],[0,-.03,0]]).T[:,:,None]
        points[3,0,1]=-1;points[:3,:,3]=0
        c={'parameters':{'POINT':{'LABELS':{'value':['RFIN','RWRA','RWRB']},'UNITS':{'value':['m']}}},'data':{'points':points}}
        r=quality(c,'R');self.assertEqual(r['valid_frames'],2);self.assertEqual(r['longest_valid_run_frames'],1)
