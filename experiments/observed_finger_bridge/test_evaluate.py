import unittest
import tempfile
from pathlib import Path
from .evaluate import region_distance,match_tips,longest_run,mean_or_none,selected,bone_variation,evaluate,freeze_regions,load


class EvaluationTests(unittest.TestCase):
    def test_freeze_without_execution_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'input.py';out=Path(tmp)/'regions.json'
            src.write_text("samples={'clip':{3:((0,0),(1,2,3,4),None)}}\nraise RuntimeError('never execute')")
            freeze_regions(src,out);before=out.read_bytes();freeze_regions(src,out)
            self.assertEqual(out.read_bytes(),before)
            self.assertEqual(load(out)['samples']['clip']['3'][1],[1,2,3,4])
            src.write_text('samples={}');self.assertRaises(ValueError,freeze_regions,src,out)

    def test_region_distance(self):
        self.assertEqual(region_distance((5,6),(0,10,0,10)),0)
        self.assertEqual(region_distance((13,14),(0,10,0,10)),5)

    def test_identity_is_unordered(self):
        rows=match_tips([(9,9),(1,1)],[(0,2,0,2),(8,10,8,10)])
        self.assertEqual([r['candidate_tip'] for r in rows],[12,8])
        self.assertEqual(sum(r['outside_region_px'] for r in rows),0)
        self.assertEqual(match_tips([],[]),[])
        with self.assertRaises(ValueError):match_tips([],[(0,2,0,2)])

    def test_missing_and_continuity(self):
        self.assertIsNone(selected({'selected_index':None,'candidates':[]}))
        self.assertEqual(longest_run([1,3,4,4,5,7]),3)
        self.assertEqual(longest_run([]),0)
        self.assertIsNone(mean_or_none([]))
        self.assertIsNone(bone_variation([]))

    def test_missing_not_zero_accuracy(self):
        row={'frame':4,'candidates':[],'selected_index':None}
        x={'clips':[{'clip_id':'a','domain':'test','methods':{'raw':[row],'rotated':[row]}}]}
        m={'clips':[{'clip_id':'a','path':'Branson4S1.mp4','samples':[{'frame':4,'role':'target'}]}]}
        t={'samples':{'Branson4S1':{'4':[[0,0],[0,2,0,2],[8,10,8,10]]}}}
        result=evaluate(x,m,t)['clips'][0]
        self.assertEqual(result['methods']['raw']['missing_tip_regions'],2)
        self.assertIsNone(result['methods']['raw']['tip_outside_region_mean_px'])
        self.assertEqual(result['input_agreement']['common_target_frames'],0)

    def test_empty_control_and_rigid_shape(self):
        world=[[float(i),0.,0.] for i in range(21)]
        hand={'world21':world}
        rows=[{'candidates':[hand],'selected_index':0}]*2
        self.assertEqual(bone_variation(rows)['median_segment_cv_percent'],0)


if __name__=='__main__':unittest.main()
