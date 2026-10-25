import unittest
from weighted_manifest import parse_name,build


class ManifestTests(unittest.TestCase):
    def test_speed_not_guessed_and_spacing_preserved(self):
        name='Motion Data/Subject 15/6oz throw 3 711.c3d'
        r=parse_name(name)
        self.assertEqual(r['archive_path'],name);self.assertEqual(r['speed_token'],'711')
        self.assertIsNone(r['speed_kmh'])
        self.assertEqual(parse_name('Motion Data/Subject 2/3 oz Throw 4 89.1.c3d')['repeat'],4)

    def test_duplicate_and_missing_are_not_silently_filled(self):
        item={'file':'Motion Data/Subject 1/3oz throw 1 80.1.c3d','labels':['FKA-RFIN','RWRA','RWRB'],'rate':240,'event_labels':[]}
        r=build({'files':[item,item]})
        self.assertEqual(r['summary']['duplicate_keys'],[[1,3,1]])
        self.assertEqual(len(r['summary']['missing_expected_repeats']),14)
        self.assertFalse(r['files'][0]['right_hand_exact'])
