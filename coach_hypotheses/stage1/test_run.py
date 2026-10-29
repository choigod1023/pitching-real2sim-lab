import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from .run import analyze, file_hash, main
from .test_metrics import fixture_tables, fixture_metadata, fixture_poi
from .metrics import COLUMNS


class EntryTests(unittest.TestCase):
    def test_streaming_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'example'
            path.write_bytes(b'abc')
            self.assertEqual(file_hash(path), hashlib.sha256(b'abc').hexdigest())

    def test_cli_dispatch_and_reject_external_output(self):
        with patch('sys.argv', ['stage1', '--data', 'dummy']), patch('coach_hypotheses.stage1.run.analyze', return_value={'test': True}) as mocked, patch('builtins.print'):
            main()
            mocked.assert_called_once()
        with patch('sys.argv', ['stage1', '--data', 'dummy', '--output', str(Path.home())]):
            with self.assertRaises(SystemExit):
                main()

    def test_real_csv_pipeline_preserves_sources_and_fails_sample_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root/'data', root/'output'
            (data/'poi').mkdir(parents=True)
            (data/'full_sig').mkdir()
            metadata, poi, tables = [], [], {name: [] for name in COLUMNS}
            for player in range(5):
                for trial in range(2):
                    key = f'{player}_{trial}'
                    metadata.append(dict(fixture_metadata(), user=player, session=player, session_pitch=key,
                                         pitch_speed_mph=75+player+trial))
                    poi.append(dict(fixture_poi(), session_pitch=key, pitch_speed_mph=75+player+trial))
                    for name, frame in fixture_tables().items():
                        frame['session_pitch'] = key
                        tables[name].append(frame)
            pd.DataFrame(metadata).to_csv(data/'metadata.csv', index=False)
            pd.DataFrame(poi).to_csv(data/'poi/poi_metrics.csv', index=False)
            for name, blocks in tables.items():
                pd.concat(blocks).to_csv(data/f'full_sig/{name}.csv', index=False)
            (data/'data_dictionary.csv').write_text('synthetic_test_only', encoding='utf-8')
            before = {str(p): file_hash(p) for p in data.rglob('*.csv')}
            result = analyze(data, output)
            self.assertEqual((result['included_players'], result['included_pitches']), (5,10))
            self.assertEqual(result['gate']['verdict'], 'FAIL')
            self.assertEqual(before, {str(p): file_hash(p) for p in data.rglob('*.csv')})
            self.assertTrue((output/'associations.csv').is_file())
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['sources']), 7)


if __name__ == '__main__':
    unittest.main()
