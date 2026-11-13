"""Check observed units, identity omissions, and exact source row linkage."""
import io
import unittest
from zipfile import ZipFile

from .evidence import LINKS, number_or_none, prime_listing_record, read_workbook, record_from_row


class EvidenceTests(unittest.TestCase):
    def test_unit_conversion(self):
        self.assertAlmostEqual(number_or_none('79.4', 1.609344), 127.7819136)
        self.assertAlmostEqual(number_or_none('13.9', 2.54), 35.306)
        self.assertEqual(number_or_none('0'), 0.)

    def test_missing_and_nonfinite(self):
        for value in ('', ' ', None):
            self.assertIsNone(number_or_none(value))
        for value in ('nan', 'inf', '-inf'):
            with self.assertRaises(ValueError):
                number_or_none(value)

    def test_record_keeps_unknown_identity_and_metrics(self):
        row = {'xlsx_row': 25, 'values': {'Game Pitch': '4', 'HS Vid': LINKS[4],
               'RelSpeed (MPH)': '79.4', 'Pitcher First Name': '', 'Pitcher Last Name': ''}}
        result = record_from_row(row, 'Branson4S1')
        self.assertEqual(result['clip_id'], 'Branson4S1')
        for key in ('pitcher_first_name', 'pitcher_last_name', 'rpm', 'ivb_cm', 'hb_cm'):
            self.assertIsNone(result[key])
        self.assertFalse(result['same_pitcher_verified'])
        self.assertAlmostEqual(result['recorded_speed_kmh'], 127.7819136)

    def test_mismatched_video_is_rejected(self):
        row = {'xlsx_row': 25, 'values': {'Game Pitch': '4', 'HS Vid': LINKS[8]},
               'video_hyperlink': LINKS[4]}
        with self.assertRaises(ValueError):
            record_from_row(row, 'Branson4S1')
        row.pop('video_hyperlink')
        with self.assertRaises(ValueError):
            record_from_row(row, 'Branson4S1')

    def test_prime_exact_event_and_missing_spin(self):
        text = '12,Edgertronic Monochrome,4,360,Movement,Dynamic,Human,Baseball Throw - 69.1 MPH\n13,OptiTrack Prime Colors,8,360,Movement,Dynamic,Human,Baseball Throw - 62.7 MPH\n'
        record = prime_listing_record(text)
        self.assertAlmostEqual(record['recorded_speed_kmh'], 100.9058688)
        self.assertEqual(record['source_line'], 2)
        self.assertIsNone(record['rpm'])
        self.assertIsNone(record['ivb_cm'])
        self.assertIsNone(record['hb_cm'])
        for bad in ('', text+text, text.replace('8,360', '8,300')):
            with self.assertRaises(ValueError):
                prime_listing_record(bad)

    def test_xlsx_hyperlink_and_row(self):
        buffer = io.BytesIO()
        ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
        with ZipFile(buffer, 'w') as archive:
            archive.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}"><si><t>Game Pitch</t></si><si><t>HS Vid</t></si></sst>')
            archive.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row r="25"><c r="A25"><v>4</v></c><c r="B25" t="inlineStr"><is><t>{LINKS[4]}</t></is></c></row></sheetData><hyperlinks><hyperlink ref="B25" r:id="rId1"/></hyperlinks></worksheet>')
            archive.writestr('xl/worksheets/_rels/sheet1.xml.rels', f'<Relationships><Relationship Id="rId1" Target="{LINKS[4]}"/></Relationships>')
        buffer.seek(0)
        row = read_workbook(buffer)[0]
        self.assertEqual(row['xlsx_row'], 25)
        self.assertEqual(row['video_cell_reference'], 'B25')
        self.assertEqual(row['video_hyperlink'], LINKS[4])
        self.assertEqual(row['values']['HS Vid'], LINKS[4])


if __name__ == '__main__':
    unittest.main()
