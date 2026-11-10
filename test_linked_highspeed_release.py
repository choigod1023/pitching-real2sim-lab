import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape
from zipfile import ZipFile

import numpy as np

from experiments.linked_highspeed_release import study


class LinkedHighSpeedReleaseTests(unittest.TestCase):
    def test_workbook_links_join_on_cell_row_and_skip_nonfastball_missing_spin(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        tmp_path = Path(temp.name)
        header = [
            "Game Pitch", "Pitch Type", "HS Vid", "RelSpeed (MPH)", "SpinRate",
            "TrueSpinRelease", "Spin Direction", "InducedVertBreak (Inches)",
            "HorzBreak (Inches)", "HorzRelAngle", "VertRelAngle", "RelHeight (Feet)",
        ]
        strings = header + ["FB", "0:00", "SL"]
        columns = "ABCDEFGHIJKL"
        header_xml = "".join(
            f'<c r="{column}1" t="s"><v>{index}</v></c>'
            for index, column in enumerate(columns)
        )
        data_xml = "".join(
            f'<c r="{column}6"{attr}><v>{value}</v></c>'
            for column, value, attr in [
                ("A", 41, ""), ("B", 12, ' t="s"'), ("D", 80, ""), ("E", 1900, ""),
                ("F", 1700, ""), ("G", 13, ' t="s"'), ("H", 14, ""), ("I", 6, ""),
                ("J", 3, ""), ("K", -1, ""), ("L", 5.4, ""),
            ]
        )
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        source = tmp_path / "pitches.xlsx"
        with ZipFile(source, "w") as workbook:
            workbook.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">' + "".join(
                f"<si><t>{escape(value)}</t></si>" for value in strings
            ) + "</sst>")
            workbook.writestr("xl/worksheets/sheet1.xml",
                f'<worksheet xmlns="{ns}" xmlns:r="{rel_ns}"><sheetData>'
                f'<row r="1">{header_xml}</row><row r="6">{data_xml}</row>'
                '<row r="7"><c r="A7"><v>42</v></c><c r="B7" t="s"><v>14</v></c></row>'
                '</sheetData><hyperlinks><hyperlink ref="C7" r:id="r2"/>'
                '<hyperlink ref="C6" r:id="r1"/></hyperlinks></worksheet>')
            workbook.writestr("xl/worksheets/_rels/sheet1.xml.rels",
                '<Relationships><Relationship Id="r1" Target="https://example.org/41"/>'
                '<Relationship Id="r2" Target="https://example.org/42"/></Relationships>')
        first, second = study.read_pitches(source)
        assert first["video"].endswith("/41")
        assert first["rpm"] == 1900
        assert second == {"pitch_id": 42, "pitch_type": "SL", "video": "https://example.org/42"}

        result = study.audit(source, {"41": 2000})
        row = result["four_seams"][0]
        assert result["linked_video_count"] == 2
        assert not result["linked_four_seam_pairs"]
        np.testing.assert_allclose(row["recorded_hb_ivb_cm"], [15.24, 35.56])
        assert row["optical_minus_recorded_rpm"] == 100
        assert row["optical_input_hb_ivb_cm"][1] > row["conditional_hb_ivb_cm"][1]
        json.dumps(result, allow_nan=False)


    def test_spin_clock_controls_movement_and_converges(self):
        pitch = dict(speed_mph=80, horizontal_angle_deg=0, vertical_angle_deg=0,
                     true_rpm=1600, direction="12:00", release_height_ft=6)
        vertical, path = study.predict_movement(pitch)
        np.testing.assert_allclose(vertical[0], 0, atol=1e-8)
        assert vertical[1] > 0
        assert path[-1, 1] == 16.8
        right, _ = study.predict_movement({**pitch, "direction": "1:30"})
        left, _ = study.predict_movement({**pitch, "direction": "10:30"})
        np.testing.assert_allclose(left, [-right[0], right[1]])
        assert right[0] > 0
        fine, _ = study.predict_movement(pitch, dt=.0005)
        np.testing.assert_allclose(vertical, fine, atol=1e-3)
        with self.assertRaisesRegex(ValueError, "clock direction"):
            study.predict_movement({**pitch, "direction": "01:60"})
