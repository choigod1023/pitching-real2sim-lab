"""Acquisition/export unit tests. Synthetic rows are not research observations."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import collect as t
from ..pitch_quality import mlb_feed as m
from ..pitch_quality.test_mlb_feed import fixture


def rows_fixture():
    feed, _, codes = fixture()
    rows, _ = m.parse_feed(feed, m.pitch_code_lookup(codes), expected_game_pk=10)
    rows[0].update(gameDate="2025-07-15", playId="test-one")
    rows[1].update(gameDate="2025-07-16", playId="test-two")
    return rows


class ExportTests(unittest.TestCase):
    def test_streaming_roundtrip_handles_unicode_nested_strings_and_small_chunks(self):
        rows = rows_fixture()
        rows[0]["test_nested"] = {"한글": ["],{", "텍스트", None, False]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"source.json"
            path.write_text(" \n"+json.dumps(rows, ensure_ascii=False)+" \n", encoding="utf-8")
            for chunk in (1, 7, 65536):
                self.assertEqual(list(t.iter_json_rows(path, chunk)), rows)
            with self.assertRaises(ValueError):
                list(t.iter_json_rows(path, 0))

    def test_streaming_rejects_bad_array_syntax_without_silently_dropping_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"source.json"
            for bad in ('{}', '[1]', '[{},]', '[{} {}]', '[{}', '[{}] trailing', ''):
                path.write_text(bad, encoding="utf-8")
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    list(t.iter_json_rows(path, 2))
            path.write_text('[]', encoding="utf-8")
            self.assertEqual(list(t.iter_json_rows(path, 1)), [])

    def test_split_boundaries_source_hashes_sort_order_and_every_value_preserved(self):
        rows = rows_fixture()
        with tempfile.TemporaryDirectory() as temp:
            root, path = Path(temp), Path(temp)/"source.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            source = dict(t.file_hash(path), rows=2)
            exports, stats = t.export_splits(path, root, ("dev_train", "dev_validation"), source)
            self.assertEqual(stats["pitch_count"], 2)
            self.assertEqual(stats["duplicate_pitch_keys"], 0)
            for name, row in zip(("dev_train", "dev_validation"), rows):
                result = root/f"{name}.json"
                self.assertEqual(json.loads(result.read_text()), [row])
                self.assertEqual(t.file_hash(result), {key: exports[name][key] for key in ("bytes", "sha256")})
                self.assertEqual(exports[name]["source_row_file_sha256"], source["sha256"])
                self.assertEqual(exports[name]["absolute_path"], str(result.resolve()))
                self.assertFalse(result.with_suffix(".json.pending").exists())
            with self.assertRaisesRegex(ValueError, "hash_mismatch"):
                t.export_splits(path, root, ("dev_train",), dict(source, sha256="0"*64))
            with self.assertRaisesRegex(ValueError, "row_count"):
                t.export_splits(path, root, ("dev_train", "dev_validation"), dict(source, rows=3))
            with self.assertRaisesRegex(ValueError, "split_names"):
                t.export_splits(path, root, ("unknown",), source)

    def test_unsorted_or_outside_rows_do_not_publish_partial_json(self):
        rows = rows_fixture()
        with tempfile.TemporaryDirectory() as temp:
            root, path = Path(temp), Path(temp)/"source.json"
            (root/"dev_train.json").write_text("[]")
            path.write_text(json.dumps(rows[::-1]))
            with self.assertRaisesRegex(ValueError, "unsorted"):
                t.export_splits(path, root, ("dev_train", "dev_validation"), t.file_hash(path))
            self.assertEqual((root/"dev_train.json").read_text(), "[]")
            rows[0]["gameDate"] = "2025-08-01"
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "outside"):
                t.export_splits(path, root, ("dev_train", "dev_validation"), t.file_hash(path))

    def test_duplicate_keys_and_playids_are_counted_across_exports(self):
        row = rows_fixture()[0]
        with tempfile.TemporaryDirectory() as temp:
            root, path = Path(temp), Path(temp)/"source.json"
            path.write_text(json.dumps([row, row]))
            keys, ids = set(), set()
            _, stats = t.export_splits(path, root, ("dev_train",), t.file_hash(path), seen_keys=keys, seen_play_ids=ids)
            self.assertEqual((stats["duplicate_pitch_keys"], stats["duplicate_play_ids"]), (1, 1))
            _, stats = t.export_splits(path, root, ("dev_train",), t.file_hash(path), seen_keys=keys, seen_play_ids=ids)
            self.assertEqual((stats["duplicate_pitch_keys"], stats["duplicate_play_ids"]), (2, 2))

    def test_atomic_result_write_and_large_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"results"/"x.json"
            t.write_result(path, {"한글": [1, None]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"한글": [1, None]})
            self.assertFalse(path.with_suffix(".json.pending").exists())
            source = Path(temp)/"large.bin"
            source.write_bytes(b"x"*(1024*1024+1))
            self.assertEqual(t.file_hash(source)["bytes"], 1024*1024+1)


class WrapperTests(unittest.TestCase):
    def test_development_published_before_august_and_root_completeness_schema(self):
        calls, notices = [], []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def fake_collect(**kwargs):
                start = kwargs["start_date"]
                calls.append((start, kwargs["end_date"]))
                if start == "2025-08-01":
                    self.assertTrue((root/"artifacts"/"dev_train.json").exists())
                    self.assertTrue(any(n.startswith("DEVELOPMENT_READY") for n in notices))
                rows, games = [], []
                dates = ["2025-06-01", "2025-07-16"] if start == "2025-06-01" else ["2025-08-01"]
                for game_day in dates:
                    pk = int(game_day.replace("-", ""))
                    feed, _, codes = fixture()
                    feed["gamePk"] = pk
                    data, parsing = m.parse_feed(feed, m.pitch_code_lookup(codes), expected_game_pk=pk)
                    for row in data:
                        row.update(gameDate=game_day, playId=f'{pk}-{row["playId"]}')
                    audit = m.audit_feed(feed, data, parsing)
                    rows.extend(data)
                    games.append(dict(game=dict(gamePk=pk), audit=audit, provenance=dict(source=dict(sha256="1"*64)), failure=None))
                rows.sort(key=lambda r: (r["gamePk"], r["atBatIndex"], r["pitchIndex"]))
                path = Path(kwargs["cache_dir"])/"parsed_pitches.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows), encoding="utf-8")
                audit = t.c.aggregate_audits(games, rows)
                days = len(t.c.fixed_dates(start, kwargs["end_date"]))
                audit.update(schedule_days_requested=days, schedule_days_received=days, failures=[], collection_complete=True)
                manifest = dict(pitch_level_export=dict(t.file_hash(path), rows=len(rows)))
                output = Path(kwargs["results_dir"])
                t.write_result(output/"collection_manifest.json", manifest)
                t.write_result(output/"collection_games.json", games)
                return audit, manifest, path
            with patch.object(t.c, "collect", side_effect=fake_collect):
                audit, manifest = t.run(artifact_dir=root/"artifacts", results_dir=root/"results", reuse_dirs=(), progress=notices.append)
            self.assertEqual(calls, [("2025-06-01", "2025-07-31"), ("2025-08-01", "2025-08-31")])
            self.assertEqual(audit["pitch_count"], 6)
            self.assertEqual(audit["schedule_days_received"], 92)
            self.assertEqual(audit["selected_game_count"], audit["parsed_game_count"])
            for key in ("interval", "collection_complete", "schedule_days_requested", "schedule_days_received",
                        "selected_game_count", "parsed_game_count", "failures", "split_exports"):
                self.assertEqual(audit[key], manifest[key])
            self.assertEqual(manifest["final_train"]["rows"], 4)
            self.assertFalse((root/"artifacts"/"final_train.json").exists())
            self.assertEqual(audit["duplicate_pitch_keys"], 0)
            self.assertEqual(audit["duplicate_play_ids"], 0)
            self.assertFalse(manifest["relationships_computed"])
            game_path = root/"results"/"collection_development"/"collection_games.json"
            self.assertEqual(len(game_path.read_text(encoding="utf-8").splitlines()), 4)
            self.assertEqual(len(json.loads(game_path.read_text(encoding="utf-8"))), 2)
            self.assertEqual(manifest["phases"][0]["games_file"], t.file_hash(game_path))
            audit.update(collection_complete=False, failures=[dict(stage="synthetic_fetch_failure")])
            t.write_result(root/"results"/"collection_audit.json", audit)
            incomplete, saved_manifest = t.verify_export_metadata(root/"results", root/"artifacts")
            self.assertFalse(incomplete["collection_complete"])
            self.assertEqual(incomplete["failures"], saved_manifest["failures"])
            audit.update(collection_complete=True, failures=[], schedule_days_received=91)
            t.write_result(root/"results"/"collection_audit.json", audit)
            with self.assertRaisesRegex(ValueError, "inconsistent_complete"):
                t.verify_export_metadata(root/"results", root/"artifacts")
            audit["schedule_days_received"] = 92
            t.write_result(root/"results"/"collection_audit.json", audit)
            (root/"artifacts"/"final_test.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "partition_hash"):
                t.verify_export_metadata(root/"results", root/"artifacts")

    def test_cli_keeps_collection_separate_from_model(self):
        result = dict(pitch_count=2, parsed_game_count=1, collection_complete=False)
        with patch.object(t, "run", return_value=(result, {})) as run, contextlib.redirect_stdout(io.StringIO()) as out:
            t.main(["--workers", "2"])
        self.assertEqual(run.call_args.kwargs["workers"], 2)
        self.assertEqual(json.loads(out.getvalue()), result)


if __name__ == "__main__":
    unittest.main()
