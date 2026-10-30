"""Synthetic edge-case tests; these fixtures are never collection observations."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from . import collect_games as c
from . import mlb_feed as m
from .test_mlb_feed import fixture, Response


def fake_opener(url, **kwargs):
    feed, schedule, codes = fixture()
    value = schedule if "/schedule?" in url else codes if url == m.PITCH_CODES_URL else feed
    return Response(json.dumps(value).encode(), url=url)


class CollectionTests(unittest.TestCase):
    def test_fixed_interval_inclusive_and_limits(self):
        days = c.fixed_dates()
        self.assertEqual((len(days), days[0], days[-1]), (14, "2025-06-01", "2025-06-14"))
        for workers, retries in ((0, 2), (4, 2), (True, 2), (3, 3), (3, -1)):
            with self.assertRaises(ValueError):
                c.collect(workers=workers, retries=retries)
        with self.assertRaises(ValueError):
            c.fetch_with_retries(m.SCHEDULE_URL, "unused", retries=3)

    def test_custom_interval_is_inclusive_and_cannot_overwrite_default_outputs(self):
        self.assertEqual(len(c.fixed_dates("2025-06-01", "2025-08-31")), 92)
        self.assertEqual(c.fixed_dates("2025-07-31", "2025-08-01"), ["2025-07-31", "2025-08-01"])
        for first, last in (("2025-08-01", "2025-07-01"), (True, "2025-08-01"), ("bad", "2025-08-01")):
            with self.assertRaises(ValueError):
                c.fixed_dates(first, last)
        with self.assertRaisesRegex(ValueError, "separate_cache"):
            c.collect(start_date="2025-07-01", end_date="2025-07-02")
        _, schedule, _ = fixture()
        game = schedule["dates"][0]["games"][0]
        game["officialDate"] = "2025-08-01"
        selected, _ = c.select_games([("date", schedule)], start_date="2025-08-01", end_date="2025-08-31")
        self.assertEqual(selected[0]["gameDate"], "2025-08-01")
        self.assertEqual(c.select_games([("date", schedule)])[0], [])

    def test_valid_prior_cache_reused_without_network_and_bad_hash_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp)/"source", Path(temp)/"destination"
            value, metadata = m.fetch_json(m.SCHEDULE_URL, source, opener=fake_opener)
            result = c.fetch_with_retries(m.SCHEDULE_URL, destination, reuse_dirs=(source,),
                                         opener=lambda *a, **k: self.fail("unexpected GET"))
            self.assertEqual(result["value"], value)
            self.assertEqual(result["source"]["sha256"], metadata["sha256"])
            self.assertEqual(result["source"]["cache_origin"], "validated_prior_cache")
            self.assertEqual(result["attempts"], [])
            # Source corruption is recorded, not repaired in another task's cache.
            raw = next(p for p in source.glob("*.json") if not p.name.endswith(".metadata.json"))
            raw.write_bytes(b"{}")
            result = c.fetch_with_retries(m.SCHEDULE_URL, Path(temp)/"new", reuse_dirs=(source,), opener=fake_opener)
            self.assertEqual(len(result["prior_cache_failures"]), 1)
            self.assertEqual(raw.read_bytes(), b"{}")
            self.assertIsNone(result["failure"])
            next(source.glob("*.metadata.json")).unlink()
            _, _, failures = c.reuse_cache(m.SCHEDULE_URL, Path(temp)/"third", (source,))
            self.assertIn("incomplete_prior_cache", failures[0]["error"])

    def test_retry_cap_permanent_failure_and_failed_attempt_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            attempts, waits = [], []
            def fail(url, **kwargs):
                attempts.append(url)
                raise URLError("synthetic connection failure")
            result = c.fetch_with_retries(m.SCHEDULE_URL, temp, opener=fail, sleeper=waits.append)
            self.assertEqual((len(attempts), waits), (3, [1, 2]))
            self.assertEqual(len(result["attempts"]), 3)
            self.assertTrue(all(x["attempted_at_utc"].endswith("+00:00") for x in result["attempts"]))
            self.assertIsNotNone(result["failure"])
            def denied(url, **kwargs):
                raise HTTPError(url, 403, "synthetic denied", {}, None)
            result = c.fetch_with_retries(m.SCHEDULE_URL, temp, opener=denied, sleeper=self.fail)
            self.assertEqual(len(result["attempts"]), 1)
            self.assertEqual(result["attempts"][0]["status"], 403)

    def test_transient_failure_then_success_preserves_both_attempts(self):
        calls = []
        def opener(url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                raise URLError("transient fixture")
            return fake_opener(url)
        with tempfile.TemporaryDirectory() as temp:
            result = c.fetch_with_retries(m.SCHEDULE_URL, temp, opener=opener, sleeper=lambda _: None)
            self.assertIsNone(result["failure"])
            self.assertEqual([x["success"] for x in result["attempts"]], [False, True])

    def test_all_final_regular_games_deduplicate_only_by_pk_and_check_links(self):
        _, schedule, _ = fixture()
        first = schedule["dates"][0]["games"][0]
        last = dict(first, gamePk=20, officialDate="2025-06-14", link="/api/v1.1/game/20/feed/live")
        invalid = [dict(first, gamePk=30, gameType="E"), dict(first, gamePk=31, status=dict(abstractGameState="Live")),
                   dict(first, gamePk=32, officialDate="2025-06-15")]
        schedule["dates"][0]["games"] = [last, first]+invalid
        games, audit = c.select_games([("day1", schedule), ("day2", schedule)])
        self.assertEqual([g["gamePk"] for g in games], [10, 20])
        self.assertEqual(audit["duplicate_game_records"], 2)
        self.assertEqual(audit["excluded_schedule_records"], dict(not_regular=2, not_final=2, official_date_outside_interval_or_missing=2))
        self.assertEqual(games[0]["schedule_urls"], ["day1", "day2"])
        bad = copy.deepcopy(schedule)
        bad["dates"][0]["games"][1]["link"] = "/api/v1.1/game/999/feed/live"
        with self.assertRaisesRegex(ValueError, "actual_game"):
            c.select_games([("bad", bad)])
        conflict = copy.deepcopy(schedule)
        conflict["dates"][0]["games"][1]["officialDate"] = "2025-06-02"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            c.select_games([("day1", schedule), ("day2", conflict)])
        for payload in ({}, dict(dates=[None]), dict(dates=[dict(games=[None])])):
            with self.assertRaises(ValueError):
                c.select_games([("invalid", payload)])

    def test_collect_game_adds_schedule_date_preserves_parse_and_fetch_failures(self):
        _, schedule, codes = fixture()
        games, _ = c.select_games([("schedule", schedule)])
        lookup = m.pitch_code_lookup(codes)
        with tempfile.TemporaryDirectory() as temp:
            result = c.collect_game(games[0], lookup, temp, opener=fake_opener)
            self.assertEqual(len(result["rows"]), 2)
            self.assertTrue(all(r["gameDate"] == "2025-06-01" for r in result["rows"]))
            self.assertTrue(result["audit"]["boxscore_pitch_count_matches"])
        def wrong_date(url, **kwargs):
            feed, _, _ = fixture()
            feed["gameData"]["datetime"] = dict(officialDate="2025-06-02")
            return Response(json.dumps(feed).encode(), url=url)
        with tempfile.TemporaryDirectory() as temp:
            result = c.collect_game(games[0], lookup, temp, opener=wrong_date)
            self.assertEqual(result["failure"]["stage"], "parse_or_audit")
            self.assertIsNotNone(result["provenance"]["source"]["sha256"])
            self.assertEqual(result["rows"], [])
        with tempfile.TemporaryDirectory() as temp:
            result = c.collect_game(games[0], lookup, temp, opener=lambda *a, **k: (_ for _ in ()).throw(URLError("failure")),
                                    sleeper=lambda _: None)
            self.assertEqual(result["failure"]["stage"], "fetch")

    def test_aggregate_preserves_unknown_units_schema_issues_and_duplicates(self):
        feed, _, codes = fixture()
        rows, parsing = m.parse_feed(feed, m.pitch_code_lookup(codes), expected_game_pk=10)
        rows[0]["spinRate"] = None
        rows[0]["identity_status"] = "uncertain_quarantined"
        audit = m.audit_feed(feed, rows, parsing)
        aggregate = c.aggregate_audits([dict(audit=audit), dict(audit=None)], rows+[copy.deepcopy(rows[0])])
        self.assertEqual(aggregate["coverage"]["spinRate"]["missing"], 1)
        self.assertEqual(aggregate["identity_quarantined_pitch_count"], 1)
        self.assertEqual(aggregate["duplicate_pitch_keys"], 1)
        self.assertEqual(aggregate["duplicate_play_ids"], 1)
        self.assertEqual(aggregate["coverage"]["pfxZ"]["unit"], "UNKNOWN")
        self.assertEqual(aggregate["games_with_schema_issues"], [10])
        self.assertFalse(aggregate["relationships_computed"])

    def test_pipeline_workers_bounded_rows_ignored_artifact_and_aggregate_only(self):
        lock, active, maximum, calls = threading.Lock(), 0, 0, []
        def opener(url, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(active, maximum)
                calls.append(url)
            time.sleep(.003)
            value = fake_opener(url)
            with lock:
                active -= 1
            return value
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            progress = []
            audit, manifest, path = c.collect(cache_dir=root/"artifacts", results_dir=root/"results", reuse_dirs=(), opener=opener,
                                              progress=progress.append)
            self.assertEqual((audit["schedule_days_received"], audit["selected_game_count"], audit["pitch_count"]), (14, 1, 2))
            self.assertTrue(audit["collection_complete"])
            self.assertLessEqual(maximum, 3)
            self.assertEqual(len(calls), 16)
            self.assertEqual(len(json.loads(path.read_text())), 2)
            self.assertFalse(path.with_suffix(".json.pending").exists())
            self.assertEqual(len(progress), 2)
            self.assertEqual(manifest["pitch_level_export"]["rows"], 2)
            for result in (root/"results").glob("collection_*.json"):
                text = result.read_text(encoding="utf-8")
                self.assertNotIn('"reported_matchup"', text)
                self.assertNotIn('"playId"', text)
            before = len(calls)
            c.collect(cache_dir=root/"artifacts", results_dir=root/"results", reuse_dirs=(), opener=opener)
            self.assertEqual(len(calls), before)

    def test_failed_schedule_and_bad_lookup_preserve_completeness_failure(self):
        def one_bad_day(url, **kwargs):
            if "date=2025-06-02" in url:
                return Response(b'{"notDates": []}', url=url)
            return fake_opener(url)
        with tempfile.TemporaryDirectory() as temp:
            audit, _, _ = c.collect(cache_dir=Path(temp)/"a", results_dir=Path(temp)/"r", reuse_dirs=(), opener=one_bad_day)
            self.assertEqual(audit["schedule_days_received"], 13)
            self.assertEqual(audit["failures"][0]["stage"], "schedule_schema")
            self.assertEqual(audit["pitch_count"], 2)
            self.assertFalse(audit["collection_complete"])
        def bad_lookup(url, **kwargs):
            return Response(b"[]", url=url) if url == m.PITCH_CODES_URL else fake_opener(url)
        with tempfile.TemporaryDirectory() as temp:
            audit, _, _ = c.collect(cache_dir=Path(temp)/"a", results_dir=Path(temp)/"r", reuse_dirs=(), opener=bad_lookup)
            self.assertEqual(audit["pitch_count"], 0)
            self.assertEqual(audit["failures"][0]["stage"], "lookup_schema")
            self.assertFalse(audit["collection_complete"])

    def test_cli_does_not_compute_relationships(self):
        with patch.object(c, "collect", return_value=(dict(pitch_count=2, parsed_game_count=1, collection_complete=True), {}, Path("rows.json"))) as run:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                c.main(["--workers", "2"])
            self.assertEqual(run.call_args.kwargs["workers"], 2)
            self.assertEqual(json.loads(out.getvalue())["pitch_count"], 2)


if __name__ == "__main__":
    unittest.main()
