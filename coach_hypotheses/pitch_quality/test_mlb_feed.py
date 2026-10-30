"""Parser edge cases use synthetic fixtures; they are never research observations."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import mlb_feed as m


def code_fixture(code="B"):
    return dict(code=code, description="fixture only", pitchResultText="fixture only",
                **{key: False for key in m.LOOKUP_FLAGS})


def fixture():
    # The automatic ball is a non-pitch event between two actual pitches.
    events = [dict(index=0, isPitch=True, playId="p1", pitchNumber=1,
                   count=dict(balls=1, strikes=0, outs=0),
                   details=dict(code="B", call=dict(code="B", description="Ball"), type=dict(code="FF"))),
              dict(index=1, isPitch=False, count=dict(balls=2, strikes=0, outs=0),
                   details=dict(code="V", eventType="pitch_timer_violation")),
              dict(index=2, isPitch=True, playId="p2", pitchNumber=2,
                   count=dict(balls=3, strikes=0, outs=0),
                   details=dict(code="B", call=dict(code="B", description="Ball"), type=dict(code="FF")))]
    for event in (events[0], events[2]):
        event["pitchData"] = dict(startSpeed=92.0, endSpeed=84.0, extension=6.0, plateTime=0.4,
                                 breaks={key: 1.0 for key, path in m.NUMERIC_PATHS.items() if path[1] == "breaks"},
                                 coordinates={key: 1.0 for key, path in m.NUMERIC_PATHS.items() if path[1] == "coordinates"})
    play = dict(about=dict(atBatIndex=0), count=dict(balls=3, strikes=0, outs=0),
                matchup=dict(pitcher=dict(id=1), batter=dict(id=2),
                             batSide=dict(code="R"), pitchHand=dict(code="L")), playEvents=events)
    stats = {side: dict(teamStats=dict(pitching=dict(pitchesThrown=1, numberOfPitches=1)))
             for side in ("away", "home")}
    feed = dict(gamePk=10, gameData=dict(game=dict(type="R"), status=dict(abstractGameState="Final")),
                liveData=dict(plays=dict(allPlays=[play]), boxscore=dict(teams=stats)))
    game = dict(gamePk=10, gameType="R", officialDate=m.SOURCE_DATE,
                status=dict(abstractGameState="Final"), link="/api/v1.1/game/10/feed/live")
    return feed, dict(dates=[dict(games=[game])]), [code_fixture()]


class Response:
    def __init__(self, raw, url=m.SCHEDULE_URL, status=200, content_type="application/json;charset=UTF-8"):
        self.raw, self.url, self.status = raw, url, status
        self.headers = {"Content-Type": content_type}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return self.raw
    def geturl(self):
        return self.url


class FetchTests(unittest.TestCase):
    def test_exact_byte_cache_provenance_and_refresh(self):
        raw = b'{"dates": []}\n'
        with tempfile.TemporaryDirectory() as temp:
            opener = unittest.mock.Mock(return_value=Response(raw))
            value, source = m.fetch_json(m.SCHEDULE_URL, temp, opener=opener)
            self.assertEqual(value, {"dates": []})
            self.assertEqual(source["bytes"], len(raw))
            self.assertFalse(source["from_cache"])
            self.assertIn("+00:00", source["accessed_at_utc"])
            cached, metadata = m.fetch_json(m.SCHEDULE_URL, temp, opener=opener)
            self.assertEqual(cached, value)
            self.assertTrue(metadata["from_cache"])
            opener.assert_called_once()
            m.fetch_json(m.SCHEDULE_URL, temp, refresh=True, opener=opener)
            self.assertEqual(opener.call_count, 2)
            raw_file = next(p for p in Path(temp).glob("*.json") if not p.name.endswith(".metadata.json"))
            raw_file.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "cache_provenance"):
                m.fetch_json(m.SCHEDULE_URL, temp, opener=opener)

    def test_invalid_json_http_origin_and_incomplete_cache(self):
        for url in ("http://statsapi.mlb.com/api/v1/schedule", "https://evil.example/api/x",
                    "https://statsapi.mlb.com:443/api/v1/schedule", "https://statsapi.mlb.com/api/x#fragment"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                m._official_url(url)
        for raw in (b"<html>denied</html>", b"NaN", b'{"x": NaN}', b'{"x": 1e999}', b"12"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                m._json_bytes(raw)
        for response in (Response(b"{}", status=403), Response(b"{}", content_type="text/html"),
                         Response(b"{}", url="https://other.example/api/x")):
            with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
                m.fetch_json(m.SCHEDULE_URL, temp, opener=lambda *a, **k: response)
        with tempfile.TemporaryDirectory() as temp:
            m.fetch_json(m.SCHEDULE_URL, temp, opener=lambda *a, **k: Response(b"{}"))
            next(Path(temp).glob("*.metadata.json")).unlink()
            with self.assertRaisesRegex(ValueError, "incomplete_cache"):
                m.fetch_json(m.SCHEDULE_URL, temp)


class SelectionAndLookupTests(unittest.TestCase):
    def test_minimum_final_regular_game_and_actual_link(self):
        _, schedule, _ = fixture()
        original = schedule["dates"][0]["games"][0]
        larger = dict(original, gamePk=20, link="/api/v1.1/game/20/feed/live")
        unfinished = dict(original, gamePk=1, status=dict(abstractGameState="Live"))
        exhibition = dict(original, gamePk=2, gameType="E")
        other_date = dict(original, gamePk=3, officialDate="2025-06-02")
        schedule["dates"][0]["games"] = [larger, unfinished, exhibition, other_date, original]
        selected, url = m.select_game(schedule)
        self.assertEqual(selected["gamePk"], 10)
        self.assertEqual(url, "https://statsapi.mlb.com/api/v1.1/game/10/feed/live")
        for bad in ({}, dict(dates=[]), dict(dates=[dict(games=[dict(original, link="/api/v1.1/game/11/feed/live")])]),
                    dict(dates=[dict(games=[original, original])]),
                    dict(dates=[dict(games=[dict(original, gamePk=True)])])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.select_game(bad)

    def test_official_flags_preserved_including_foul_tip(self):
        tip = code_fixture("T")
        tip.update(swingStatus=True, swingMissStatus=True)
        lookup = m.pitch_code_lookup([tip])
        self.assertTrue(lookup["T"]["swingMissStatus"])
        self.assertNotIn("whiff", lookup["T"])
        for bad in ([], {}, [dict(tip, swingStatus=1)], [tip, tip], [dict(code="T")], [dict(tip, code="")]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.pitch_code_lookup(bad)


class ParserTests(unittest.TestCase):
    def parse(self, feed=None):
        original, _, codes = fixture()
        return m.parse_feed(feed or original, m.pitch_code_lookup(codes), expected_game_pk=10)

    def test_all_nonpitch_counts_update_pre_count_and_raw_values_survive(self):
        feed, _, _ = fixture()
        rows, info = self.parse(feed)
        self.assertEqual(rows[0]["pre_count"], dict(balls=0, strikes=0))
        self.assertEqual(rows[1]["pre_count"], dict(balls=2, strikes=0))
        self.assertEqual(rows[1]["post_count"], dict(balls=3, strikes=0))
        self.assertEqual(rows[0]["startSpeed"], 92.0)
        self.assertEqual(info["issues"], [])
        self.assertEqual(rows[0]["pitcherId"], 1)
        self.assertEqual(rows[0]["official_pitch_code"], code_fixture())
        self.assertIsNone(m._value({}, ("pitchData", "breaks", "spinRate")))
        for invalid in ({"balls": True, "strikes": 0}, {"balls": 5, "strikes": 0}, None):
            self.assertIsNone(m._count(invalid))

    def test_mid_play_identity_changes_quarantine_whole_ab_but_pr_does_not(self):
        for kind, position, quarantine in (("pitching_substitution", "P", True),
                                           ("offensive_substitution", "PH", True),
                                           ("offensive_substitution", None, True),
                                           ("defensive_substitution", "P", True),
                                           ("offensive_substitution", "PR", False)):
            feed, _, _ = fixture()
            event = feed["liveData"]["plays"]["allPlays"][0]["playEvents"][1]
            event.update(details=dict(eventType=kind), position=dict(abbreviation=position))
            rows, info = self.parse(feed)
            self.assertEqual(rows[0]["pitcherId"] is None, quarantine)
            self.assertEqual(rows[1]["batterId"] is None, quarantine)
            self.assertEqual(rows[0]["reported_matchup"]["pitcherId"], 1)
            self.assertEqual(bool(rows[0]["identity_substitutions"]), quarantine)
            if quarantine:
                self.assertTrue(info["substitutions"][0]["after_first_pitch"])

    def test_before_first_pitch_substitution_is_not_mid_ab(self):
        feed, _, _ = fixture()
        events = feed["liveData"]["plays"]["allPlays"][0]["playEvents"]
        for event in events:
            event["index"] += 1
        events.insert(0, dict(index=0, isPitch=False, details=dict(eventType="pitching_substitution"),
                              count=dict(balls=0, strikes=0)))
        rows, info = self.parse(feed)
        self.assertEqual(rows[0]["pitcherId"], 1)
        self.assertFalse(info["substitutions"][0]["after_first_pitch"])

    def test_missing_invalid_count_does_not_backfill_future_count(self):
        feed, _, _ = fixture()
        events = feed["liveData"]["plays"]["allPlays"][0]["playEvents"]
        events[1]["count"] = dict(balls=9, strikes=0)
        rows, info = self.parse(feed)
        self.assertIsNone(rows[1]["pre_count"])
        self.assertEqual(rows[1]["post_count"], dict(balls=3, strikes=0))
        self.assertEqual(info["issues"][0]["issue"], "invalid_or_missing_event_count")
        feed["liveData"]["plays"]["allPlays"][0]["count"]["balls"] = 4
        _, info = self.parse(feed)
        self.assertIn("play_final_count_mismatch", [x["issue"] for x in info["issues"]])

    def test_automatic_count_before_first_pitch_and_ab_reset(self):
        feed, _, _ = fixture()
        plays = feed["liveData"]["plays"]["allPlays"]
        first = plays[0]
        for event in first["playEvents"]:
            event["index"] += 1
            event["count"]["balls"] += 1
        first["count"]["balls"] = 4
        first["playEvents"].insert(0, dict(index=0, isPitch=False, details=dict(code="V"),
                                          count=dict(balls=1, strikes=0)))
        second = copy.deepcopy(first)
        second["about"]["atBatIndex"] = 1
        second["playEvents"] = [dict(second["playEvents"][1], index=0, playId="p3", pitchNumber=1)]
        second["playEvents"][0]["count"] = dict(balls=1, strikes=0)
        second["count"] = dict(balls=1, strikes=0)
        plays.append(second)
        rows, info = self.parse(feed)
        self.assertEqual(rows[0]["pre_count"], dict(balls=1, strikes=0))
        self.assertEqual(rows[2]["pre_count"], dict(balls=0, strikes=0))
        self.assertEqual(info["issues"], [])

    def test_automatic_strike_nonpitch_is_in_next_precount(self):
        feed, _, _ = fixture()
        play = feed["liveData"]["plays"]["allPlays"][0]
        play["playEvents"][1]["details"]["code"] = "AC"
        play["playEvents"][1]["count"] = dict(balls=1, strikes=1)
        play["playEvents"][2]["count"] = dict(balls=2, strikes=1)
        play["count"] = dict(balls=2, strikes=1)
        rows, info = self.parse(feed)
        self.assertEqual(rows[1]["pre_count"], dict(balls=1, strikes=1))
        self.assertEqual(info["issues"], [])

    def test_unhashable_bad_identifiers_are_reported_not_filled(self):
        feed, _, _ = fixture()
        event = feed["liveData"]["plays"]["allPlays"][0]["playEvents"][0]
        event["playId"] = {"invalid": "identifier"}
        event["details"]["call"]["code"] = ["not", "a", "code"]
        rows, info = self.parse(feed)
        audit = m.audit_feed(feed, rows, info)
        self.assertEqual(audit["invalid_identifiers"], 1)
        self.assertEqual(audit["unknown_call_codes"], 1)
        self.assertFalse(audit["technical_access_ready"])

    def test_feed_structure_and_event_order_rejected(self):
        for mutation in ("game", "status", "order", "no_events", "missing_plays", "bad_ab"):
            feed, _, _ = fixture()
            play = feed["liveData"]["plays"]["allPlays"][0]
            if mutation == "game": feed["gamePk"] = 20
            if mutation == "status": feed["gameData"]["status"]["abstractGameState"] = "Live"
            if mutation == "order": play["playEvents"].reverse()
            if mutation == "no_events": play["playEvents"] = []
            if mutation == "missing_plays": feed["liveData"]["plays"]["allPlays"] = None
            if mutation == "bad_ab": play["about"]["atBatIndex"] = -1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.parse(feed)

    def test_audit_numbers_missing_types_nonfinite_duplicates_and_unknown_units(self):
        feed, _, _ = fixture()
        rows, info = self.parse(feed)
        audit = m.audit_feed(feed, rows, info)
        self.assertTrue(audit["technical_access_ready"])
        self.assertFalse(audit["conditional_study_ready"])
        self.assertFalse(audit["body_to_spin_ready"])
        self.assertTrue(all(x["unit"] == "UNKNOWN" for x in audit["coverage"].values()))
        for key, value, issue in (("spinRate", None, "missing"), ("pfxX", "1", "invalid_type"),
                                   ("pX", float("inf"), "non_finite"), ("startSpeed", True, "invalid_type")):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            result = m.audit_feed(feed, changed, info)
            self.assertEqual(result["coverage"][key][issue], 1)
            self.assertFalse(result["schema_valid"])
        changed = copy.deepcopy(rows)
        changed.append(copy.deepcopy(changed[0]))
        audit = m.audit_feed(feed, changed, info)
        self.assertEqual(audit["duplicate_pitch_keys"], 1)
        self.assertEqual(audit["duplicate_play_ids"], 1)
        self.assertFalse(audit["boxscore_pitch_count_matches"])

    def test_unsafe_identity_unknown_code_and_invalid_identifiers_fail_gate(self):
        feed, _, _ = fixture()
        play = feed["liveData"]["plays"]["allPlays"][0]
        play["matchup"]["pitcher"]["id"] = None
        play["playEvents"][0]["details"]["call"]["code"] = "UNKNOWN"
        play["playEvents"][0]["playId"] = None
        play["playEvents"][0]["pitchNumber"] = False
        play["playEvents"][0]["details"]["type"] = {}
        rows, info = self.parse(feed)
        audit = m.audit_feed(feed, rows, info)
        self.assertEqual(audit["identity_quarantined_pitch_count"], 2)
        self.assertEqual(audit["unknown_call_codes"], 1)
        self.assertEqual(audit["invalid_identifiers"], 1)
        self.assertEqual(audit["invalid_pitch_types"], 1)
        self.assertFalse(audit["technical_access_ready"])

    def test_duplicate_nonpitch_indices_and_wrong_boolean_are_audited(self):
        feed, _, _ = fixture()
        events = feed["liveData"]["plays"]["allPlays"][0]["playEvents"]
        events[1]["index"] = 0
        events[1]["isPitch"] = "false"
        rows, info = self.parse(feed)
        self.assertEqual({x["issue"] for x in info["issues"]}, {"duplicate_event_indices", "invalid_isPitch_type"})
        self.assertFalse(m.audit_feed(feed, rows, info)["technical_access_ready"])

    def test_boxscore_absence_is_not_zero_filled(self):
        feed, _, _ = fixture()
        feed["liveData"]["boxscore"] = {}
        rows, info = self.parse(feed)
        audit = m.audit_feed(feed, rows, info)
        self.assertIsNone(audit["boxscore_pitch_totals"]["away"]["pitchesThrown"])
        self.assertFalse(audit["technical_access_ready"])


class IntegrationTests(unittest.TestCase):
    def test_bounded_mock_pipeline_caches_rows_but_results_are_aggregate(self):
        feed, schedule, codes = fixture()
        urls = {m.SCHEDULE_URL: schedule, "https://statsapi.mlb.com/api/v1.1/game/10/feed/live": feed,
                m.PITCH_CODES_URL: codes, m.METRICS_URL: [dict(name="releaseSpeed", unit="MPH")]}
        called = []
        def opener(url, **kwargs):
            called.append(url)
            return Response(json.dumps(urls[url]).encode(), url=url)
        with tempfile.TemporaryDirectory() as temp:
            cache, results = Path(temp) / "artifacts", Path(temp) / "results"
            audit, manifest, verification = m.run(cache_dir=cache, results_dir=results, opener=opener)
            self.assertEqual(len(called), 4)
            self.assertEqual(audit["pitch_count"], 2)
            self.assertTrue(verification["technical_access_ready"])
            self.assertFalse(manifest["current_sample_matches_initial_audit"])
            self.assertFalse(manifest["metrics_aliases_to_feed_fields_confirmed"])
            self.assertEqual(len(json.loads((cache / "parsed_pitches.json").read_text())), 2)
            for path in results.glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertNotIn("records", payload)
            self.assertEqual(len(list(results.glob("mlb_*.json"))), 3)
            m.run(cache_dir=cache, results_dir=results, opener=opener)
            self.assertEqual(len(called), 4)

    def test_cli_summary_has_no_model_claim(self):
        result = dict(gamePk=10, pitch_count=2, technical_access_ready=True,
                      conditional_study_ready=False, body_to_spin_ready=False)
        with patch.object(m, "run", return_value=(result, {}, {})) as mocked, contextlib.redirect_stdout(io.StringIO()) as output:
            m.main(["--refresh"])
        mocked.assert_called_once_with(refresh=True)
        self.assertEqual(json.loads(output.getvalue()), result)


if __name__ == "__main__":
    unittest.main()
