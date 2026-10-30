"""One preregistered MLB JSON access/schema audit; no outcome model or unit conversion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import urlopen


SOURCE_DATE = "2025-06-01"
SCHEDULE_URL = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={SOURCE_DATE}"
PITCH_CODES_URL = "https://statsapi.mlb.com/api/v1/pitchCodes"
METRICS_URL = "https://statsapi.mlb.com/api/v1/metrics"
INITIAL_FEED_SHA256 = "7f40e379d47c491aa1318190a030fe6ba5ff9f609dd9f42ebc2c852a692d0fc7"
ROOT = Path(__file__).resolve().parents[2]
NUMERIC_PATHS = {
    **{k: ("pitchData", k) for k in ("startSpeed", "endSpeed", "extension", "plateTime")},
    **{k: ("pitchData", "breaks", k) for k in (
        "spinRate", "spinDirection", "breakAngle", "breakLength", "breakY",
        "breakHorizontal", "breakVertical", "breakVerticalInduced")},
    **{k: ("pitchData", "coordinates", k) for k in (
        "pfxX", "pfxZ", "pX", "pZ", "x0", "y0", "z0", "vX0", "vY0", "vZ0",
        "aX", "aY", "aZ")},
}
LOOKUP_FLAGS = ("swingStatus", "swingMissStatus", "swingContactStatus", "contactStatus",
                "buntAttemptStatus", "pitchStatus")


def _official_url(url):
    if not isinstance(url, str):
        raise ValueError("official_url_must_be_string")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "statsapi.mlb.com"
            or not parsed.path.startswith("/api/") or parsed.fragment):
        raise ValueError("only_public_official_statsapi_urls_allowed")
    return url


def _json_bytes(raw):
    def reject_constant(value):
        raise ValueError(f"non_json_numeric_constant:{value}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non_finite_json_number")
        return number
    value = json.loads(raw.decode("utf-8"), parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(value, (dict, list)):
        raise ValueError("json_response_must_be_object_or_array")
    return value


def fetch_json(url, cache_dir, *, refresh=False, opener=None):
    """Cache exact response bytes and access provenance, never a fabricated response."""
    _official_url(url)
    cache_dir = Path(cache_dir)
    key = hashlib.sha256(url.encode()).hexdigest()
    raw_path = cache_dir / f"{key}.json"
    meta_path = cache_dir / f"{key}.metadata.json"
    if not refresh and raw_path.exists() and meta_path.exists():
        raw = raw_path.read_bytes()
        metadata = _json_bytes(meta_path.read_bytes())
        if (not isinstance(metadata, dict) or metadata.get("requested_url") != url
                or metadata.get("sha256") != hashlib.sha256(raw).hexdigest()
                or metadata.get("bytes") != len(raw) or metadata.get("status") != 200
                or "json" not in str(metadata.get("content_type", "")).lower()):
            raise ValueError("cache_provenance_mismatch")
        _official_url(metadata.get("response_url", ""))
        value = _json_bytes(raw)
        return value, dict(metadata, from_cache=True)
    if raw_path.exists() != meta_path.exists() and not refresh:
        raise ValueError("incomplete_cache")
    with (opener or urlopen)(url, timeout=40) as response:
        if response.status != 200:
            raise ValueError(f"unexpected_http_status:{response.status}")
        response_url = _official_url(response.geturl())
        content_type = response.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            raise ValueError("non_json_content_type")
        raw = response.read()
    value = _json_bytes(raw)
    metadata = dict(requested_url=url, response_url=response_url, status=200,
                    content_type=content_type, bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    accessed_at_utc=datetime.now(timezone.utc).isoformat(),
                    from_cache=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    return value, metadata


def select_game(schedule):
    """Use the response's actual link for the smallest completed regular gamePk."""
    if not isinstance(schedule, dict) or not isinstance(schedule.get("dates"), list):
        raise ValueError("invalid_schedule_schema")
    candidates = []
    for day in schedule["dates"]:
        if not isinstance(day, dict) or not isinstance(day.get("games"), list):
            raise ValueError("invalid_schedule_day_schema")
        for game in day["games"]:
            if not isinstance(game, dict):
                raise ValueError("invalid_schedule_game_schema")
            if (game.get("gameType") == "R"
                    and game.get("officialDate") == SOURCE_DATE
                    and game.get("status", {}).get("abstractGameState") == "Final"):
                pk = game.get("gamePk")
                if type(pk) is not int or pk <= 0:
                    raise ValueError("invalid_game_pk")
                candidates.append(game)
    if not candidates:
        raise ValueError("no_completed_regular_game_on_fixed_date")
    if len({g["gamePk"] for g in candidates}) != len(candidates):
        raise ValueError("duplicate_schedule_game_pk")
    game = min(candidates, key=lambda g: g["gamePk"])
    link = game.get("link", "")
    if not re.fullmatch(r"/api/v1(?:\.1)?/game/" + str(game["gamePk"]) + r"/feed/live", link):
        raise ValueError("missing_or_mismatched_actual_feed_link")
    return game, _official_url("https://statsapi.mlb.com" + link)


def pitch_code_lookup(payload):
    if not isinstance(payload, list) or not payload:
        raise ValueError("invalid_pitch_code_lookup")
    lookup = {}
    for item in payload:
        if (not isinstance(item, dict) or not isinstance(item.get("code"), str)
                or not isinstance(item.get("description"), str)):
            raise ValueError("invalid_pitch_code_entry")
        code = item["code"]
        if code in lookup or not code:
            raise ValueError("duplicate_or_empty_pitch_code")
        if any(type(item.get(flag)) is not bool for flag in LOOKUP_FLAGS):
            raise ValueError("invalid_official_lookup_boolean")
        lookup[code] = dict(item)
    return lookup


def _value(obj, path):
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _count(count):
    if (isinstance(count, dict) and type(count.get("balls")) is int
            and type(count.get("strikes")) is int
            and 0 <= count["balls"] <= 4 and 0 <= count["strikes"] <= 3):
        return {key: count[key] for key in ("balls", "strikes")}
    return None


def _identity_substitution(event):
    kind = _value(event, ("details", "eventType"))
    position = _value(event, ("position", "abbreviation"))
    if kind == "pitching_substitution" or (kind == "defensive_substitution" and position == "P"):
        return "pitcher"
    if kind == "offensive_substitution" and position != "PR":
        return "batter_or_undocumented_offensive_substitution"
    return None


def parse_feed(feed, lookup, *, expected_game_pk):
    """Return pitch rows and audit issues, leaving unknown values unfilled."""
    if (not isinstance(feed, dict) or type(feed.get("gamePk")) is not int
            or feed.get("gamePk") != expected_game_pk
            or _value(feed, ("gameData", "game", "type")) != "R"
            or _value(feed, ("gameData", "status", "abstractGameState")) != "Final"):
        raise ValueError("feed_game_identity_or_status_mismatch")
    plays = _value(feed, ("liveData", "plays", "allPlays"))
    if not isinstance(plays, list) or not plays:
        raise ValueError("missing_all_plays")
    rows, issues, substitutions = [], [], []
    for play in plays:
        if not isinstance(play, dict):
            raise ValueError("invalid_play_object")
        ab = _value(play, ("about", "atBatIndex"))
        events = play.get("playEvents")
        if type(ab) is not int or ab < 0 or not isinstance(events, list):
            raise ValueError("invalid_play_schema")
        if any(not isinstance(e, dict) for e in events):
            raise ValueError("invalid_event_object")
        indices = [e.get("index") for e in events]
        if any(type(i) is not int or i < 0 for i in indices) or indices != sorted(indices):
            raise ValueError("events_not_in_valid_index_order")
        if len(set(indices)) != len(indices):
            issues.append(dict(issue="duplicate_event_indices", atBatIndex=ab))
        if any(type(e.get("isPitch")) is not bool for e in events):
            issues.append(dict(issue="invalid_isPitch_type", atBatIndex=ab))
        first_pitch_position = next((i for i, e in enumerate(events) if e.get("isPitch") is True), None)
        mid_subs = []
        for position, event in enumerate(events):
            kind = _identity_substitution(event)
            if kind:
                after = first_pitch_position is not None and position > first_pitch_position
                item = dict(atBatIndex=ab, eventIndex=event["index"], kind=kind,
                            after_first_pitch=after)
                substitutions.append(item)
                if after:
                    mid_subs.append(item)
        reported_pitcher = _value(play, ("matchup", "pitcher", "id"))
        reported_batter = _value(play, ("matchup", "batter", "id"))
        side = _value(play, ("matchup", "batSide", "code"))
        hand = _value(play, ("matchup", "pitchHand", "code"))
        identity_ok = (not mid_subs and type(reported_pitcher) is int and reported_pitcher > 0
                       and type(reported_batter) is int and reported_batter > 0
                       and side in ("R", "L") and hand in ("R", "L"))
        state = dict(balls=0, strikes=0)
        for event in events:
            post = _count(event.get("count"))
            if post is None:
                issues.append(dict(issue="invalid_or_missing_event_count", atBatIndex=ab,
                                   eventIndex=event["index"]))
            if event.get("isPitch") is True:
                call_code = _value(event, ("details", "call", "code"))
                official = lookup.get(call_code) if isinstance(call_code, str) else None
                row = dict(gamePk=expected_game_pk, atBatIndex=ab, pitchIndex=event["index"],
                           playId=event.get("playId"), pitchNumber=event.get("pitchNumber"),
                           startTime=event.get("startTime"), endTime=event.get("endTime"),
                           pitcherId=reported_pitcher if identity_ok else None,
                           batterId=reported_batter if identity_ok else None,
                           batSide=side if identity_ok else None, pitchHand=hand if identity_ok else None,
                           reported_matchup=dict(pitcherId=reported_pitcher, batterId=reported_batter,
                                                 batSide=side, pitchHand=hand),
                           identity_status="verified_within_play" if identity_ok else "uncertain_quarantined",
                           identity_substitutions=mid_subs,
                           pitchType=_value(event, ("details", "type", "code")),
                           callCode=call_code, detailsCode=_value(event, ("details", "code")),
                           callDescription=_value(event, ("details", "call", "description")),
                           detailsDescription=_value(event, ("details", "description")),
                           pre_count=dict(state) if state is not None else None,
                           post_count=dict(post) if post is not None else None,
                           official_pitch_code=dict(official) if official else None,
                           **{key: _value(event, path) for key, path in NUMERIC_PATHS.items()})
                rows.append(row)
            # Non-pitch events, including timer violations, update the next pitch's pre-count.
            state = post
        if state != _count(play.get("count")) or state is None:
            issues.append(dict(issue="play_final_count_mismatch", atBatIndex=ab))
    if not rows:
        raise ValueError("no_pitch_events")
    return rows, dict(play_count=len(plays), issues=issues, substitutions=substitutions)


def audit_feed(feed, rows, parsing):
    """Aggregate schema/coverage only. No swing rates, associations or causal claims."""
    coverage = {}
    for key in NUMERIC_PATHS:
        values = [r[key] for r in rows]
        missing = sum(v is None for v in values)
        numeric = [v for v in values if type(v) in (int, float)]
        finite = [v for v in numeric if math.isfinite(v)]
        coverage[key] = dict(rows=len(rows), missing=missing, finite=len(finite),
                             non_finite=len(numeric)-len(finite),
                             invalid_type=len(values)-missing-len(numeric),
                             minimum=min(finite) if finite else None,
                             maximum=max(finite) if finite else None,
                             unit="UNKNOWN", coordinate_convention="UNKNOWN",
                             temporal_reference="UNKNOWN", source_path=".".join(NUMERIC_PATHS[key]))
    ids = [(r["gamePk"], r["atBatIndex"], r["pitchIndex"]) for r in rows]
    play_ids = [r["playId"] for r in rows if isinstance(r["playId"], str) and r["playId"]]
    duplicates = len(ids)-len(set(ids))
    duplicate_play_ids = len(play_ids)-len(set(play_ids))
    invalid_identifiers = sum(not isinstance(r["playId"], str) or not r["playId"]
                              or type(r["pitchNumber"]) is not int or r["pitchNumber"] < 1 for r in rows)
    unknown_calls = sum(r["official_pitch_code"] is None for r in rows)
    inconsistent_call_codes = sum(r["callCode"] != r["detailsCode"] for r in rows)
    invalid_pitch_type = sum(not isinstance(r["pitchType"], str) or not r["pitchType"] for r in rows)
    identity_quarantine = sum(r["identity_status"] != "verified_within_play" for r in rows)
    counts_missing = sum(r["pre_count"] is None or r["post_count"] is None for r in rows)
    invalid_pre_counts = sum(r["pre_count"] is not None and
                             (r["pre_count"]["balls"] > 3 or r["pre_count"]["strikes"] > 2) for r in rows)
    totals = {}
    for side in ("away", "home"):
        stats = _value(feed, ("liveData", "boxscore", "teams", side, "teamStats", "pitching"))
        totals[side] = {k: stats.get(k) if isinstance(stats, dict) else None
                        for k in ("pitchesThrown", "numberOfPitches")}
    totals_valid = all(type(totals[s][k]) is int and totals[s][k] >= 0
                       for s in totals for k in totals[s])
    boxscore_matches = (totals_valid and all(sum(totals[s][k] for s in totals) == len(rows)
                                             for k in ("pitchesThrown", "numberOfPitches")))
    finite_complete = all(c["finite"] == len(rows) for c in coverage.values())
    schema_valid = (finite_complete and not any((duplicates, duplicate_play_ids, invalid_identifiers,
                    unknown_calls, inconsistent_call_codes, invalid_pitch_type, identity_quarantine,
                    counts_missing, invalid_pre_counts, len(parsing["issues"]))))
    return dict(gamePk=feed["gamePk"], play_count=parsing["play_count"], pitch_count=len(rows),
                pitcher_count=len({r["pitcherId"] for r in rows if r["pitcherId"] is not None}),
                unique_pitch_type_codes=sorted({r["pitchType"] for r in rows if isinstance(r["pitchType"], str)}),
                coverage=coverage, duplicate_pitch_keys=duplicates, duplicate_play_ids=duplicate_play_ids,
                invalid_identifiers=invalid_identifiers, unknown_call_codes=unknown_calls,
                inconsistent_call_codes=inconsistent_call_codes, invalid_pitch_types=invalid_pitch_type,
                identity_quarantined_pitch_count=identity_quarantine,
                mid_play_identity_substitutions=[s for s in parsing["substitutions"] if s["after_first_pitch"]],
                before_first_pitch_identity_substitution_count=sum(not s["after_first_pitch"] for s in parsing["substitutions"]),
                missing_pitch_counts=counts_missing, invalid_pre_pitch_count_count=invalid_pre_counts,
                parsing_issues=parsing["issues"], boxscore_pitch_totals=totals,
                boxscore_pitch_count_matches=boxscore_matches, schema_valid=schema_valid,
                technical_access_ready=schema_valid and boxscore_matches,
                conditional_study_ready=False,
                conditional_study_blockers=["feed_field_units_coordinate_and_time_definitions_unconfirmed",
                                            "one_game_only", "association_analysis_not_preregistered_here"],
                body_to_spin_ready=False, spin_3d_ready=False, intervention_effect_verified=False,
                target_location_observed=False, grip_observed=False,
                unobserved_fields=dict(arm_angle=None, spin_3d_vector=None, spin_efficiency=None,
                                       seam_orientation=None, intended_target_location=None,
                                       grip=None, full_body_kinematics=None),
                outcome_model_fitted=False, research_whiff_definition=None,
                official_lookup_flags_preserved=list(LOOKUP_FLAGS))


def run(*, cache_dir=None, results_dir=None, refresh=False, opener=None):
    cache = Path(cache_dir) if cache_dir is not None else ROOT / "artifacts" / "mlb_feed_audit"
    results = Path(results_dir) if results_dir is not None else Path(__file__).resolve().parent / "results"
    schedule, schedule_source = fetch_json(SCHEDULE_URL, cache, refresh=refresh, opener=opener)
    game, feed_url = select_game(schedule)
    feed, feed_source = fetch_json(feed_url, cache, refresh=refresh, opener=opener)
    codes, codes_source = fetch_json(PITCH_CODES_URL, cache, refresh=refresh, opener=opener)
    metrics, metrics_source = fetch_json(METRICS_URL, cache, refresh=refresh, opener=opener)
    lookup = pitch_code_lookup(codes)
    rows, parsing = parse_feed(feed, lookup, expected_game_pk=game["gamePk"])
    audit = audit_feed(feed, rows, parsing)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "parsed_pitches.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    manifest = dict(protocol="coach_hypotheses/pitch_quality/DATA_PROTOCOL.md",
                    source_format="official_mlb_json_not_statcast_or_mendeley_csv",
                    fixed_date=SOURCE_DATE, game_selection="minimum_numeric_gamePk_completed_R",
                    gamePk=game["gamePk"], schedule_actual_feed_link=game["link"],
                    sources=dict(schedule=schedule_source, feed=feed_source,
                                 pitch_codes=codes_source, metrics=metrics_source),
                    initial_feed_sha256=INITIAL_FEED_SHA256,
                    current_feed_matches_initial_sha256=feed_source["sha256"] == INITIAL_FEED_SHA256,
                    initial_audit=dict(gamePk=777677, play_count=69, pitch_count=256, pitcher_count=8),
                    current_sample_matches_initial_audit=all(audit[k] == v for k, v in
                        dict(gamePk=777677, play_count=69, pitch_count=256, pitcher_count=8).items()),
                    pitch_code_lookup_entries=len(lookup),
                    metrics_registry_entry_count=len(metrics) if isinstance(metrics, list) else None,
                    metrics_aliases_to_feed_fields_confirmed=False,
                    raw_and_pitch_level_exports="gitignored_artifacts_only",
                    prior_csv_access_failures_preserved=[
                        dict(source="Mendeley xfnz6mkdzm v2 actual files", status="HTTP_403_NOT_ACQUIRED"),
                        dict(source="Baseball Savant search/CSV", status="HTTP_403_CSV_NOT_ACQUIRED")],
                    license="public_access_does_not_establish_redistribution_permission")
    verification = dict(technical_access_ready=audit["technical_access_ready"],
                        conditional_study_ready=False, body_to_spin_ready=False,
                        outcome_analysis_performed=False, fixtures_count_as_research_data=False,
                        record_count=len(rows), source_sha256=feed_source["sha256"],
                        all_numeric_fields_finite_complete=all(c["finite"] == len(rows) for c in audit["coverage"].values()),
                        official_pitch_flags_complete=all(r["official_pitch_code"] is not None for r in rows),
                        boxscore_pitch_count_matches=audit["boxscore_pitch_count_matches"])
    results.mkdir(parents=True, exist_ok=True)
    for name, payload in (("mlb_manifest.json", manifest), ("mlb_audit.json", audit),
                          ("mlb_verification.json", verification)):
        (results / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return audit, manifest, verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refetch the fixed single-game sample")
    args = parser.parse_args(argv)
    audit, _, _ = run(refresh=args.refresh)
    print(json.dumps({k: audit[k] for k in ("gamePk", "pitch_count", "technical_access_ready",
                                          "conditional_study_ready", "body_to_spin_ready")}))


if __name__ == "__main__":
    main()
