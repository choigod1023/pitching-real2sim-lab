"""Collect the fixed June 1--14 MLB JSON sample; perform data audits only."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError

from . import mlb_feed as m

START_DATE = date(2025, 6, 1)
END_DATE = date(2025, 6, 14)
DEFAULT_CACHE = m.ROOT / "artifacts" / "mlb_quality_games"
DEFAULT_REUSE = m.ROOT.parent / "curve-timing-and-quality-data-20261030" / "artifacts" / "mlb_feed_audit"
SPEED_JS_URL = "https://builds.mlbstatic.com/baseballsavant.mlb.com/v1/sections/gamefeed/builds/4a2930fb3794eed37b9aabce1f4ff3d0966f246b/scripts/build/gamefeed-all.js"


def fixed_dates():
    return [(START_DATE + timedelta(days=i)).isoformat() for i in range((END_DATE-START_DATE).days+1)]


def reuse_cache(url, destination, sources):
    """Copy only a prior cache that passes the original byte/hash/URL validation."""
    key = hashlib.sha256(url.encode()).hexdigest()
    destination = Path(destination)
    failures = []
    for source in map(Path, sources):
        raw_path = source / f"{key}.json"
        meta_path = source / f"{key}.metadata.json"
        if not raw_path.exists() and not meta_path.exists():
            continue
        try:
            if not raw_path.exists() or not meta_path.exists():
                raise ValueError("incomplete_prior_cache")
            value, metadata = m.fetch_json(url, source, opener=lambda *a, **k: (_ for _ in ()).throw(ValueError("unexpected_network_in_cache_validation")))
            raw = raw_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != metadata["sha256"] or len(raw) != metadata["bytes"]:
                raise ValueError("prior_cache_changed_during_validation")
            destination.mkdir(parents=True, exist_ok=True)
            (destination / raw_path.name).write_bytes(raw)
            (destination / meta_path.name).write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
            return value, dict(metadata, cache_origin="validated_prior_cache"), failures
        except (ValueError, OSError) as exc:
            failures.append(dict(source_directory=source.name, error_type=type(exc).__name__, error=str(exc)))
    return None, None, failures


def fetch_with_retries(url, cache, *, reuse_dirs=(), retries=2, opener=None, sleeper=time.sleep):
    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError("at_most_two_additional_retries")
    m._official_url(url)
    key = hashlib.sha256(url.encode()).hexdigest()
    existing = (Path(cache) / f"{key}.json").exists() or (Path(cache) / f"{key}.metadata.json").exists()
    prior_failures = []
    if not existing:
        value, metadata, prior_failures = reuse_cache(url, cache, reuse_dirs)
        if metadata is not None:
            return dict(value=value, source=metadata, attempts=[], prior_cache_failures=prior_failures, failure=None)
    attempts = []
    for index in range(retries+1):
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            value, metadata = m.fetch_json(url, cache, opener=opener)
            attempts.append(dict(attempt=index+1, attempted_at_utc=stamp, success=True,
                                 from_cache=metadata["from_cache"], status=metadata["status"]))
            return dict(value=value, source=dict(metadata, cache_origin="current_cache" if metadata["from_cache"] else "network"),
                        attempts=attempts, prior_cache_failures=prior_failures, failure=None)
        except Exception as exc:
            status = exc.code if isinstance(exc, HTTPError) else None
            attempts.append(dict(attempt=index+1, attempted_at_utc=stamp, success=False,
                                 error_type=type(exc).__name__, error=str(exc), status=status))
            retryable = isinstance(exc, (OSError, URLError, TimeoutError))
            if isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code != 429:
                retryable = False
            if not retryable or index == retries:
                break
            sleeper(index+1)
    return dict(value=None, source=None, attempts=attempts, prior_cache_failures=prior_failures,
                failure=dict(url=url, stage="fetch", error=attempts[-1]["error"], error_type=attempts[-1]["error_type"]))


def select_games(schedules):
    """Deduplicate every completed R game in the fixed officialDate interval."""
    games, duplicates, excluded = {}, 0, Counter()
    for schedule_url, payload in schedules:
        if not isinstance(payload, dict) or not isinstance(payload.get("dates"), list):
            raise ValueError("invalid_daily_schedule")
        for day in payload["dates"]:
            if not isinstance(day, dict) or not isinstance(day.get("games"), list):
                raise ValueError("invalid_schedule_day")
            for game in day["games"]:
                if not isinstance(game, dict):
                    raise ValueError("invalid_schedule_game")
                official = game.get("officialDate")
                if game.get("gameType") != "R":
                    excluded["not_regular"] += 1
                    continue
                if game.get("status", {}).get("abstractGameState") != "Final":
                    excluded["not_final"] += 1
                    continue
                if not isinstance(official, str) or not START_DATE.isoformat() <= official <= END_DATE.isoformat():
                    excluded["official_date_outside_interval_or_missing"] += 1
                    continue
                # Reject malformed date strings that happened to pass lexical bounds.
                if date.fromisoformat(official).isoformat() != official:
                    raise ValueError("invalid_official_date")
                pk, link = game.get("gamePk"), game.get("link")
                if type(pk) is not int or pk <= 0 or not isinstance(link, str) or not re.fullmatch(
                        r"/api/v1(?:\.1)?/game/" + str(pk) + r"/feed/live", link):
                    raise ValueError("invalid_actual_game_pk_or_feed_link")
                record = dict(gamePk=pk, gameDate=official, feed_url=m._official_url("https://statsapi.mlb.com"+link),
                              actual_schedule_link=link, schedule_urls=[schedule_url])
                if pk in games:
                    old = games[pk]
                    if (old["gameDate"], old["feed_url"]) != (record["gameDate"], record["feed_url"]):
                        raise ValueError("conflicting_schedule_records_for_game")
                    old["schedule_urls"] = sorted(set(old["schedule_urls"]+[schedule_url]))
                    duplicates += 1
                else:
                    games[pk] = record
    return [games[pk] for pk in sorted(games)], dict(duplicate_game_records=duplicates, excluded_schedule_records=dict(excluded))


def collect_game(game, lookup, cache, *, reuse_dirs=(), retries=2, opener=None, sleeper=time.sleep):
    response = fetch_with_retries(game["feed_url"], cache, reuse_dirs=reuse_dirs, retries=retries,
                                 opener=opener, sleeper=sleeper)
    provenance = {key: value for key, value in response.items() if key != "value"}
    result = dict(game=game, provenance=provenance, audit=None, rows=[], failure=response["failure"])
    if result["failure"]:
        return result
    feed = response["value"]
    try:
        feed_date = m._value(feed, ("gameData", "datetime", "officialDate"))
        if feed_date is not None and feed_date != game["gameDate"]:
            raise ValueError("feed_and_schedule_official_date_mismatch")
        rows, parsing = m.parse_feed(feed, lookup, expected_game_pk=game["gamePk"])
        audit = m.audit_feed(feed, rows, parsing)
        # Preserve all parsed rows. Conditional complete-case exclusion belongs to the analysis.
        for row in rows:
            row["gameDate"] = game["gameDate"]
            row["game_boxscore_pitch_count_matches"] = audit["boxscore_pitch_count_matches"]
        result.update(audit=audit, rows=rows)
    except Exception as exc:
        result["failure"] = dict(url=game["feed_url"], gamePk=game["gamePk"], stage="parse_or_audit",
                                 error_type=type(exc).__name__, error=str(exc))
    return result


def aggregate_audits(game_results, rows):
    audits = [g["audit"] for g in game_results if g["audit"] is not None]
    coverage = {}
    for field in m.NUMERIC_PATHS:
        parts = [a["coverage"][field] for a in audits]
        minima = [p["minimum"] for p in parts if p["minimum"] is not None]
        maxima = [p["maximum"] for p in parts if p["maximum"] is not None]
        coverage[field] = {key: sum(p[key] for p in parts) for key in ("rows", "missing", "finite", "non_finite", "invalid_type")}
        coverage[field].update(minimum=min(minima) if minima else None, maximum=max(maxima) if maxima else None,
                               unit="UNKNOWN", coordinate_convention="UNKNOWN", temporal_reference="UNKNOWN")
    ids = [(r["gamePk"], r["atBatIndex"], r["pitchIndex"]) for r in rows]
    play_ids = [r["playId"] for r in rows if isinstance(r["playId"], str) and r["playId"]]
    totals = {}
    for name in ("unknown_call_codes", "identity_quarantined_pitch_count", "missing_pitch_counts",
                 "invalid_pre_pitch_count_count", "invalid_identifiers", "inconsistent_call_codes", "invalid_pitch_types"):
        totals[name] = sum(a[name] for a in audits)
    return dict(selected_game_count=len(game_results), parsed_game_count=len(audits), pitch_count=len(rows),
                play_count=sum(a["play_count"] for a in audits),
                pitcher_count=len({r["pitcherId"] for r in rows if r["pitcherId"] is not None}),
                games_with_boxscore_mismatch=[a["gamePk"] for a in audits if not a["boxscore_pitch_count_matches"]],
                games_with_schema_issues=[a["gamePk"] for a in audits if not a["schema_valid"]],
                parsing_issue_count=sum(len(a["parsing_issues"]) for a in audits),
                duplicate_pitch_keys=len(ids)-len(set(ids)), duplicate_play_ids=len(play_ids)-len(set(play_ids)),
                coverage=coverage, **totals, prior_conditional_study_ready=False, prior_gate_modified=False,
                body_to_spin_ready=False, relationships_computed=False,
                numeric_values_zero_filled=False, field_unit_conversion_performed=False,
                startSpeed_display_unit=dict(unit="mph", evidence=SPEED_JS_URL,
                                             evidence_scope="display_only_from_preregistered_protocol",
                                             physical_evaluation_location_and_time="UNKNOWN"))


def collect(*, cache_dir=DEFAULT_CACHE, results_dir=None, reuse_dirs=(DEFAULT_REUSE,), workers=3,
            retries=2, opener=None, sleeper=time.sleep, progress=None):
    if type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError("one_to_three_workers_required")
    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError("at_most_two_additional_retries")
    cache, results = Path(cache_dir), Path(results_dir or Path(__file__).resolve().parent / "results")
    cache.mkdir(parents=True, exist_ok=True)
    source_records, failures, schedules = [], [], []
    urls = [f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day}" for day in fixed_dates()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fetch_with_retries, url, cache, reuse_dirs=reuse_dirs, retries=retries,
                            opener=opener, sleeper=sleeper): url for url in urls}
        for job in as_completed(jobs):
            url, response = jobs[job], job.result()
            source_records.append(dict(url=url, **{k: v for k, v in response.items() if k != "value"}))
            if response["failure"]:
                failures.append(response["failure"])
            else:
                try:
                    select_games([(url, response["value"])])
                    schedules.append((url, response["value"]))
                except Exception as exc:
                    failures.append(dict(stage="schedule_schema", url=url, error_type=type(exc).__name__, error=str(exc)))
    # Keep successful days if any other schedule failed; completeness remains explicitly false.
    try:
        games, selection = select_games(schedules)
    except Exception as exc:
        failures.append(dict(stage="schedule_selection", error_type=type(exc).__name__, error=str(exc)))
        games, selection = [], dict(duplicate_game_records=None, excluded_schedule_records={})
    if progress:
        progress(f"schedules={len(schedules)}/14 selected_games={len(games)}")
    response = fetch_with_retries(m.PITCH_CODES_URL, cache, reuse_dirs=reuse_dirs, retries=retries,
                                 opener=opener, sleeper=sleeper)
    source_records.append(dict(url=m.PITCH_CODES_URL, **{k: v for k, v in response.items() if k != "value"}))
    if response["failure"]:
        failures.append(response["failure"])
        lookup = None
    else:
        try:
            lookup = m.pitch_code_lookup(response["value"])
        except ValueError as exc:
            lookup = None
            failures.append(dict(stage="lookup_schema", url=m.PITCH_CODES_URL, error=str(exc), error_type=type(exc).__name__))
    game_results, all_rows = [], []
    if lookup is not None:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(collect_game, game, lookup, cache, reuse_dirs=reuse_dirs, retries=retries,
                                opener=opener, sleeper=sleeper): game for game in games}
            for job in as_completed(jobs):
                result = job.result()
                all_rows.extend(result.pop("rows"))
                game_results.append(result)
                if result["failure"]:
                    failures.append(result["failure"])
                if progress and (len(game_results) % 25 == 0 or len(game_results) == len(games)):
                    progress(f"processed_games={len(game_results)}/{len(games)} parsed_pitches={len(all_rows)} failures={len(failures)}")
    else:
        game_results = [dict(game=game, audit=None, provenance=None,
                             failure=dict(stage="not_attempted_lookup_unavailable", gamePk=game["gamePk"])) for game in games]
    game_results.sort(key=lambda r: r["game"]["gamePk"])
    all_rows.sort(key=lambda r: (r["gamePk"], r["atBatIndex"], r["pitchIndex"]))
    row_path = cache / "parsed_pitches.json"
    # Compact pitch-level export avoids redundant pretty-print whitespace in the ignored artifact.
    row_raw = json.dumps(all_rows, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    temporary_row_path = row_path.with_suffix(".json.pending")
    temporary_row_path.write_bytes(row_raw)
    temporary_row_path.replace(row_path)
    audit = aggregate_audits(game_results, all_rows)
    audit.update(schedule_days_requested=14, schedule_days_received=len(schedules), failures=failures,
                 collection_complete=(len(schedules) == 14 and lookup is not None and bool(games)
                                      and len(game_results) == len(games) and not failures), **selection)
    manifest = dict(protocol="coach_hypotheses/pitch_quality/CONDITIONAL_PROTOCOL.md",
                    interval=[START_DATE.isoformat(), END_DATE.isoformat()], max_workers=workers,
                    maximum_additional_retries=retries, selection="all_Final_R_officialDate_in_interval_deduplicated_gamePk",
                    source_format="official_mlb_json_not_csv", sources=sorted(source_records, key=lambda x: x["url"]),
                    games=[dict(**g["game"], provenance=g["provenance"], failure=g["failure"]) for g in game_results],
                    pitch_level_export=dict(relative_path="artifacts/mlb_quality_games/parsed_pitches.json",
                                            bytes=len(row_raw), rows=len(all_rows), sha256=hashlib.sha256(row_raw).hexdigest()),
                    generated_at_utc=datetime.now(timezone.utc).isoformat(), raw_redeployment_license_assumed=False,
                    prior_csv_access_failures_unchanged=True, body_to_spin_ready=False, relationships_computed=False)
    results.mkdir(parents=True, exist_ok=True)
    for name, value in (("collection_manifest.json", manifest), ("collection_audit.json", audit),
                        ("collection_games.json", game_results)):
        (results / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    return audit, manifest, row_path.resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    audit, _, path = collect(workers=args.workers, progress=lambda text: print(text, flush=True))
    print(json.dumps(dict(pitch_count=audit["pitch_count"], parsed_games=audit["parsed_game_count"],
                          collection_complete=audit["collection_complete"], parsed_pitches=str(path))))


if __name__ == "__main__":
    main()
