"""Acquire the preregistered temporal sample and export unchanged date partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..pitch_quality import collect_games as c

PROTOCOL = "coach_hypotheses/temporal_quality/PROTOCOL.md"
ARTIFACTS = c.m.ROOT / "artifacts" / "temporal_quality_games"
RESULTS = Path(__file__).resolve().parent / "results"
REUSE_DIRS = (
    c.m.ROOT.parent / "mlb-quality-collection-agent-20261030" / "artifacts" / "mlb_quality_games",
    c.m.ROOT.parent / "conditional-pitch-quality-20261030" / "artifacts" / "mlb_quality_games",
    c.DEFAULT_REUSE,
)
SPLITS = dict(dev_train=("2025-06-01", "2025-07-15"),
              dev_validation=("2025-07-16", "2025-07-31"),
              final_test=("2025-08-01", "2025-08-31"))


def file_hash(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return dict(bytes=size, sha256=digest.hexdigest())


def iter_json_rows(path, chunk_size=65536):
    """Decode a JSON object array incrementally rather than loading every row."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("positive_chunk_size_required")
    decoder = json.JSONDecoder()
    buffer, position, eof, started, value_expected, empty_allowed = "", 0, False, False, True, True
    with Path(path).open(encoding="utf-8") as handle:
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                if eof:
                    raise ValueError("truncated_row_array")
                buffer, position = handle.read(chunk_size), 0
                eof = not buffer
                continue
            character = buffer[position]
            if not started:
                if character != "[":
                    raise ValueError("row_array_required")
                started, position = True, position+1
                continue
            if character == "]" and (not value_expected or empty_allowed):
                if (buffer[position+1:] + handle.read()).strip():
                    raise ValueError("trailing_content_after_row_array")
                return
            if not value_expected:
                if character != ",":
                    raise ValueError("row_separator_required")
                position += 1
                value_expected, empty_allowed = True, False
                continue
            try:
                row, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError("invalid_or_truncated_row") from exc
                more = handle.read(chunk_size)
                buffer, position, eof = buffer[position:]+more, 0, not more
                continue
            if not isinstance(row, dict):
                raise ValueError("row_objects_required")
            yield row
            position, value_expected = end, False


def export_splits(row_path, artifact_dir, names, expected_hash, *, seen_keys=None, seen_play_ids=None):
    """Preserve every field/value; split only on officialDate-derived gameDate."""
    if file_hash(row_path) != {k: expected_hash[k] for k in ("bytes", "sha256")}:
        raise ValueError("source_row_file_hash_mismatch")
    if not names or any(name not in SPLITS for name in names) or len(set(names)) != len(names):
        raise ValueError("known_distinct_split_names_required")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    keys = seen_keys if seen_keys is not None else set()
    play_ids = seen_play_ids if seen_play_ids is not None else set()
    pitchers, duplicate_keys, duplicate_ids, previous, total = set(), 0, 0, None, 0
    handles, hashes, sizes, counts, game_ids = {}, {}, {}, {}, {}
    try:
        for name in names:
            handles[name] = (artifact_dir / f"{name}.json.pending").open("wb")
            hashes[name], sizes[name], counts[name], game_ids[name] = hashlib.sha256(), 0, 0, set()
            handles[name].write(b"[")
            hashes[name].update(b"[")
            sizes[name] += 1
        for row in iter_json_rows(row_path):
            key = (row.get("gamePk"), row.get("atBatIndex"), row.get("pitchIndex"))
            if any(type(part) is not int for part in key) or (previous is not None and key < previous):
                raise ValueError("invalid_or_unsorted_source_row_key")
            previous = key
            duplicate_keys += key in keys
            keys.add(key)
            play_id = row.get("playId")
            if isinstance(play_id, str) and play_id:
                duplicate_ids += play_id in play_ids
                play_ids.add(play_id)
            day = row.get("gameDate")
            matched = [name for name in names if isinstance(day, str) and SPLITS[name][0] <= day <= SPLITS[name][1]]
            if len(matched) != 1:
                raise ValueError("row_outside_requested_split_dates")
            name = matched[0]
            raw = (b"," if counts[name] else b"") + json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            handles[name].write(raw)
            hashes[name].update(raw)
            sizes[name] += len(raw)
            counts[name] += 1
            game_ids[name].add(key[0])
            total += 1
            if type(row.get("pitcherId")) is int:
                pitchers.add(row["pitcherId"])
        if "rows" in expected_hash and total != expected_hash["rows"]:
            raise ValueError("source_row_count_mismatch")
        for name in names:
            handles[name].write(b"]")
            handles[name].close()
            hashes[name].update(b"]")
            sizes[name] += 1
        exports = {}
        for name in names:
            final = artifact_dir / f"{name}.json"
            final.with_suffix(".json.pending").replace(final)
            exports[name] = dict(interval=list(SPLITS[name]), rows=counts[name], game_count=len(game_ids[name]),
                                 gamePks=sorted(game_ids[name]), bytes=sizes[name], sha256=hashes[name].hexdigest(),
                                 source_row_file_sha256=expected_hash["sha256"], unchanged_row_values=True,
                                 sort_key=["gamePk", "atBatIndex", "pitchIndex"],
                                 absolute_path=str(final.resolve()),
                                 relative_path=f"artifacts/temporal_quality_games/{name}.json")
        return exports, dict(pitch_count=total, pitcher_ids=pitchers, duplicate_pitch_keys=duplicate_keys,
                             duplicate_play_ids=duplicate_ids)
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()


def write_result(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".pending")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def verify_export_metadata(results_dir=RESULTS, artifact_dir=ARTIFACTS):
    """Verify completed split bytes and publish one consistent completeness schema."""
    results, artifacts = Path(results_dir), Path(artifact_dir)
    audit = json.loads((results / "collection_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((results / "collection_manifest.json").read_text(encoding="utf-8"))
    exports = manifest["split_exports"]
    if set(exports) != set(SPLITS) or audit.get("interval") != ["2025-06-01", "2025-08-31"]:
        raise ValueError("unexpected_collection_interval_or_partitions")
    for name, interval in SPLITS.items():
        path = artifacts / f"{name}.json"
        if exports[name]["interval"] != list(interval) or file_hash(path) != {key: exports[name][key] for key in ("bytes", "sha256")}:
            raise ValueError("completed_partition_hash_or_interval_mismatch")
        exports[name]["absolute_path"] = str(path.resolve())
    if audit["collection_complete"] and (audit["schedule_days_requested"] != 92
            or audit["schedule_days_received"] != 92 or audit["failures"]
            or audit["selected_game_count"] != audit["parsed_game_count"] or audit["parsed_game_count"] <= 0):
        raise ValueError("inconsistent_complete_collection_claim")
    # Only the new temporal reports: one complete game object per line in a JSON array.
    for phase in manifest["phases"]:
        if phase["name"] not in ("development", "final_test"):
            raise ValueError("unexpected_collection_phase")
        game_path = results / f'collection_{phase["name"]}' / "collection_games.json"
        game_values = json.loads(game_path.read_text(encoding="utf-8"))
        if not isinstance(game_values, list) or any(not isinstance(g, dict) for g in game_values):
            raise ValueError("game_audit_object_array_required")
        compact = "[\n" + ",\n".join(json.dumps(g, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                                     for g in game_values) + "\n]\n"
        temporary = game_path.with_suffix(".json.pending")
        temporary.write_text(compact, encoding="utf-8", newline="\n")
        temporary.replace(game_path)
        phase["games_file"] = file_hash(game_path)
    audit["split_exports"] = exports
    for key in ("interval", "collection_complete", "schedule_days_requested", "schedule_days_received",
                "selected_game_count", "parsed_game_count", "failures"):
        manifest[key] = audit[key]
    write_result(results / "collection_audit.json", audit)
    write_result(results / "collection_manifest.json", manifest)
    write_result(results / "collection_splits.json", exports)
    return audit, manifest


def run(*, artifact_dir=ARTIFACTS, results_dir=RESULTS, reuse_dirs=REUSE_DIRS,
        workers=3, retries=2, opener=None, sleeper=c.time.sleep, progress=None):
    artifacts, results = Path(artifact_dir), Path(results_dir)
    phases = (("development", "2025-06-01", "2025-07-31", ("dev_train", "dev_validation")),
              ("final_test", "2025-08-01", "2025-08-31", ("final_test",)))
    phase_records, all_games, exports, pitchers = [], [], {}, set()
    keys, play_ids = set(), set()
    duplicate_keys = duplicate_ids = 0
    for name, start, end, split_names in phases:
        if progress:
            progress(f"phase={name} interval={start}..{end}")
        cache = artifacts / name
        phase_results = results / f"collection_{name}"
        sources = tuple(reuse_dirs)+(artifacts / "development",)
        audit, manifest, row_path = c.collect(cache_dir=cache, results_dir=phase_results,
                            reuse_dirs=sources, workers=workers, retries=retries, opener=opener, sleeper=sleeper,
                            progress=progress, start_date=start, end_date=end, protocol=PROTOCOL,
                            export_relative_path=f"artifacts/temporal_quality_games/{name}/parsed_pitches.json")
        split_exports, stats = export_splits(row_path, artifacts, split_names, manifest["pitch_level_export"],
                                            seen_keys=keys, seen_play_ids=play_ids)
        exports.update(split_exports)
        pitchers.update(stats["pitcher_ids"])
        duplicate_keys += stats["duplicate_pitch_keys"]
        duplicate_ids += stats["duplicate_play_ids"]
        games_file = phase_results / "collection_games.json"
        all_games.extend(json.loads(games_file.read_text(encoding="utf-8")))
        record = dict(name=name, interval=[start, end], audit=audit,
                      manifest_relative_path=f"collection_{name}/collection_manifest.json",
                      manifest_file=file_hash(phase_results / "collection_manifest.json"),
                      games_relative_path=f"collection_{name}/collection_games.json", games_file=file_hash(games_file))
        phase_records.append(record)
        write_result(results / "collection_progress.json", dict(protocol=PROTOCOL, completed_phases=[p["name"] for p in phase_records],
                                                                split_exports=exports, relationships_computed=False))
        if progress:
            progress("DEVELOPMENT_READY " + json.dumps({n: str((artifacts/f"{n}.json").resolve()) for n in split_names})
                     if name == "development" else f"FINAL_TEST_READY {artifacts / 'final_test.json'}")
    aggregate = c.aggregate_audits(all_games, [])
    aggregate.update(pitch_count=sum(x["rows"] for x in exports.values()), pitcher_count=len(pitchers),
                     duplicate_pitch_keys=duplicate_keys, duplicate_play_ids=duplicate_ids,
                     schedule_days_requested=sum(p["audit"]["schedule_days_requested"] for p in phase_records),
                     schedule_days_received=sum(p["audit"]["schedule_days_received"] for p in phase_records),
                     failures=[f for p in phase_records for f in p["audit"]["failures"]],
                     collection_complete=all(p["audit"]["collection_complete"] for p in phase_records),
                     interval=["2025-06-01", "2025-08-31"], no_outcome_relationships_computed=True,
                     split_exports=exports)
    manifest = dict(protocol=PROTOCOL, phases=phase_records, split_exports=exports,
                    final_train=dict(concatenate=["dev_train.json", "dev_validation.json"],
                                     rows=exports["dev_train"]["rows"]+exports["dev_validation"]["rows"]),
                    source_feed_hashes={str(g["game"]["gamePk"]): g["provenance"]["source"]["sha256"]
                                        for g in all_games if g["provenance"] and g["provenance"]["source"]},
                    **{key: aggregate[key] for key in ("interval", "collection_complete", "schedule_days_requested",
                       "schedule_days_received", "selected_game_count", "parsed_game_count", "failures")},
                    max_workers=workers, maximum_additional_retries=retries,
                    prior_results_modified=False, relationships_computed=False, body_to_spin_ready=False)
    write_result(results / "collection_audit.json", aggregate)
    write_result(results / "collection_manifest.json", manifest)
    write_result(results / "collection_splits.json", exports)
    return verify_export_metadata(results, artifacts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    audit, _ = run(workers=args.workers, progress=lambda message: print(message, flush=True))
    print(json.dumps({key: audit[key] for key in ("pitch_count", "parsed_game_count", "collection_complete")}))


if __name__ == "__main__":
    main()
