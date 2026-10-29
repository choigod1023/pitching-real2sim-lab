"""Audit already-downloaded originals; no speed prediction or tuning."""
import argparse
import json
from pathlib import Path
import subprocess

from .audit import (archive_trials, digest, extract_member, inspect_c3d, inspect_mat_cohort,
                    link_trials, read_trial_book, select_new_trials)
from .requirements import readiness, scale_bias_kmh, slope_noise_sd_kmh, timing_budget_ms


def run(raw, download_manifest, mat_root, output):
    raw, output = Path(raw), Path(output)
    manifest = json.loads(Path(download_manifest).read_text(encoding='utf-8'))
    expected = {f['filename']: f['sha256'] for f in manifest['files']}
    hashes = {}
    for filename in ('S1.rar', 'Trialinfo.xlsx'):
        actual = digest(raw/filename)
        if actual != expected[filename]:
            raise ValueError('changed_original:' + filename)
        hashes[filename] = actual
    listed = subprocess.run(['tar', '-tf', str(raw/'S1.rar')], capture_output=True,
                            text=True, encoding='utf-8', errors='strict', check=True).stdout.splitlines()
    trials = archive_trials(listed)
    book = read_trial_book(raw/'Trialinfo.xlsx', 'S1')
    linked = link_trials(trials, book)
    selected = select_new_trials(trials, raw)
    output.mkdir(parents=True, exist_ok=True)
    (output/'selected_before_inspection.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    inspections = [inspect_c3d(extract_member(raw/'S1.rar', name, output/'extracted')) for name in selected]
    statics = [name for name in listed if name.lower().endswith('.c3d') and '_static_' in name.lower()]
    summary = dict(subjects_locally_archived=1, sessions=len({r['day'] for r in trials}),
                   dynamic_c3d=len(trials), static_c3d=len(statics), workbook_trial_rows=len(book),
                   archive_unique_trial_keys=len({(r['subject'], r['day'], r['side'], r['trial_number']) for r in trials}),
                   unique_workbook_links=sum(r['workbook_matches'] == 1 for r in linked),
                   missing_workbook_links=sum(r['workbook_matches'] == 0 for r in linked),
                   ambiguous_workbook_links=sum(r['workbook_matches'] > 1 for r in linked),
                   linked_speed_record_present=sum(r.get('speed_record_present', False) for r in linked),
                   status_disagreements=[r['member'] for r in linked if r.get('status_agrees') is False],
                   status_code_disagreements=[r['member'] for r in linked if r.get('status_code_agrees') is False],
                   speed_unit='UNKNOWN', ball_center_validated=False)
    result = dict(protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),
                  input_hashes=hashes, ozkaya_summary=summary,
                  new_c3d_inspections=inspections, mat_summary=inspect_mat_cohort(mat_root),
                  engineering_scenarios=dict(
                      position_sd_mm=.5, fps=1000, samples=11,
                      ideal_slope_noise_sd_kmh=slope_noise_sd_kmh(.5, 1000, 11),
                      scale_1percent_at_150kmh_bias_kmh=scale_bias_kmh(150, .01),
                      scale_point2percent_at_150kmh_bias_kmh=scale_bias_kmh(150, .002),
                      example_acceleration_mps2=1000., example_speed_budget_kmh=.3,
                      conditional_sync_budget_ms=timing_budget_ms(.3, 1000)),
                  speed_model_fitted=False, athlete_intervention_performed=False)
    registry = json.loads(Path(__file__).with_name('SOURCE_EVIDENCE.json').read_text(encoding='utf-8'))
    result['reviewed_source_readiness'] = {s['id']: readiness(s['evidence']) for s in registry['sources']}
    (output/'archive_workbook_links.json').write_text(json.dumps(linked, indent=2), encoding='utf-8')
    (output/'audit_results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(ozkaya=summary, mat={k:v for k,v in result['mat_summary'].items() if k!='records'},
                         selected=selected), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--download-manifest', type=Path, required=True)
    parser.add_argument('--mat-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('artifacts')/'audit')
    args = parser.parse_args()
    run(args.raw, args.download_manifest, args.mat_root, args.output)


if __name__ == '__main__':
    main()
