"""Post-preregistered numerical follow-up; original design verdicts stay fixed."""
import argparse
import json
from pathlib import Path

from .uncertainty import (PITCHES, prepare_case, speeds, grouped_mc, convergence, file_hash)


def refine(data, output):
    original_files = {name: file_hash(output/name) for name in ('scenario_results.json', 'mc_designs.json', 'recorded_comparison.json', 'manifest.json')}
    original = json.loads((output/'mc_designs.json').read_text(encoding='utf-8'))
    followups = []
    for pitch in PITCHES:
        case = prepare_case(data, pitch)
        for mode, radius, motion in [('primary',0.,True), ('relative_offset_stress',.017,True), ('constant_distance_companion',0.,False)]:
            previous = next(row for row in original if row['pitch'] == pitch and row['mode'] == mode)
            additional = [grouped_mc(lambda u:speeds(case,u,radius,motion),131072,seed) for seed in (20261029,20261030)]
            comparisons = []
            for result in additional:
                prior = next(row for row in previous['results'] if row['n_per_matrix'] == 32768 and row['seed'] == result['seed'])
                comparisons.append(convergence([prior,result])['comparisons'][0])
            followups.append({'pitch': pitch, 'mode': mode, 'original_verdict_retained': previous['convergence']['verdict'],
                              'additional_results': additional, 'same_seed_32768_to_131072': comparisons,
                              'seed_comparison_131072': convergence(additional)})
    preserved = all(file_hash(output/name) == value for name,value in original_files.items())
    if not preserved:
        raise AssertionError('original_results_modified')
    package = Path(__file__).parent
    payload = {'followup_protocol_commit': 'e9fb983', 'heading_correction_commit': '938cc8c',
               'original_results_sha256': original_files, 'original_results_preserved': preserved,
               'followups': followups,
               'current_code_sha256': {p.name:file_hash(p) for p in package.glob('*.py')},
               'followup_protocol_sha256':file_hash(package/'FOLLOWUP_PRECISION_PROTOCOL.md')}
    (output/'precision_followup.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).parent.resolve()):
        parser.error('output must remain in coach_hypotheses/stage2')
    payload=refine(args.data,args.output)
    print('Additional designs:',len(payload['followups']),'Original results preserved:',payload['original_results_preserved'])


if __name__=='__main__':main()
