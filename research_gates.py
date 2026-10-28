"""Fail-closed research progression; code tests cannot replace scientific evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'AUTONOMOUS_RESEARCH_PROTOCOL.md'
DEFAULT_STATE=ROOT/'artifacts/research-gates/state.json'
CHECKS={
 'data':('same_pitch_identity','sources_hashed','units_coordinates_time_verified','body_hand_ball_observed',
         'release_window_usable','alignment_uncertainty_quantified','speed_reference_definition_documented','fit_eval_split_frozen'),
 'hand_ball':('no_future_ball_used_in_prediction','parameters_fit_on_training_only','contact_release_physics_checks',
              'beats_simple_comparator_on_holdout','velocity_direction_position_jointly_validated','assumption_sensitivity_passed','code_tests_passed'),
 'heldout':('unseen_data_verified','change_beats_no_change','change_beats_simpler_model','uncertainty_includes_measurement',
            'precision_resolves_target_effect','sample_units_and_precision_justified','failed_excluded_trials_reported'),
 'whole_body':('joint_pose_validated','grf_and_support_not_double_counted','release_accuracy_preserved',
               'work_energy_consistent','muscle_effort_constraints_validated','code_tests_passed'),
 'candidate':('candidate_conclusion_supported','all_peak_force_budgets_nonincreasing',
              'all_positive_negative_work_budgets_nonincreasing','pose_joint_ranges_preserved',
              'spin_command_claims_supported_or_withheld','full_sensitivity_grid_reported','pitching_overlay_export_verified')}
STAGES=tuple(CHECKS)
KINDS={'measurement','calculation','source','review','tests','video'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def initialize(path=DEFAULT_STATE):
    path=Path(path)
    if path.exists():raise FileExistsError('Existing research state is never reset by init')
    state={'current_stage':'data','workflow_status':'active','protocol_sha256':sha(PROTOCOL),'history':[]}
    atomic_write(path,state);return state


def verify(stage,receipt,base):
    errors=[]
    if stage not in CHECKS:raise ValueError('Unknown stage')
    if receipt.get('stage')!=stage:errors.append('receipt_stage_mismatch')
    if receipt.get('protocol_sha256')!=sha(PROTOCOL):errors.append('protocol_changed_or_missing')
    if not receipt.get('dataset_pitch_scope'):errors.append('missing_dataset_pitch_scope')
    artifacts=receipt.get('artifacts',{})
    valid={}
    for key,item in artifacts.items():
        if not isinstance(item,dict):errors.append('invalid_artifact:'+key);continue
        path=Path(item.get('path',''))
        if not path.is_absolute():path=Path(base)/path
        try:
            if item.get('kind') not in KINDS or not path.is_file() or sha(path)!=item.get('sha256'):
                errors.append('artifact_missing_or_changed:'+key)
            else:valid[key]=item['kind']
        except OSError:errors.append('artifact_unreadable:'+key)
    for name in CHECKS[stage]:
        check=receipt.get('checks',{}).get(name,{})
        if check.get('status')!='PASS':errors.append('check_not_passed:'+name)
        refs=check.get('evidence',[])
        if not isinstance(refs,list) or not refs or not all(x in valid for x in refs):
            errors.append('missing_check_evidence:'+name)
        elif all(valid[x]=='tests' for x in refs) and name!='code_tests_passed':
            errors.append('tests_are_not_scientific_validation:'+name)
        if not isinstance(check.get('reason'),str) or not check['reason'].strip():errors.append('missing_reason:'+name)
    return {'stage':stage,'passed':not errors,'errors':errors,
            'scope':'Evidence integrity and progression guard; scientific claims require the referenced source/measurement/calculation review.'}


def validate_history(state):
    if state.get('protocol_sha256')!=sha(PROTOCOL):raise ValueError('Protocol changed; invalidate/review prior results before continuing')
    history=state.get('history')
    if not isinstance(history,list) or len(history)>len(STAGES):raise ValueError('Invalid history')
    expected=STAGES[len(history)] if len(history)<len(STAGES) else None
    if state.get('current_stage')!=expected:raise ValueError('Current stage does not match verified history')
    if (state.get('workflow_status')=='complete') != (expected is None):raise ValueError('Completion does not match verified history')
    for i,item in enumerate(state['history']):
        path=Path(item['receipt'])
        if item['stage']!=STAGES[i] or not path.is_file() or sha(path)!=item['receipt_sha256']:
            raise ValueError('Prior receipt missing/changed or stages skipped')
        result=verify(item['stage'],read(path),path.parent)
        if not result['passed']:raise ValueError('Prior gate no longer passes: '+str(result['errors']))


def advance(state_path,receipt_path):
    state_path=Path(state_path);receipt_path=Path(receipt_path).resolve()
    lock=state_path.with_name(state_path.name+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        state=read(state_path);validate_history(state)
        if state['workflow_status']!='active':raise ValueError('Workflow is not active')
        stage=state['current_stage'];result=verify(stage,read(receipt_path),receipt_path.parent)
        if not result['passed']:return result
        state['history'].append({'stage':stage,'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),
                                 'passed_at_utc':datetime.now(timezone.utc).isoformat()})
        count=len(state['history'])
        state['current_stage']=STAGES[count] if count<len(STAGES) else None
        if count==len(STAGES):state['workflow_status']='complete'
        atomic_write(state_path,state)
        return {**result,'next_stage':state['current_stage'],'workflow_status':state['workflow_status']}
    finally:
        os.close(fd);lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('init','status','verify','advance'))
    p.add_argument('--state',type=Path,default=DEFAULT_STATE);p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.action=='init':result=initialize(a.state)
    elif a.action=='status':
        result=read(a.state);validate_history(result)
    elif a.action=='verify':
        if a.receipt is None:p.error('--receipt required')
        state=read(a.state);validate_history(state)
        result=verify(state['current_stage'],read(a.receipt),a.receipt.resolve().parent)
    else:
        if a.receipt is None:p.error('--receipt required')
        result=advance(a.state,a.receipt)
    print(json.dumps(result,ensure_ascii=True,indent=2))
    if result.get('passed') is False:raise SystemExit(2)
