"""Compare donor template, subject geometry scaling, and muscle force screen."""
import copy,json,shutil
from pathlib import Path
import numpy as np
import opensim as o
import wrist_muscle_integration as wrist
import elbow_registration_audit as elbow
import wrist_residual_audit as audit

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/elbow-moment-validation'
PREVIOUS=ROOT/'artifacts/elbow-registration-audit'
BASE=ROOT/'artifacts/wrist-muscle'
NAMES=['BRD','BRA','ECRL','PT','TRIlat']
LITERATURE={'BRD':77.,'BRA':26.,'ECRL':32.,'PT':17.,'TRIlat':23.}

def get_elbow_vector(path):
    m=o.Model(str(path));s=m.initSystem();m.realizePosition(s)
    p=m.getJointSet().get('elbow').getParentFrame().findTransformInBaseFrame().p()
    return np.array([p.get(i) for i in range(3)])

def build():
    OUT.mkdir(parents=True,exist_ok=True);scratch=OUT/'scratch';(scratch/'artifacts/anatomical-shoulder').mkdir(parents=True,exist_ok=True);(scratch/'artifacts/wrist-muscle').mkdir(parents=True,exist_ok=True)
    source=BASE/'mobl_source.osim';shutil.copyfile(source,scratch/'artifacts/wrist-muscle/mobl_source.osim')
    manifest=json.loads((ROOT/'artifacts/anatomical-shoulder/source_manifest.json').read_text())
    subject=np.linalg.norm(get_elbow_vector(BASE/'scaled_source.osim'));template=np.linalg.norm(get_elbow_vector(source));new_scale=subject/template
    old_scale=manifest['upper_scale'];manifest['upper_scale']=float(new_scale)
    (scratch/'artifacts/anatomical-shoulder/source_manifest.json').write_text(json.dumps(manifest,indent=2))
    wrist.ROOT=scratch;wrist.OUT=scratch/'artifacts/wrist-muscle';wrist.BASE=ROOT/'artifacts/shoulder-marker-stability';wrist.build()
    elbow.INPUT=wrist.OUT;elbow.OUT=OUT/'matched';elbow.build()
    info={'subject_joint_center_distance_mm':float(subject*1000),'source_template_distance_mm':float(template*1000),'previous_humerus_scale':old_scale,'new_humerus_scale':float(new_scale),'assumption':'Uniform donor humerus scaling to match the already scaled reference shoulder-to-elbow distance; not measured bone length or personal anatomy. Forearm scale and all Fmax retained.'}
    (OUT/'scale_manifest.json').write_text(json.dumps(info,indent=2));print(info,flush=True)

def moment_curves():
    paths={'source_unscaled':BASE/'mobl_source.osim','source_prior_scaled':BASE/'donor_scaled.osim','subject_matched':OUT/'scratch/artifacts/wrist-muscle/donor_scaled.osim','transplanted_prior':PREVIOUS/'registered_plus_elbow.osim','transplanted_corrected':OUT/'matched/registered_plus_elbow.osim'}
    result={}
    # Published Murray range: 20–120° flexors, 30–120° triceps,
    # neutral forearm (0°); actual segment axes remain model-specific.
    for label,path in paths.items():
        m=o.Model(str(path));s=m.initSystem();cs=m.getCoordinateSet();cs.get('pro_sup').setValue(s,0.,False);c=cs.get('elbow_flexion');v={n:[] for n in NAMES}
        for degree in np.arange(20.,121.,1.):
            c.setValue(s,float(np.deg2rad(degree)),False);m.realizePosition(s)
            for n in NAMES:v[n].append(float(m.getMuscles().get(n).computeMomentArm(s,c)*1000))
        result[label]={}
        for n,series in v.items():
            valid=np.array(series[10:] if n=='TRIlat' else series);peak=float(np.max(np.abs(valid)))
            result[label][n]={'peak_mm':peak,'mean_abs_mm':float(np.mean(np.abs(valid))),'at_90_mm':series[70],'reference_group_mean_peak_mm':LITERATURE[n],'model_values_mm':series}
    (OUT/'moment_curves.json').write_text(json.dumps(result,indent=2));return result

def run():
    o.Logger.setLevelString('error');build();curves=moment_curves();audit.INPUT=BASE;rows=[]
    models={'prior_39':PREVIOUS/'registered_wrist.osim','corrected_39':OUT/'matched/registered_wrist.osim','prior_47':PREVIOUS/'registered_plus_elbow.osim','corrected_47':OUT/'matched/registered_plus_elbow.osim'}
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        for key,path in models.items():
            audit.OUT=OUT/'series'/key;audit.OUT.mkdir(parents=True,exist_ok=True)
            r=audit.run(path,pitch,'original');r['condition']=key;rows.append(r)
            (OUT/'results.json').write_text(json.dumps({'rows':rows},indent=2))
    return curves,rows

if __name__=='__main__':run()
