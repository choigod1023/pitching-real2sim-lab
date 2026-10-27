"""Length-only surrogate for one fixed-point shoulder muscle.

The geometry box uses already-seen development angles; validation points are
synthetic source geometry, NOT new measured athlete trials. No radar fitting.
"""
import itertools,json
from pathlib import Path
import numpy as np
import opensim as o
from scipy.stats import qmc
from anatomical_shoulder_fit import arr
from anatomical_muscle_screen import ov
from scholz_wrapping_trial import evaluate
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/wrapping-consistency';SOURCE=ROOT/'artifacts/shoulder-load-audit/mass_corrected.osim'
COORDS=['plane_elv','shoulder_elv','axial_rot'];NAME='Subscapularis_I'

def exponents(degree):return [x for x in itertools.product(range(degree+1),repeat=3) if sum(x)<=degree]
def design(x,powers):return np.array([np.prod(x**p,axis=1) for p in powers]).T

def main(local=False,radius=.02):
    if local and (not np.isfinite(radius) or radius<=0):raise ValueError('Positive finite local radius required')
    OUT.mkdir(parents=True,exist_ok=True);o.Logger.setLevelString('error');m=o.Model(str(SOURCE));s=m.initSystem();cs=m.getCoordinateSet();mus=m.getMuscles().get(NAME);observed=[]
    for i in range(1,6):
        z=np.load(ROOT/f'artifacts/wrist-muscle/3034_{i}_ik.npz');n=list(z['coordinate_names']);observed.extend(z['q'][:,[n.index(c) for c in COORDS]])
    observed=np.array(observed);low=observed.min(0);high=observed.max(0)
    if local:
        z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz');names=list(z['coordinate_names']);seed=z['q'][28,[names.index(n) for n in COORDS]]
        low=np.maximum(seed-radius,[cs.get(n).getRangeMin() for n in COORDS]);high=np.minimum(seed+radius,[cs.get(n).getRangeMax() for n in COORDS])
    center=(low+high)/2;scale=(high-low)/2
    unit=qmc.Sobol(3,scramble=True,seed=20261027).random_base2(12);q=low+(high-low)*unit;length=[]
    for values in q:
        for n,v in zip(COORDS,values):cs.get(n).setValue(s,float(v),False)
        m.realizePosition(s);length.append(mus.getLength(s))
    length=np.array(length);x=(q-center)/scale;train=np.arange(0,2048);selection=np.arange(2048,3072);test=np.arange(3072,4096);models=[]
    for degree in [3,5,7,9]:
        powers=exponents(degree);A=design(x,powers);coef=np.linalg.lstsq(A[train],length[train],rcond=1e-10)[0];e=A@coef-length
        models.append({'degree':degree,'powers':powers,'coef':coef,'selection_rmse_mm':float(np.sqrt(np.mean(e[selection]**2))*1000),'selection_max_mm':float(np.max(abs(e[selection]))*1000),'test_rmse_mm':float(np.sqrt(np.mean(e[test]**2))*1000),'test_max_mm':float(np.max(abs(e[test]))*1000)})
    chosen=min(models,key=lambda x:x['selection_rmse_mm']);geometry_pass=chosen['test_rmse_mm']<=.5 and chosen['test_max_mm']<=2.
    report={'muscle':NAME,'local_only':local,'local_radius_rad':radius if local else None,'scope':'Synthetic source-geometry fit/selection/test. Local trial is limited to its recorded radius around one development pose, clipped to original coordinate bounds; full trial uses five development-pitch bounds. Neither is anatomical or causal validation.','domain_low_rad':low.tolist(),'domain_high_rad':high.tolist(),'models':[{k:v for k,v in r.items() if k not in ['powers','coef']} for r in models],'selected_degree':chosen['degree'],'geometry_pass':geometry_pass,'engineering_criteria':{'test_rmse_mm':.5,'test_max_mm':2.},'muscle_Fmax_changed':False}
    # Lepton expressions provide analytic length derivatives. No speed or
    # moment-arm function is supplied, ensuring one shared scalar length.
    terms=[]
    for coefficient,powers in zip(chosen['coef'],chosen['powers']):
        if abs(coefficient)<1e-15:continue
        term=f'({coefficient:.17g})'
        for j,p in enumerate(powers):
            if p:term+=f'*(((q{j}-({center[j]:.17g}))/({scale[j]:.17g}))^{p})'
        terms.append(term)
    expression='+'.join(terms);new=o.Model(str(SOURCE));mus=new.getMuscles().get(NAME);path=o.FunctionBasedPath();path.setName('path')
    for n in COORDS:path.appendCoordinatePath('/jointset/GlenoHumeral/'+n)
    path.setLengthFunction(o.ExpressionBasedFunction(expression,['q0','q1','q2']));mus.set_path(path);mus.finalizeFromProperties();new.finalizeConnections();new.initSystem();prefix=('local_'+str(radius).replace('.','p')+'_') if local else '';file=OUT/f'{prefix}length_based_trial.osim';new.printToXML(str(file))
    selection=[('3034_4',28)] if local else None
    report['before']=evaluate(SOURCE,[NAME],selection);report['after']=evaluate(file,[NAME],selection);(OUT/f'{prefix}length_based_checks.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ['before','after']},flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--local',action='store_true');parser.add_argument('--radius',type=float,default=.02);args=parser.parse_args();main(args.local,args.radius)
