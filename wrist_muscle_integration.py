"""Exploratory MoBL wrist transplant; frozen marker targets, no speed tuning."""
import copy,json,hashlib
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import opensim as o
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from directional_shoulder_fit import batch_refine,orientations,BODIES
from anatomical_shoulder_fit import model_points,arr
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/wrist-muscle'
BASE=ROOT/'artifacts/shoulder-marker-stability'
NAMES=['ECRL','ECRB','ECU','FCR','FCU','PL']

def build():
    o.Logger.setLevelString('error')
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'mobl_source.osim').exists():
        from urllib.request import urlretrieve
        urlretrieve('https://raw.githubusercontent.com/mitkof6/OpenSimDatabase/091168791bf007adb792c0c62488b57fff91de1a/upper_limb/complex/MoBL_ARMS_2016/ModelFiles/MoBL_ARMS_module2_4_allmuscles.osim',OUT/'mobl_source.osim')
    if hashlib.sha256((OUT/'mobl_source.osim').read_bytes()).hexdigest()!='4d6bb22273edfaedd99a71856a93b373ed8d0b6d3e47f12a9cac5906f504cb70':raise ValueError('Donor hash mismatch')
    donor=o.Model(str(OUT/'mobl_source.osim'));s=donor.initSystem()
    manifest=json.loads((ROOT/'artifacts/anatomical-shoulder/source_manifest.json').read_text())
    scales=o.ScaleSet()
    for name in ['humerus','ulna','radius','proximal_row','hand']:
        sc=o.Scale();sc.setSegmentName(name);sc.setApply(True)
        sc.setScaleFactors(o.Vec3(manifest['upper_scale'] if name=='humerus' else manifest['forearm_scale']));scales.cloneAndAppend(sc)
    donor.scale(s,scales,True);donor.printToXML(str(OUT/'donor_scaled.osim'))
    src=E.parse(OUT/'donor_scaled.osim');dst=E.parse(BASE/'scaled_source.osim')
    joints=dst.find('.//JointSet/objects');old=joints.find("*[@name='rc']");joints.remove(old)
    for name in ['radiocarpal','wrist_hand']:
        j=copy.deepcopy(src.find(f".//JointSet/objects/*[@name='{name}']"))
        if name=='radiocarpal':
            j.set('name','rc')
            # Match the existing radius attachment exactly.
            j.find('frames/PhysicalOffsetFrame/translation').text=old.find('frames/PhysicalOffsetFrame/translation').text
        joints.append(j)
    bodies=dst.find('.//BodySet/objects');prox=copy.deepcopy(src.find(".//BodySet/objects/*[@name='proximal_row']"))
    # Kinematic intermediary: retain all existing body mass/inertia exactly.
    prox.find('mass').text='0';prox.find('inertia').text='0 0 0 0 0 0'
    bodies.append(prox)
    for c in src.findall('.//ConstraintSet/objects/*'):
        if c.get('name','').startswith('wrist_hand'):dst.find('.//ConstraintSet/objects').append(copy.deepcopy(c))
    forces=dst.find('.//ForceSet/objects');needed=set()
    for name in NAMES:
        mus=copy.deepcopy(src.find(f".//ForceSet/objects/*[@name='{name}']"));forces.append(mus)
        needed.update(x.text for x in mus.findall('.//wrap_object'))
    for body in src.findall('.//BodySet/objects/*'):
        target=bodies.find(f"*[@name='{body.get('name')}']")
        if target is None:continue
        wraps=target.find('WrapObjectSet/objects')
        if wraps is None:
            ws=E.SubElement(target,'WrapObjectSet');wraps=E.SubElement(ws,'objects')
        for wrap in body.findall('WrapObjectSet/objects/*'):
            if wrap.get('name') in needed and wraps.find(f"*[@name='{wrap.get('name')}']") is None:wraps.append(copy.deepcopy(wrap))
    dst.write(OUT/'scaled_source.osim',encoding='utf-8',xml_declaration=True)
    m=o.Model(str(OUT/'scaled_source.osim'));s=m.initSystem();m.realizePosition(s)
    report=dict(source='https://github.com/mitkof6/OpenSimDatabase/blob/091168791bf007adb792c0c62488b57fff91de1a/upper_limb/complex/MoBL_ARMS_2016/ModelFiles/MoBL_ARMS_module2_4_allmuscles.osim',sha256=hashlib.sha256((OUT/'mobl_source.osim').read_bytes()).hexdigest(),added_muscles=NAMES,muscles=m.getMuscles().getSize(),coordinates=m.getNumCoordinates(),massless_proximal_row=True,existing_body_mass_inertia_unchanged=True,scope='Generic source wrist kinematics, coupled carpal motion, Schutte source muscles and wrap objects. Cross-model anatomical registration is an assumption; no athlete capacity validation or ball dynamics.')
    (OUT/'integration_manifest.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

def fit():
    o.Logger.setLevelString('error');m=o.Model(str(OUT/'scaled_source.osim'));s=m.initSystem();cs=m.getCoordinateSet();coords=[cs.get(i) for i in range(cs.getSize())]
    lo=np.array([c.getRangeMin() for c in coords]);hi=np.array([c.getRangeMax() for c in coords]);rows=[]
    for pitch in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        z=np.load(BASE/f'{pitch}_ik.npz');names=list(z['coordinate_names']);q=np.zeros((len(z['q']),len(coords)))
        for j,c in enumerate(coords):
            if c.getName() in names:q[:,j]=z['q'][:,names.index(c.getName())]
        cn=[c.getName() for c in coords];wi=[cn.index(n) for n in ['deviation','flexion']]
        couplers=[o.CoordinateCouplerConstraint.safeDownCast(m.getConstraintSet().get(n)) for n in ['wrist_hand_r1_con','wrist_hand_r3_con']]
        for i in range(len(q)):
            def seed(x):
                q[i,wi]=x
                for con,value in zip(couplers,x):
                    v=o.Vector(1,float(value));q[i,cn.index(con.getDependentCoordinateName())]=con.getFunction().calcValue(v)
                for c,v in zip(coords,q[i]):c.setValue(s,float(v),False)
                m.realizePosition(s)
                return Rotation.from_matrix(z['orientation_targets'][i,2].T@orientations(m,s)[2]).as_rotvec()
            opt=least_squares(seed,np.zeros(2),bounds=(lo[wi],hi[wi]),max_nfev=60)
            seed(opt.x)
        q,info=batch_refine(m,s,coords,q,z['target'],z['orientation_targets'],lo,hi,max_nfev=30)
        points=[];angles=[];closures=[];rots=[]
        for i,x in enumerate(q):
            for c,v in zip(coords,x):c.setValue(s,float(v),False)
            m.realizePosition(s);points.append(model_points(m,s));rot=orientations(m,s);rots.append(rot)
            angles.append(Rotation.from_matrix(np.einsum('bji,bjk->bik',z['orientation_targets'][i],rot)).magnitude());closures.append(np.max(abs(arr(s.getQErr()))))
        err=np.linalg.norm(np.array(points)-z['target'],axis=2)*1000
        row=dict(pitch=pitch,arm_rmse_mm=float(np.sqrt(np.mean(err[:,3:]**2))),orientation_rmse_deg={n:float(np.rad2deg(np.sqrt(np.mean(np.array(angles)[:,j]**2)))) for j,n in enumerate(BODIES)},constraint_max=float(max(closures)),optimizer=info)
        np.savez_compressed(OUT/f'{pitch}_ik.npz',q=q,time=z['time'],target=z['target'],predicted=points,orientation_targets=z['orientation_targets'],coordinate_names=[c.getName() for c in coords],rotations=rots)
        rows.append(row);print(row,flush=True)
    (OUT/'ik_report.json').write_text(json.dumps(dict(rows=rows),indent=2))

if __name__=='__main__':build();fit()
