"""Experimental multi-obstacle geodesic wrapping with original geometry.

Only fixed two-point paths are supported. The source quadrant is approximated
by a frozen source contact-side hint, not an anatomical calibration.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import opensim as o
from anatomical_shoulder_fit import arr
from anatomical_muscle_screen import ov
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/wrapping-consistency';SOURCE=ROOT/'artifacts/shoulder-load-audit/mass_corrected.osim'

def convert(names):
    m=o.Model(str(SOURCE));s=m.initSystem();cs=m.getCoordinateSet();z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz')
    for j,v in enumerate(z['q'][28]):cs.get(j).setValue(s,float(v),False)
    m.realizePosition(s);tree=E.parse(SOURCE);specs=[]
    for name in names:
        muscle=m.getMuscles().get(name);old=muscle.getGeometryPath();muscle.getLength(s);pts=old.getPathPointSet()
        if pts.getSize()!=2 or any(pts.get(i).getConcreteClassName()!='PathPoint' for i in range(pts.getSize())):raise ValueError('Only fixed two-point paths supported')
        endpoints=[(pts.get(i).getParentFrame().getAbsolutePathString(),arr(pts.get(i).getLocation(s))) for i in range(2)];wraps=[]
        for j in range(old.getWrapSet().getSize()):
            ws=old.getWrapSet().get(j);wrap=ws.getWrapObject();xml=tree.find(f".//WrapObjectSet/objects/*[@name='{wrap.getName()}']");X=wrap.getTransform();R=X.R().asMat33();R=np.array([[R.get(a,b) for b in range(3)] for a in range(3)])
            current=old.getCurrentPath(s);candidates=[current.get(k) for k in range(current.getSize()) if current.get(k).getName()=='pwpt1' and current.get(k).getParentFrame().getAbsolutePathString()==wrap.getFrame().getAbsolutePathString()]
            if not candidates:raise ValueError('No source wrap contact hint at diagnostic seed pose')
            gp=candidates[0].getLocationInGround(s);local=arr(m.getGround().findStationLocationInAnotherFrame(s,gp,wrap.getFrame()));hint=R.T@(local-arr(X.p()))
            wraps.append({'name':wrap.getName(),'type':xml.tag,'frame':wrap.getFrame().getAbsolutePathString(),'location':np.fromstring(xml.findtext('translation'),sep=' ').tolist(),'orientation':np.fromstring(xml.findtext('xyz_body_rotation'),sep=' ').tolist(),'radius':float(xml.findtext('radius')) if xml.find('radius') is not None else None,'hint':hint.tolist(),'quadrant':xml.findtext('quadrant')})
        specs.append({'muscle':name,'endpoints':endpoints,'wraps':wraps})
    # Replace component types only in a fresh, uninitialized topology. The
    # source inspection State must never retain pointers to a replaced path.
    m=o.Model(str(SOURCE))
    keepalive=[]
    for spec in specs:
        mus=m.getMuscles().get(spec['muscle']);mus.set_path(o.Scholz2015GeometryPath());mus.finalizeFromProperties();path=o.Scholz2015GeometryPath.safeDownCast(mus.updPath());path.setName('path')
        frame,location=spec['endpoints'][0];path.appendPathPoint(m.getBodySet().get(frame.split('/')[-1]),o.Vec3(*location))
        for j,w in enumerate(spec['wraps']):
            frame=m.getBodySet().get(w['frame'].split('/')[-1]);location=o.Vec3(*w['location']);orientation=o.Vec3(*w['orientation'])
            if w['type']=='WrapCylinder':geometry=o.ContactCylinder(w['radius'],location,orientation,frame)
            elif w['type']=='WrapSphere':geometry=o.ContactSphere(w['radius'],location,frame)
            else:raise ValueError('Unsupported wrap type '+w['type'])
            geometry.setName(f'geodesic_{spec["muscle"]}_{j}');m.addComponent(geometry);keepalive.append(geometry);path.appendObstacle(geometry,o.Vec3(*w['hint']))
        frame,location=spec['endpoints'][1];path.appendPathPoint(m.getBodySet().get(frame.split('/')[-1]),o.Vec3(*location))
    for j,v in enumerate(z['q'][28]):m.getCoordinateSet().get(j).setDefaultValue(float(v))
    m.finalizeConnections();m.initSystem();path=OUT/'geodesic_trial.osim';m.printToXML(str(path))
    (OUT/'geodesic_manifest.json').write_text(json.dumps(specs,indent=2,default=lambda x:x.tolist()));return path

def evaluate(path,names,selection=None):
    m=o.Model(str(path));s=m.initSystem();cs=m.getCoordinateSet();mat=m.getMatterSubsystem();nu=s.getNU();rows=[]
    for pitch in [f'3034_{i}' for i in range(1,6)]:
        z=np.load(ROOT/f'artifacts/wrist-muscle/{pitch}_ik.npz')
        for sample in [2,len(z['q'])//2,len(z['q'])-3]:
            if selection is not None and (pitch,sample) not in selection:continue
            for j,v in enumerate(z['q'][sample]):cs.get(j).setValue(s,float(v),False)
            m.realizePosition(s);Q=arr(s.getQ());G=[];u=np.linspace(-.1,.1,nu)
            for k in range(nu):
                g=o.Vector();mat.multiplyByG(s,ov(np.eye(nu)[k]),g);G.append(arr(g))
            G=np.array(G).T;u-=G.T@np.linalg.lstsq(G@G.T,G@u,rcond=None)[0];s.setU(ov(u));m.realizeVelocity(s);dq=o.Vector();mat.multiplyByN(s,False,ov(u),dq);dq=arr(dq)
            rates=[m.getMuscles().get(n).getPath().getLengtheningSpeed(s) for n in names];lengths=[m.getMuscles().get(n).getLength(s) for n in names];sweeps=[]
            for eps in [1e-3,1e-4,1e-5,1e-6]:
                vals=[]
                for sign in [-1,1]:
                    s.setQ(ov(Q+sign*eps*dq));m.realizePosition(s);vals.append([m.getMuscles().get(n).getLength(s) for n in names])
                derivative=(np.array(vals[1])-vals[0])/2/eps;sweeps.append({'eps_s':eps,'derivative_m_s':derivative.tolist(),'error_m_s':abs(derivative-rates).tolist()})
            rows.append({'pitch':pitch,'sample':int(sample),'names':names,'length_m':lengths,'path_speed_m_s':rates,'sweeps':sweeps});print(pitch,sample,max(v for sw in sweeps for v in sw['error_m_s']),flush=True)
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True);o.Logger.setLevelString('error');names=['Subscapularis_I']
    try:path=convert(names)
    except RuntimeError as exc:
        (OUT/'geodesic_status.json').write_text(json.dumps({'status':'rejected_initialization','error':str(exc),'scope':'Experimental solver; no same-anatomy replacement validated. Source quadrant restrictions are not preserved by full contact surfaces.'},indent=2));print(str(exc),flush=True);return
    before=evaluate(SOURCE,names)
    try:after=evaluate(path,names)
    except RuntimeError as exc:
        (OUT/'geodesic_status.json').write_text(json.dumps({'status':'rejected_evaluation','error':str(exc),'scope':'Seed initialization succeeded but not all development states. Full contact surfaces do not preserve source quadrant restrictions.'},indent=2));print(str(exc),flush=True);return
    (OUT/'geodesic_checks.json').write_text(json.dumps({'names':names,'before':before,'after':after},indent=2))
    (OUT/'geodesic_status.json').write_text(json.dumps({'status':'evaluated','scope':'Experimental solver; source quadrant restrictions still differ.'},indent=2))

if __name__=='__main__':main()
