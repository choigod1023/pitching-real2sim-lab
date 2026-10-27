"""One muscle, frozen pose: wrap association and path history diagnostics."""
import copy,json
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import opensim as o
from anatomical_shoulder_fit import arr
from anatomical_muscle_screen import ov
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/wrapping-consistency';SOURCE=ROOT/'artifacts/shoulder-load-audit/mass_corrected.osim'
MUSCLE='Subscapularis_I'

def check(path):
    m=o.Model(str(path));s=m.initSystem();cs=m.getCoordinateSet();mat=m.getMatterSubsystem();nu=s.getNU();z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz')
    for j,v in enumerate(z['q'][28]):cs.get(j).setValue(s,float(v),False)
    m.realizePosition(s);Q=arr(s.getQ());G=[];u=np.linspace(-.1,.1,nu)
    for k in range(nu):
        g=o.Vector();mat.multiplyByG(s,ov(np.eye(nu)[k]),g);G.append(arr(g))
    G=np.array(G).T;u-=G.T@np.linalg.lstsq(G@G.T,G@u,rcond=None)[0];s.setU(ov(u));m.realizeVelocity(s);mus=m.getMuscles().get(MUSCLE);p=mus.getGeometryPath();length=mus.getLength(s);speed=p.getLengtheningSpeed(s);dq=o.Vector();mat.multiplyByN(s,False,ov(u),dq);dq=arr(dq);rows=[]
    for eps in [1e-3,1e-4,1e-5,1e-6]:
        ls=[]
        for sign in [-1,1]:s.setQ(ov(Q+sign*eps*dq));m.realizePosition(s);ls.append(mus.getLength(s))
        rows.append({'eps_s':eps,'length_derivative_m_s':float((ls[1]-ls[0])/2/eps),'speed_error_m_s':float(abs((ls[1]-ls[0])/2/eps-speed))})
    s.setQ(ov(Q));s.setU(ov(u));m.realizeVelocity(s);warm_length=mus.getLength(s);warm_speed=p.getLengtheningSpeed(s)
    # Endpoint inclusion in each obstacle in its own surface frame.
    points=p.getPathPointSet();inside=[]
    for k in range(p.getWrapSet().getSize()):
        wrap=p.getWrapSet().get(k).getWrapObject();X=wrap.getTransform();R=X.R().asMat33();R=np.array([[R.get(a,b) for b in range(3)] for a in range(3)]);translation=arr(X.p());frame=wrap.getFrame()
        for j in [0,points.getSize()-1]:
            pp=points.get(j);globalp=pp.getLocationInGround(s);local=arr(m.getGround().findStationLocationInAnotherFrame(s,globalp,frame));v=R.T@(local-translation)
            if wrap.getConcreteClassName()=='WrapSphere':value=float(np.linalg.norm(v)/o.WrapSphere.safeDownCast(wrap).get_radius())
            elif wrap.getConcreteClassName()=='WrapCylinder':value=float(np.linalg.norm(v[:2])/o.WrapCylinder.safeDownCast(wrap).get_radius())
            else:value=None
            inside.append({'wrap':wrap.getName(),'point':pp.getName(),'radial_ratio':value})
    return {'model':path.name,'cold_length_m':float(length),'cold_speed_m_s':float(speed),'warm_length_m':float(warm_length),'warm_speed_m_s':float(warm_speed),'sweep':rows,'endpoint_radial_ratios':inside}

def main():
    OUT.mkdir(parents=True,exist_ok=True);o.Logger.setLevelString('error');tree=E.parse(SOURCE);objects=tree.find(f".//ForceSet/objects/*[@name='{MUSCLE}']/GeometryPath/PathWrapSet/objects");names=[x.findtext('wrap_object') for x in objects];reports=[]
    for label,order in [('original',names),('reverse',names[::-1]),('first_only',names[:1]),('second_only',names[1:]),('none',[])]:
        t=copy.deepcopy(tree);dest=t.find(f".//ForceSet/objects/*[@name='{MUSCLE}']/GeometryPath/PathWrapSet/objects");old=list(dest)
        for x in old:dest.remove(x)
        for name in order:dest.append(copy.deepcopy(next(x for x in old if x.findtext('wrap_object')==name)))
        p=OUT/f'{label}.osim';t.write(p,encoding='utf-8',xml_declaration=True);r=check(p);reports.append(r);print(r,flush=True)
    (OUT/'isolation.json').write_text(json.dumps({'muscle':MUSCLE,'pitch':'3034_4','sample':28,'reports':reports},indent=2))

if __name__=='__main__':main()
