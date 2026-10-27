"""Independent finite-difference path derivatives and generalized power checks."""
import json
from pathlib import Path
import numpy as np
import opensim as o
import xml.etree.ElementTree as E
from anatomical_shoulder_fit import arr
from anatomical_muscle_screen import ov
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/research-evidence-audit'

def main():
    o.Logger.setLevelString('error');m=o.Model(str(ROOT/'artifacts/shoulder-load-audit/mass_corrected.osim'));s=m.initSystem();cs=m.getCoordinateSet();mus=[m.getMuscles().get(i) for i in range(m.getMuscles().getSize())];nu=s.getNU();matter=m.getMatterSubsystem();reports=[]
    for pitch in [f'3034_{i}' for i in range(1,6)]:
        z=np.load(ROOT/f'artifacts/wrist-muscle/{pitch}_ik.npz')
        for sample in [2,len(z['q'])//2,len(z['q'])-3]:
            for j,v in enumerate(z['q'][sample]):cs.get(j).setValue(s,float(v),False)
            m.realizePosition(s);Q=arr(s.getQ());u=np.linspace(-.1,.1,nu);G=[]
            for k in range(nu):
                g=o.Vector();matter.multiplyByG(s,ov(np.eye(nu)[k]),g);G.append(arr(g))
            G=np.array(G).T;u-=G.T@np.linalg.lstsq(G@G.T,G@u,rcond=None)[0]
            s.setU(ov(u));m.realizeVelocity(s)
            rates=np.array([x.getGeometryPath().getLengtheningSpeed(s) for x in mus]);dq=o.Vector();matter.multiplyByN(s,False,ov(u),dq);dq=arr(dq);sweep=[]
            for eps in [1e-3,1e-4,1e-5,1e-6]:
                values=[]
                for sign in [-1.,1.]:
                    s.setQ(ov(Q+sign*eps*dq));m.realizePosition(s);values.append(np.array([x.getLength(s) for x in mus]))
                fd=(values[1]-values[0])/(2*eps);err=abs(fd-rates)
                sweep.append({'epsilon_s':eps,'max_error_m_s':float(err.max()),'largest_error_muscle':mus[int(err.argmax())].getName()})
            s.setQ(ov(Q));m.realizePosition(s);R=[]
            for k in range(nu):
                s.setU(ov(np.eye(nu)[k]));m.realizeVelocity(s);R.append([-x.getGeometryPath().getLengtheningSpeed(s) for x in mus])
            R=np.array(R);f=np.array([x.getMaxIsometricForce()*.3 for x in mus]);power=(R@f)@u;expected=-f@rates;err=abs(fd-rates)
            reports.append({'pitch':pitch,'sample':int(sample),'muscles':len(mus),'constraint_velocity_norm':float(np.linalg.norm(G@u)),'epsilon_sweep':sweep,'max_path_derivative_error_m_s':float(err.max()),'largest_error_muscle':mus[int(err.argmax())].getName(),'generalized_power_w':float(power),'path_contraction_power_w':float(expected),'power_difference_w':float(abs(power-expected)),'scope':'Synthetic small tangent test velocity and test force, not actual athlete effort.'})
    (OUT/'formula_checks.json').write_text(json.dumps({'reports':reports},indent=2));print('states',len(reports),'max_derivative_error',max(x['max_path_derivative_error_m_s'] for x in reports),'max_power_difference',max(x['power_difference_w'] for x in reports))

def wrapping_control():
    original=ROOT/'artifacts/shoulder-load-audit/mass_corrected.osim';tree=E.parse(original)
    for el in tree.findall('.//WrapObjectSet/objects/*/active'):el.text='false'
    diagnostic=OUT/'wrapping_disabled_diagnostic.osim';tree.write(diagnostic,encoding='utf-8',xml_declaration=True)
    z=np.load(ROOT/'artifacts/wrist-muscle/3034_4_ik.npz');report=[]
    for label,path in [('wrapped',original),('wrapping_disabled_diagnostic',diagnostic)]:
        m=o.Model(str(path));s=m.initSystem();cs=m.getCoordinateSet();mat=m.getMatterSubsystem();nu=s.getNU()
        for j,v in enumerate(z['q'][28]):cs.get(j).setValue(s,float(v),False)
        m.realizePosition(s);Q=arr(s.getQ());G=[];u=np.linspace(-.1,.1,nu)
        for k in range(nu):
            g=o.Vector();mat.multiplyByG(s,ov(np.eye(nu)[k]),g);G.append(arr(g))
        G=np.array(G).T;u-=G.T@np.linalg.lstsq(G@G.T,G@u,rcond=None)[0];s.setU(ov(u));m.realizeVelocity(s)
        mus=m.getMuscles().get('Subscapularis_I');rate=mus.getGeometryPath().getLengtheningSpeed(s);dq=o.Vector();mat.multiplyByN(s,False,ov(u),dq);dq=arr(dq)
        point=o.Vec3(.01,.01,.01);body=m.getBodySet().get('humerus');vel=arr(body.findStationVelocityInGround(s,point));lengths=[];positions=[]
        for sign in [-1.,1.]:
            s.setQ(ov(Q+sign*1e-4*dq));m.realizePosition(s);lengths.append(mus.getLength(s));positions.append(arr(body.findStationLocationInGround(s,point)))
        report.append({'condition':label,'path_speed_m_s':float(rate),'finite_difference_speed_m_s':float((lengths[1]-lengths[0])/2e-4),'body_station_velocity_error_m_s':float(np.linalg.norm((positions[1]-positions[0])/2e-4-vel)),'constraint_velocity_norm':float(np.linalg.norm(G@u))})
    (OUT/'wrapping_control.json').write_text(json.dumps({'scope':'One-state numerical control; disabling wraps is NOT an anatomical correction and is not adopted.','rows':report},indent=2));print(report)

if __name__=='__main__':main();wrapping_control()
