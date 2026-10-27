"""Restore the intended source mass preservation and consistent scaled inertia."""
import json,copy
import xml.etree.ElementTree as E
import numpy as np
import opensim as o
import shoulder_load_audit as audit
from shoulder_load_audit import ROOT,OUT,MODEL

def build():
    OUT.mkdir(parents=True,exist_ok=True);o.Logger.setLevelString('error')
    manifest=json.loads((ROOT/'artifacts/anatomical-shoulder/source_manifest.json').read_text())
    m=o.Model(str(ROOT/'artifacts/anatomical-shoulder/TSM_subject_noWeight.osim'));s=m.initSystem();ss=o.ScaleSet()
    before={m.getBodySet().get(i).getName():m.getBodySet().get(i).getMass() for i in range(m.getBodySet().getSize())}
    for name in before:
        f=manifest['upper_scale'] if name=='humerus' else manifest['forearm_scale'] if name in ['ulna','radius','hand'] else manifest['torso_scale']
        sc=o.Scale();sc.setSegmentName(name);sc.setScaleFactors(o.Vec3(float(f)));sc.setApply(True);ss.cloneAndAppend(sc)
    m.scale(s,ss,True);m.printToXML(str(OUT/'mass_preserved_source.osim'))
    src=E.parse(OUT/'mass_preserved_source.osim');dst=E.parse(MODEL);report=[]
    for body in src.findall('.//BodySet/objects/*'):
        name=body.get('name');old=dst.find(f".//BodySet/objects/*[@name='{name}']")
        if not np.allclose(np.fromstring(body.findtext('mass_center'),sep=' '),np.fromstring(old.findtext('mass_center'),sep=' '),atol=1e-10):raise ValueError('COM geometry differs')
        newmass=float(body.findtext('mass'))
        if not np.isclose(newmass,before[name],rtol=1e-12):raise ValueError('Source mass not preserved')
        report.append({'body':name,'before_kg':float(old.findtext('mass')),'after_kg':newmass,'source_kg':before[name]})
        for field in ['mass','inertia']:old.find(field).text=body.findtext(field)
    path=OUT/'mass_corrected.osim';dst.write(path,encoding='utf-8',xml_declaration=True)
    (OUT/'mass_correction_manifest.json').write_text(json.dumps({'bodies':report,'scope':'Fix previously intended source mass preservation; these generic masses are NOT athlete-specific measurements. Geometry, trajectory, muscle Fmax and paths unchanged.'},indent=2))
    return path

def main():
    path=build();audit.MODEL=path;rows=[]
    for p in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        rows.append(audit.run(p));(OUT/'corrected_results.json').write_text(json.dumps({'reports':rows},indent=2))

if __name__=='__main__':main()
