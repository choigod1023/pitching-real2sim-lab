"""Compare humeral elbow-frame registration and missing donor elbow muscles.

Registration assumes matching body-frame axes and aligns elbow centers. It is
not patient-specific anatomical calibration. Source Fmax is never increased.
"""
import copy,json,hashlib
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import opensim as o
from wrist_residual_audit import INPUT,OUT as OLDOUT,run
import wrist_residual_audit as audit
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/elbow-registration-audit'
WRIST=['ECRL','ECRB','ECU','FCR','FCU','PL']
ELBOW=['TRIlat','TRImed','ANC','BRA','BRD','SUP','PT','PQ']

def xyz(text):return np.fromstring(text,sep=' ')
def shift(el,delta):el.text=' '.join(f'{x:.17g}' for x in xyz(el.text)+delta)
def elbow_anchor(tree):
    j=tree.find(".//JointSet/objects/*[@name='elbow']")
    return xyz(next(f.findtext('translation') for f in j.findall('frames/PhysicalOffsetFrame') if f.findtext('socket_parent')=='/bodyset/humerus'))
def axes(tree):
    j=tree.find(".//JointSet/objects/*[@name='elbow']")
    return xyz(next(x.findtext('axis') for x in j.findall('.//TransformAxis') if x.findtext('coordinates')=='elbow_flexion'))
def build():
    OUT.mkdir(parents=True,exist_ok=True)
    donor=E.parse(INPUT/'donor_scaled.osim');base=E.parse(INPUT/'scaled_source.osim')
    delta=elbow_anchor(base)-elbow_anchor(donor)
    angle=np.rad2deg(np.arccos(np.clip(axes(base)@axes(donor)/np.linalg.norm(axes(base))/np.linalg.norm(axes(donor)),-1,1)))
    if angle>1e-5:raise ValueError('Elbow axes are not aligned; translation-only assumption invalid')
    for name in WRIST:
        mus=base.find(f".//ForceSet/objects/*[@name='{name}']")
        for p in mus.findall('.//PathPointSet/objects/*'):
            if p.findtext('socket_parent_frame')=='/bodyset/humerus':
                if p.find('location') is None:raise ValueError('Moving humeral path point needs explicit translation')
                shift(p.find('location'),delta)
    # This object was imported with wrist muscles and only used by them.
    wrap=base.find(".//BodySet/objects/*[@name='humerus']/WrapObjectSet/objects/*[@name='Elbow_PT_ECRL']")
    shift(wrap.find('translation'),delta)
    base.write(OUT/'registered_wrist.osim',encoding='utf-8',xml_declaration=True)
    extended=copy.deepcopy(base);needed=set()
    for name in ELBOW:
        m=copy.deepcopy(donor.find(f".//ForceSet/objects/*[@name='{name}']"))
        if m is None:raise ValueError('Missing donor muscle '+name)
        for p in m.findall('.//PathPointSet/objects/*'):
            if p.findtext('socket_parent_frame')=='/bodyset/humerus':
                if p.find('location') is None:raise ValueError('Moving humeral path point')
                shift(p.find('location'),delta)
        for w in m.findall('.//wrap_object'):
            needed.add(w.text);w.text='elbow_donor_'+w.text
        extended.find('.//ForceSet/objects').append(m)
    for body in donor.findall('.//BodySet/objects/*'):
        dest=extended.find(f".//BodySet/objects/*[@name='{body.get('name')}']")
        if dest is None:continue
        wraps=dest.find('WrapObjectSet/objects')
        if wraps is None:
            ws=E.SubElement(dest,'WrapObjectSet');wraps=E.SubElement(ws,'objects')
        for w in body.findall('WrapObjectSet/objects/*'):
            if w.get('name') in needed:
                cp=copy.deepcopy(w);cp.set('name','elbow_donor_'+w.get('name'))
                if body.get('name')=='humerus':shift(cp.find('translation'),delta)
                wraps.append(cp)
    extended.write(OUT/'registered_plus_elbow.osim',encoding='utf-8',xml_declaration=True)
    info={'humeral_elbow_anchor_shift_m':delta.tolist(),'shift_norm_mm':float(np.linalg.norm(delta)*1000),'source_axes_angle_deg':float(angle),'wrist_names':WRIST,'added_elbow_muscles':ELBOW,'note':'Elbow anchor alignment is a model-to-model registration assumption, not measured individual attachment sites. Added muscles change modeled force capacity; NOT a same-capacity improvement comparison.'}
    (OUT/'manifest.json').write_text(json.dumps(info,indent=2));print(info,flush=True)

def main():
    o.Logger.setLevelString('error');build();audit.OUT=OUT
    reports=[]
    for p in ['3034_1','3034_2','3034_3','3034_4','3034_5']:
        for file in [INPUT/'scaled_source.osim',OUT/'registered_wrist.osim',OUT/'registered_plus_elbow.osim']:
            result=run(file,p,'original');reports.append(result)
            (OUT/'results.json').write_text(json.dumps({'rows':reports},indent=2))

if __name__=='__main__':main()
