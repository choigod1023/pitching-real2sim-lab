"""Read-only source-index XML extraction; no spreadsheet authoring or formulas executed."""
from pathlib import Path,PurePosixPath
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as E

SOURCE='https://docs.google.com/spreadsheets/d/1NhpF8DnfBdio_xsU7B44KNuHghtePDp-d3juvCYNm9Q/edit'
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_cells(path):
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in x.findall('.//m:t',NS)) for x in E.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',NS)]
        rels={x.attrib['Id']:x.attrib['Target'] for x in E.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        output={}
        for sheet in E.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',NS):
            rid=sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'];target=rels[rid]
            member=target.lstrip('/') if target.startswith('/') else str(PurePosixPath('xl')/target)
            cells={}
            for c in E.fromstring(z.read(member)).findall('.//m:c',NS):
                v=c.find('m:v',NS)
                if c.attrib.get('t')=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//m:t',NS))
                elif v is None:continue
                elif c.attrib.get('t')=='s':value=strings[int(v.text)]
                else:value=v.text
                cells[c.attrib['r']]=value
            output[sheet.attrib['name']]=cells
        return output


def speed_label(text):
    match=re.search(r'(\d+(?:\.\d+)?)\s*MPH',text,re.I)
    if not match:raise ValueError('No explicit MPH source label')
    mph=float(match.group(1));return dict(mph=mph,kmh=mph*1.609344,kind='source label; not instrument accuracy verification')


def require_same_trial(left,right):
    if left!=right:raise ValueError('Different trial identity; no cross-system pose substitution')


def run():
    root=Path(__file__).resolve().parent;p=root/'artifacts/paired-source/obp-cv-source-index.xlsx';s=read_cells(p)
    values={}
    for name,cell,fpscell in [('Prime Color - Shot List','H17','D17'),('Edgertronic - Shot List','H16','D16'),('iPhone - Shot List','H13','D13')]:
        values[name]=dict(cell=cell,text=s[name][cell],speed_label=speed_label(s[name][cell]),
                          capture_rate_cell=fpscell,declared_capture_fps=float(s[name][fpscell]))
    report=dict(source_url=SOURCE,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
       shot_lists=values,
       calibration_definition=dict(cube_source_cell='OBP-CV Readme!B14',cube_source_text=s['OBP-CV Readme']['B14'],
         outer_red_joint_spacing_mm=27.5*25.4,midpoint_spacing_mm=27.5*25.4/2,
         checkerboard_cell='OBP-CV Readme!B17',checkerboard_text=s['OBP-CV Readme']['B17']),
       synchronization_cell='OBP-CV Readme!B20',synchronization_text=s['OBP-CV Readme']['B20'],
       timing_caveat_cell='OBP-CV Readme!B26',timing_caveat=s['OBP-CV Readme']['B26'],
       edgertronic_filename_labels=[dict(file_label_mph=v,file_label_kmh=v*1.609344,
              measured_speed_kmh=None,worksheet_trial_correspondence='unresolved') for v in [67,70]],
       files_with_pose='Only BaseballThrow_001 was found among eight initial C3D folder entries; no new 67/70 Theia assignment established.')
    (root/'source_index_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(values,ensure_ascii=False,indent=2));return report


if __name__=='__main__':run()
