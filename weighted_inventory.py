from pathlib import Path
import zipfile,ezc3d,json,hashlib
p=Path('artifacts/external/Motion Data.zip');z=zipfile.ZipFile(p);names=[n for n in z.namelist() if n.lower().endswith('.c3d')];out=[]
tmp=Path('artifacts/external/inspect.c3d')
for name in names:
 tmp.write_bytes(z.read(name));c=ezc3d.c3d(str(tmp));point=c['parameters']['POINT'];event=c['parameters'].get('EVENT',{})
 out.append({'file':name,'labels':point['LABELS']['value'],'event_used':int(event.get('USED',{}).get('value',[0])[0]),'event_labels':event.get('LABELS',{}).get('value',[]),'header_labels':[x for x in c['header']['events']['events_label'] if x],'rate':float(point['RATE']['value'][0])})
summary={'zip_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'c3d_count':len(names),'subjects':len(set(n.split('/')[1] for n in names)),'static_count':sum('model' in n.lower() for n in names),'labels':sorted(set(x for r in out for x in r['labels'])),'events_used_files':sum(r['event_used']>0 for r in out),'header_named_files':sum(bool(r['header_labels']) for r in out),'rates':sorted(set(r['rate'] for r in out))}
Path('artifacts/external/weighted-inventory.json').write_text(json.dumps({'summary':summary,'files':out},indent=2),encoding='utf-8');print(json.dumps(summary))
