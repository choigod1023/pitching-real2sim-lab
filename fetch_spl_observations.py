"""Fetch a fixed, outcome-independent 30-trial SPL auxiliary sample."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT='a3f9cffbde917b1e1747cedd6ec25dfab18c6051'
BASE='https://raw.githubusercontent.com/Sport-Performance-Lab/SPL-Open-Data/'+COMMIT+'/'


def main():
    root=Path('artifacts/spl');root.mkdir(parents=True,exist_ok=True)
    paths=[f'basketball/freethrow/data/2025-12-18/P{p:04d}/BB_FT_P{p:04d}_T{t:04d}.json'
           for p in range(1,6) for t in range(1,7)]
    selection={'commit':COMMIT,'rule':'2025-12-18 five ID sorted participants, first six trial IDs each; no outcome selection','paths':paths}
    (root/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    existing=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else []
    hashes={row['path']:row['sha256'] for row in existing}
    def fetch(path):
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and path in hashes:
            payload=target.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=hashes[path]:raise ValueError('Cached file changed')
        else:
            payload=urllib.request.urlopen(BASE+path,timeout=30).read();json.loads(payload);target.write_bytes(payload)
        return {'path':path,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:manifest=list(pool.map(fetch,paths))
    for name in ['README.md','LICENSE','basketball/freethrow/README.md']:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(urllib.request.urlopen(BASE+name,timeout=30).read())
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Verified',len(manifest),'files from pinned SPL commit; retain CC BY-NC-SA 4.0 attribution.')


if __name__=='__main__':main()
