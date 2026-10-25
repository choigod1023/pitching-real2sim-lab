"""Download frozen public author files without executing third-party code."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

COMMIT='29aa44eee86eff5c1fc6f5e221cb9b16a0381f16'
SHA256='e0f6e6cc8fc40f8744810967f98853a18953a91e5b990af2207d20d6a9431e3a'


def main():
    root=Path('artifacts/external/takamido');root.mkdir(parents=True,exist_ok=True)
    archive=root/'source.zip'
    content=archive.read_bytes() if archive.exists() else urllib.request.urlopen(
        'https://codeload.github.com/takamido/Ball_speed_pred_cross_ind/zip/'+COMMIT,timeout=60).read()
    if hashlib.sha256(content).hexdigest()!=SHA256:raise ValueError('Archive checksum changed')
    archive.write_bytes(content);manifest=[]
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            parts=PurePosixPath(name).parts
            if len(parts)<2 or name.endswith('/'):continue
            rel=PurePosixPath(*parts[1:])
            if '..' in rel.parts or rel.is_absolute():raise ValueError('Unsafe archive member')
            path=str(rel)
            if not (path.startswith(('MATLAB_code/prediction/base_models/','ball_speed_pred_colab/data/')) or
                    path in ['README.md','MATLAB_code/Analysis1.m','ball_speed_pred_colab/program/base_model_inter.ipynb']):continue
            payload=z.read(name);dest=root/path
            if not dest.resolve().is_relative_to(root.resolve()):raise ValueError('Outside destination')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
            manifest.append({'path':path,'sha256':hashlib.sha256(payload).hexdigest()})
    (root/'download-manifest.json').write_text(json.dumps({'commit':COMMIT,'archive_sha256':SHA256,'files':manifest},indent=2),encoding='utf-8')
    print('Verified frozen archive; saved',len(manifest),'files. No external code executed.')


if __name__=='__main__':main()
