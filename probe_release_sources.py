"""Read-only public source probe; no author contact, authentication, or metadata rewrite."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import urllib.request


def main():
    out = Path('artifacts/release-budget')
    out.mkdir(parents=True, exist_ok=True)
    url = 'https://api.github.com/repos/drivelineresearch/openbiomechanics/releases?per_page=20'
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
    (out/'upstream_release_snapshot.json').write_bytes(raw)
    releases = json.loads(raw)
    root = Path('../openbiomechanics')
    tracked = subprocess.check_output(['git', 'ls-files', 'computer_vision'], cwd=root, text=True).splitlines()
    sources = [root/'baseball_pitching/code/v3d/model/v6_model_hybrid_lm.mdh',
               root/'baseball_pitching/code/py/copy_anonymize_c3ds_pull_db_csvs.ipynb']
    result = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'release_url': url,
        'release_snapshot_sha256': hashlib.sha256(raw).hexdigest(),
        'releases': [{'tag': r['tag_name'], 'assets': [
            {'name': a['name'], 'size': a['size'], 'url': a['browser_download_url']}
            for a in r['assets']]} for r in releases],
        'tracked_cv_data_filenames': [s for s in tracked if s.lower().endswith(('.mp4', '.avi', '.mov', '.c3d', '.csv', '.xlsx'))],
        'local_source_hashes': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'same_pitch_direct_ball_observation_confirmed': False,
        'scope': 'Public release inventory and extraction provenance. An absent asset name is not proof that no original3034 video exists anywhere.'}
    (out/'public_source_inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Public release inventory and local provenance hashes saved.')


if __name__ == '__main__':
    main()
