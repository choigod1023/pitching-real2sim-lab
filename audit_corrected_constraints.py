"""Recheck differentiated constraint consistency after mass correction."""
import json
from pathlib import Path
import opensim as o
import wrist_residual_audit as audit
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research-evidence-audit'

def main():
    OUT.mkdir(parents=True,exist_ok=True);audit.OUT=OUT;audit.INPUT=ROOT/'artifacts/wrist-muscle';o.Logger.setLevelString('error');rows=[]
    for p in [f'3034_{i}' for i in range(1,6)]:
        for mode in ['original','consistent']:
            rows.append(audit.run(ROOT/'artifacts/shoulder-load-audit/mass_corrected.osim',p,mode));(OUT/'constraint_checks.json').write_text(json.dumps({'rows':rows},indent=2))

if __name__=='__main__':main()
