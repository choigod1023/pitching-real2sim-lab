import json,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from acceleration_follow import simulate


def check(args):
    return simulate(*args,dt=1e-5)[0]


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    args=[(axis,a,follow) for axis,a in [(0,250.),(2,50.),(2,100.)] for follow in [False,True]]
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(check,args))
    (root/'acceleration_resolution.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
