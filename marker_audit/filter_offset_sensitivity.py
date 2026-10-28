"""BR 시점 공 중심 대리 속도의 필터 차단 주파수, 손 오프셋 민감도. 어떤 값도 레이더에 맞춰 고르지 않는다."""
import sys, glob, json, numpy as np, ezc3d
from scipy.signal import butter, sosfiltfilt
root = sys.argv[1]
BR = {"002": 1.6, "005": 1.5583, "006": 1.4583, "008": 1.4528, "011": 1.4139}
NAME = {"002": "3034_1", "005": "3034_2", "006": "3034_3", "008": "3034_4", "011": "3034_5"}
RADAR = {"3034_1": 130.196, "3034_2": 131.001, "3034_3": 129.230, "3034_4": 128.426, "3034_5": 128.587}
R = 360.0
def run(P, ibr, fc, off):
    f = (lambda a: a) if fc is None else (lambda a: sosfiltfilt(butter(4, fc, fs=R, output="sos"), a, axis=0))
    fin, a, b = f(P["RFIN"]), f(P["RWRA"]), f(P["RWRB"])
    mid = (a + b) / 2; x = fin - mid; x /= np.linalg.norm(x, axis=1)[:, None]
    ball = mid + off * x
    return float(np.linalg.norm(np.gradient(ball, 1 / R, axis=0, edge_order=2)[ibr]) * 3.6)
res = {}
for fl in sorted(glob.glob(root + "/*_FF_*.c3d")):
    k = fl.split("_207_")[1][:3]; c = ezc3d.c3d(fl)
    L = [l.strip() for l in c["parameters"]["POINT"]["LABELS"]["value"]]
    P = {m: c["data"]["points"][:3, L.index(m)].T.astype(float) for m in ("RFIN", "RWRA", "RWRB")}
    ibr = int(round(BR[k] * R)); n = NAME[k]
    res[n] = dict(
        radar=RADAR[n],
        cutoff_at_off15={str(fc): run(P, ibr, fc, .15) for fc in (13, 20, 30, 50, 80, None)},
        offset_at_20Hz={str(o): run(P, ibr, 20, o) for o in (0.0, .05, .10, .15, .20, .25)},
        wrist_mid_speed_20Hz=run(P, ibr, 20, 0.0))
print(json.dumps(res, indent=1))
json.dump(res, open("filter_offset_results.json", "w"), indent=1)
