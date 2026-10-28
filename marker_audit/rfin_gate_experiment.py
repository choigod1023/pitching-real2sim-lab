"""RFIN 가중치 제외(게이트) 비교. 사전 지정 규칙, 결과로 규칙을 고르지 않는다.

변형
  V0  현재 방식: 원 RFIN을 20Hz 양방향 필터
  G0  BR 이후 RFIN을 전완/손목 기준틀 좌표에서 마지막 값으로 고정한 뒤 필터
  G50 BR 50ms 전부터 고정한 뒤 필터 (게이트 위치 민감도)
손 x축은 손목 중점에서 RFIN을 향하는 방향(ball_release.hand_frame과 같은 정의).
공 중심 가정은 손목 중점에서 x축으로 0.15m(기존 가정을 그대로 유지).
"""
import sys, glob, json, numpy as np, ezc3d
from scipy.signal import butter, sosfiltfilt

root = sys.argv[1]
BR = {"002": 1.6, "005": 1.5583, "006": 1.4583, "008": 1.4528, "011": 1.4139}
NAME = {"002": "3034_1", "005": "3034_2", "006": "3034_3", "008": "3034_4", "011": "3034_5"}
R = 360.0
SOS = butter(4, 20, fs=R, output="sos")
filt = lambda a: sosfiltfilt(SOS, a, axis=0)

def local_frame(P):
    o = P["RWRB"]; a = P["RWRA"] - P["RWRB"]; b = P["RFRM"] - P["RWRB"]
    e1 = a / np.linalg.norm(a, axis=1)[:, None]
    e3 = np.cross(e1, b); e3 /= np.linalg.norm(e3, axis=1)[:, None]
    e2 = np.cross(e3, e1)
    return o, np.stack([e1, e2, e3], axis=2)  # axes in columns

def hand_x(fin, wra, wrb):
    x = fin - (wra + wrb) / 2
    return x / np.linalg.norm(x, axis=1)[:, None]

out = {}
for f in sorted(glob.glob(root + "/*_FF_*.c3d")):
    k = f.split("_207_")[1][:3]
    c = ezc3d.c3d(f)
    L = [l.strip() for l in c["parameters"]["POINT"]["LABELS"]["value"]]
    X = c["data"]["points"][:3].astype(float)
    P = {m: X[:, L.index(m)].T for m in ("RFRM", "RWRA", "RWRB", "RFIN")}
    n = len(P["RFIN"]); t = np.arange(n) / R
    o, E = local_frame(P)
    loc = np.einsum("nij,ni->nj", E, P["RFIN"] - o)  # RFIN in local coords (raw)
    ibr = int(round(BR[k] * R))
    res = {}
    def variant(gate_idx):
        l2 = loc.copy()
        if gate_idx is not None:
            l2[gate_idx:] = loc[gate_idx]
        fin = o + np.einsum("nij,nj->ni", E, l2)
        fin = fin  # RFIN 재구성(좌표틀과 함께 움직임)
        return fin
    for name, g in (("V0", None), ("G0", ibr), ("G50", ibr - int(0.05 * R))):
        fin_raw = variant(g)
        P_f = {m: filt(P[m]) for m in ("RWRA", "RWRB")}
        fin_f = filt(fin_raw)
        x = hand_x(fin_f, P_f["RWRA"], P_f["RWRB"])
        mid = (P_f["RWRA"] + P_f["RWRB"]) / 2
        ball = mid + 0.15 * x
        v = np.gradient(ball, 1 / R, axis=0, edge_order=2)
        sp = np.linalg.norm(v, axis=1) * 3.6
        dx = np.gradient(x, 1 / R, axis=0, edge_order=2)
        w = np.degrees(np.linalg.norm(np.cross(x, dx), axis=1))
        res[name] = dict(x=x, speed=sp, w=w, fin=fin_f)
    d = lambda a, b: float(np.degrees(np.arccos(np.clip((a * b).sum(), -1, 1))))
    row = {}
    for name in ("V0", "G0", "G50"):
        row[name] = dict(speed_BR=float(res[name]["speed"][ibr]),
                         x_ang_speed_BR=float(res[name]["w"][ibr]))
    for name in ("G0", "G50"):
        ang = [d(res["V0"]["x"][i], res[name]["x"][i]) for i in range(ibr - 18, ibr + 19)]
        row[name]["x_dir_diff_deg_at_BR"] = float(d(res["V0"]["x"][ibr], res[name]["x"][ibr]))
        row[name]["x_dir_diff_deg_max_BR_pm50ms"] = float(max(ang))
        row[name]["speed_diff_kmh_at_BR"] = row[name]["speed_BR"] - row["V0"]["speed_BR"]
        row[name]["speed_diff_kmh_max_pm50ms"] = float(np.max(np.abs(res[name]["speed"][ibr - 18:ibr + 19] - res["V0"]["speed"][ibr - 18:ibr + 19])))
    # 필터가 BR 이후 RFIN 이동을 BR 시점으로 끌어오는 정도: 원 RFIN vs 필터된 RFIN 위치 차
    raw_fin = P["RFIN"][ibr]; fil_fin = res["V0"]["fin"][ibr]
    row["filter_smear_mm_at_BR"] = float(np.linalg.norm(raw_fin - fil_fin) * 1000)
    out[NAME[k]] = row
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open("rfin_gate_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
