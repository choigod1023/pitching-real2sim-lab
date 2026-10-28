"""OFFSET_CV_PROTOCOL.md의 사전 규칙대로 실행. 규칙은 결과를 보고 바꾸지 않았다."""
import sys, glob, json, itertools, numpy as np, ezc3d
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
root = sys.argv[1]
BR = {"002": 1.6, "005": 1.5583, "006": 1.4583, "008": 1.4528, "011": 1.4139}
NAME = {"002": "3034_1", "005": "3034_2", "006": "3034_3", "008": "3034_4", "011": "3034_5"}
RADAR = {"3034_1": 130.196, "3034_2": 131.001, "3034_3": 129.230, "3034_4": 128.426, "3034_5": 128.587}
R = 360.0
raw = {}
for f in sorted(glob.glob(root + "/*_FF_*.c3d")):
    k = f.split("_207_")[1][:3]; c = ezc3d.c3d(f)
    L = [l.strip() for l in c["parameters"]["POINT"]["LABELS"]["value"]]
    P = {m: c["data"]["points"][:3, L.index(m)].T.astype(float) for m in ("RFRM", "RWRA", "RWRB", "RFIN")}
    raw[NAME[k]] = (P, int(round(BR[k] * R)))

def local_frame(P):
    o = P["RWRB"]; a = P["RWRA"] - P["RWRB"]; b = P["RFRM"] - P["RWRB"]
    e1 = a / np.linalg.norm(a, axis=1)[:, None]
    e3 = np.cross(e1, b); e3 /= np.linalg.norm(e3, axis=1)[:, None]
    return o, np.stack([e1, np.cross(e3, e1), e3], axis=2)

def prep(P, ibr, fc, gate):
    fin = P["RFIN"].copy()
    if gate:
        o, E = local_frame(P); loc = np.einsum("nij,ni->nj", E, fin - o); loc[ibr:] = loc[ibr]
        fin = o + np.einsum("nij,nj->ni", E, loc)
    sos = butter(4, fc, fs=R, output="sos"); f = lambda a: sosfiltfilt(sos, a, axis=0)
    fin, a, b = f(fin), f(P["RWRA"]), f(P["RWRB"])
    mid = (a + b) / 2; x = fin - mid; x /= np.linalg.norm(x, axis=1)[:, None]
    return mid, x

def speed(mid, x, ibr, off):
    return float(np.linalg.norm(np.gradient(mid + off * x, 1 / R, axis=0, edge_order=2)[ibr]) * 3.6)

out = {}
names = list(RADAR)
for fc, gate in itertools.product((13, 20, 30, 50), (False, True)):
    cache = {n: prep(raw[n][0], raw[n][1], fc, gate) for n in names}
    spd = lambda n, o: speed(*cache[n], raw[n][1], o)
    fitted, pred, err_cal, err_mean, err_15 = {}, {}, {}, {}, {}
    for held in names:
        train = [n for n in names if n != held]
        res = minimize_scalar(lambda o: sum((spd(n, o) - RADAR[n]) ** 2 for n in train), bounds=(0, .4), method="bounded")
        fitted[held] = float(res.x); pred[held] = spd(held, res.x)
        err_cal[held] = pred[held] - RADAR[held]
        err_mean[held] = float(np.mean([RADAR[n] for n in train]) - RADAR[held])
        err_15[held] = spd(held, .15) - RADAR[held]
    s15 = [spd(n, .15) for n in names]; rr = [RADAR[n] for n in names]
    rho = float(spearmanr(s15, rr).statistic)
    mae = lambda d: float(np.mean(np.abs(list(d.values()))))
    out[f"fc{fc}_{'G0' if gate else 'raw'}"] = dict(
        fitted_offset_cm={k: round(v * 100, 1) for k, v in fitted.items()},
        heldout_err_kmh={k: round(v, 2) for k, v in err_cal.items()},
        MAE_calibrated=round(mae(err_cal), 2), MAE_train_mean_radar=round(mae(err_mean), 2), MAE_uncalibrated_15cm=round(mae(err_15), 2),
        spearman_rank_vs_radar=round(rho, 2))
for k, v in out.items():
    print(k, "| 오프셋(cm)", list(v["fitted_offset_cm"].values()), "| 평가오차", list(v["heldout_err_kmh"].values()),
          "| MAE 보정 %.2f / 평균예측 %.2f / 15cm %.2f | 순위상관 %.2f" % (v["MAE_calibrated"], v["MAE_train_mean_radar"], v["MAE_uncalibrated_15cm"], v["spearman_rank_vs_radar"]))
json.dump(out, open("offset_cv_results.json", "w"), indent=1)
