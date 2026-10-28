import sys, glob, numpy as np, ezc3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for n in ("Malgun Gothic", "NanumGothic"):
    if any(n == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = n; break
plt.rcParams["axes.unicode_minus"] = False
root, out = sys.argv[1], sys.argv[2]
BR = {"002": 1.6, "005": 1.5583, "006": 1.4583, "008": 1.4528, "011": 1.4139}
NAME = {"002": "3034_1", "005": "3034_2", "006": "3034_3", "008": "3034_4", "011": "3034_5"}
COL = {"002": "#5f6368", "005": "#9aa0a6", "008": "#34a853", "006": "#f9ab00", "011": "#d93025"}
D = {}
for f in sorted(glob.glob(root + "/*_FF_*.c3d")):
    k = f.split("_207_")[1][:3]; c = ezc3d.c3d(f)
    L = [l.strip() for l in c["parameters"]["POINT"]["LABELS"]["value"]]
    D[k] = (L, c["data"]["points"][:3].astype(float), 360.0)

def frame(P):  # forearm/wrist local frame from RFRM, RWRA, RWRB
    o = P["RWRB"]; a = P["RWRA"] - P["RWRB"]
    b = P["RFRM"] - P["RWRB"]
    e1 = a / np.linalg.norm(a, axis=0)
    e3 = np.cross(e1, b, axis=0); e3 /= np.linalg.norm(e3, axis=0)
    e2 = np.cross(e3, e1, axis=0)
    return o, e1, e2, e3

res = {}
for k, (L, x, r) in D.items():
    P = {m: x[:, L.index(m)] for m in L}
    o, e1, e2, e3 = frame(P)
    v = P["RFIN"] - o
    loc = np.stack([(v * e).sum(0) for e in (e1, e2, e3)]) * 1000
    t = np.arange(x.shape[2]) / r - BR[k]
    acc = np.linalg.norm(np.diff(P["RFIN"], 2, axis=1), axis=0) * r * r
    # nearest other marker to RFIN (excluding hand/wrist set)
    excl = {"RFIN", "RWRA", "RWRB", "RFRM"}
    others = [m for m in L if m not in excl]
    dn = np.stack([np.linalg.norm(P["RFIN"] - P[m], axis=0) for m in others]) * 1000
    nearest = dn.min(0); nn = [others[i] for i in dn.argmin(0)]
    dA = np.linalg.norm(P["RFIN"] - P["RWRA"], axis=0) * 1000
    dB = np.linalg.norm(P["RFIN"] - P["RWRB"], axis=0) * 1000
    dF = np.linalg.norm(P["RFIN"] - P["RFRM"], axis=0) * 1000
    res[k] = dict(t=t, loc=loc, acc=acc, nearest=nearest, nn=nn, dA=dA, dB=dB, dF=dF)
    post = t > 0
    pre = t < -0.3
    sd_pre = loc[:, pre].std(1); 
    print(NAME[k], "RFIN 국소좌표 투구전(-1.5~-0.3s) 표준편차 mm:", np.round(sd_pre, 1),
          "| BR 후 평균 mm:", np.round(loc[:, post].mean(1), 1),
          "| BR 후 가속도 p99 %.0f m/s2, 최대 %.0f" % (np.percentile(acc[post[2:]], 99), acc[post[2:]].max()),
          "| 다른 마커와 최근접 최소 %.0f mm (%s)" % (nearest.min(), nn[int(nearest.argmin())]))

fig, axs = plt.subplots(3, 2, figsize=(15, 12))
for k in ("002", "005", "008", "006", "011"):
    R = res[k]; lw = 2.2 if k in ("006", "011") else 1.2
    for i, nm in enumerate(("e1 (손목A 방향)", "e2", "e3 (전완 면의 수직)")):
        pass
    axs[0, 0].plot(R["t"], R["loc"][0], c=COL[k], lw=lw, label=NAME[k])
    axs[0, 1].plot(R["t"], R["loc"][1], c=COL[k], lw=lw)
    axs[1, 0].plot(R["t"], R["loc"][2], c=COL[k], lw=lw)
    axs[1, 1].plot(R["t"][2:], R["acc"], c=COL[k], lw=lw)
axs[0, 0].set_title("RFIN 위치: 손목 기준틀 x축 성분 (mm)"); axs[0, 1].set_title("RFIN 위치: y축 성분 (mm)")
axs[1, 0].set_title("RFIN 위치: z축 성분 (mm)"); axs[1, 1].set_title("RFIN 가속도 크기 (m/s²): 순간적 튐 확인"); axs[1, 1].set_yscale("log")
for k, a in (("006", axs[2, 0]), ("011", axs[2, 1])):
    R = res[k]
    a.plot(R["t"], R["dA"], label="RFIN-RWRA"); a.plot(R["t"], R["dB"], label="RFIN-RWRB")
    a.plot(R["t"], R["dF"], label="RFIN-RFRM"); a.plot(R["t"], R["nearest"], c="k", ls=":", label="다른 마커 중 최근접")
    a.set_title(f"{NAME[k]}: RFIN과 주변 마커 거리 (mm)"); a.legend(fontsize=8)
for a in axs.ravel():
    a.axvline(0, c="k", lw=1); a.grid(alpha=.3); a.set_xlabel("공 놓기 기준 시간 (초)"); a.set_xlim(-0.6, 0.5)
axs[0, 0].legend(ncol=2, fontsize=8)
fig.suptitle("손가락 마커(RFIN) 진단: 전완/손목 마커 3개로 만든 기준틀에서 RFIN이 어떻게 움직이는가", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=105); print("saved", out)
