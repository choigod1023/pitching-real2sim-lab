import glob, sys, numpy as np, ezc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for n in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(n == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = n
        break
plt.rcParams["axes.unicode_minus"] = False

root, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(root + "/000002_003034_73_207_0*_FF_*.c3d"))
BR = {"002": 1.6, "005": 1.5583, "006": 1.4583, "008": 1.4528, "011": 1.4139}
NAME = {"002": "3034_1", "005": "3034_2", "006": "3034_3", "008": "3034_4", "011": "3034_5"}
ARM = ["RSHO", "RUPA", "RELB", "RMELB", "RFRM", "RWRA", "RWRB", "RFIN"]
ARM_LINES = [("RSHO", "RUPA"), ("RUPA", "RELB"), ("RELB", "RFRM"), ("RFRM", "RWRA"),
             ("RWRA", "RWRB"), ("RWRB", "RFIN")]

def load(f):
    c = ezc3d.c3d(f)
    L = [l.strip() for l in c["parameters"]["POINT"]["LABELS"]["value"]]
    return L, c["data"]["points"][:3], c["parameters"]["POINT"]["RATE"]["value"][0]

data = {}
for f in files:
    k = f.split("_207_")[1][:3]
    data[k] = load(f)

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.25, 1.0])

# 행 1: 3034_5(문제 투구) 세 시점의 전신 마커 + 팔 강조
L, x, r = data["011"]
t_br = BR["011"]
times = [("공 놓기 0.3초 전", t_br - 0.3), ("공 놓기 (BR)", t_br), ("공 놓기 0.3초 후", t_br + 0.3)]
cent = np.nanmean(x[:, :, int(t_br * r)], axis=1)
for i, (title, t) in enumerate(times):
    ax = fig.add_subplot(gs[0, i])
    fr = min(int(round(t * r)), x.shape[2] - 1)
    p = x[:, :, fr] - cent[:, None]
    ax.scatter(p[1], p[2], s=22, c="#9aa0a6")
    for a_, b_ in ARM_LINES:
        ia, ib = L.index(a_), L.index(b_)
        ax.plot([p[1, ia], p[1, ib]], [p[2, ia], p[2, ib]], c="#d93025", lw=2.5)
    cols = {"RWRB": "#1a73e8", "RFIN": "#f9ab00"}
    for m in ARM:
        im = L.index(m)
        ax.scatter(p[1, im], p[2, im], s=70, c=cols.get(m, "#d93025"), zorder=3)
        if m in ("RSHO", "RELB", "RWRB", "RFIN"):
            ax.text(p[1, im] + 0.04, p[2, im] + 0.04, m, fontsize=10)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.3, 0.9); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"3034_5  {title}  (t={t:.2f}초)")
    ax.set_xlabel("y (m, 옆에서 본 모습)"); ax.set_ylabel("z (m, 높이)")

# 행 2: 손 확대 (손목B 기준 상대 위치), 투구 3034_5 / 3034_3
def hand_panel(ax, key, tt, title):
    L, x, r = data[key]
    fr = min(int(round(tt * r)), x.shape[2] - 1)
    ref = x[:, L.index("RWRB"), fr]
    for m, c in (("RFRM", "#d93025"), ("RWRA", "#d93025"), ("RWRB", "#1a73e8"), ("RFIN", "#f9ab00")):
        pp = (x[:, L.index(m), fr] - ref) * 1000
        ax.scatter(pp[1], pp[2], s=80, c=c)
        ax.text(pp[1] + 3, pp[2] + 3, m, fontsize=10)
    for a, b in (("RFRM", "RWRA"), ("RWRA", "RWRB"), ("RWRB", "RFIN")):
        pa = (x[:, L.index(a), fr] - ref) * 1000; pb = (x[:, L.index(b), fr] - ref) * 1000
        ax.plot([pa[1], pb[1]], [pa[2], pb[2]], c="#d93025" if a != "RWRB" else "#1a73e8", lw=2)
    d = np.linalg.norm(x[:, L.index("RWRB"), fr] - x[:, L.index("RFIN"), fr]) * 1000
    ax.set_title(f"{title}\nRWRB-RFIN 거리 {d:.0f}mm")
    ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("y (mm, 손목B 기준)"); ax.set_ylabel("z (mm)")

hand_panel(fig.add_subplot(gs[1, 0]), "008", BR["008"], "정상 예: 3034_4 공 놓기 시점 손 확대")
hand_panel(fig.add_subplot(gs[1, 1]), "011", BR["011"], "3034_5 공 놓기 시점 손 확대")
hand_panel(fig.add_subplot(gs[1, 2]), "011", 1.767, "3034_5 거리가 가장 벌어진 시점(1.767초)")

# 행 3: 5개 투구 RWRB-RFIN 거리 시계열
ax = fig.add_subplot(gs[2, :])
colors = {"002": "#5f6368", "005": "#80868b", "008": "#34a853", "006": "#f9ab00", "011": "#d93025"}
for k in ("002", "005", "008", "006", "011"):
    L, x, r = data[k]
    d = np.linalg.norm(x[:, L.index("RWRB")] - x[:, L.index("RFIN")], axis=0) * 1000
    t = np.arange(len(d)) / r - BR[k]
    ax.plot(t, d, c=colors[k], lw=2.2 if k in ("006", "011") else 1.3, label=NAME[k])
ax.axhline(105.0, c="k", ls="--", lw=1)
ax.text(-1.45, 107, "정적 보정 105.0mm", fontsize=9)
ax.axvline(0, c="k", lw=1)
ax.text(0.01, 150, "공 놓기(BR)", fontsize=9)
ax.set_xlim(-1.5, 1.0)
ax.set_xlabel("공 놓기 기준 시간 (초)"); ax.set_ylabel("RWRB-RFIN 거리 (mm)")
ax.set_title("손목B 마커와 손가락 마커 사이 거리: 3034_3, 3034_5는 공 놓기 직후부터 크게 벌어짐")
ax.legend(ncol=5, loc="upper left"); ax.grid(alpha=.3)

fig.suptitle("3034 선수 투구 C3D 마커 점검  (회색 점: 전신 45개 마커 / 빨강: 오른팔 / 파랑: RWRB 손목 / 노랑: RFIN 손가락)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out, dpi=110)
print("saved", out)
