import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for n in ("Malgun Gothic", "NanumGothic"):
    if any(n == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = n; break
plt.rcParams["axes.unicode_minus"] = False
g = json.load(open("rfin_gate_results.json", encoding="utf-8"))
s = json.load(open("filter_offset_results.json"))
names = list(g)
col = dict(zip(names, ["#5f6368", "#9aa0a6", "#f9ab00", "#34a853", "#d93025"]))
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
w = .38; x = np.arange(5)
ax[0, 0].bar(x - w/2, [g[n]["G0"]["speed_diff_kmh_at_BR"] for n in names], w, label="BR부터 RFIN 고정(G0)", color="#1a73e8")
ax[0, 0].bar(x + w/2, [g[n]["G50"]["speed_diff_kmh_at_BR"] for n in names], w, label="BR 50ms 전부터 고정(G50)", color="#bdc1c6")
ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(names); ax[0, 0].axhline(0, c="k", lw=.8)
ax[0, 0].set_title("A. RFIN 고정 시 BR 시점 공 중심 대리 속도 변화 (km/h)"); ax[0, 0].legend(fontsize=9)
ax[0, 1].bar(x - w/2, [g[n]["G0"]["x_dir_diff_deg_at_BR"] for n in names], w, color="#1a73e8", label="G0")
ax[0, 1].bar(x + w/2, [g[n]["G50"]["x_dir_diff_deg_at_BR"] for n in names], w, color="#bdc1c6", label="G50")
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(names)
ax[0, 1].set_title("B. 손 길이축 방향 변화 (도): G50은 실제 손목 동작까지 지워서 50도 이상 변함"); ax[0, 1].legend(fontsize=9)
for n in names:
    cut = s[n]["cutoff_at_off15"]; ks = ["13", "20", "30", "50", "80", "None"]
    ax[1, 0].plot(range(6), [cut[k] for k in ks], "-o", c=col[n], label=n)
ax[1, 0].set_xticks(range(6)); ax[1, 0].set_xticklabels(["13Hz", "20Hz\n(현재)", "30Hz", "50Hz", "80Hz", "필터없음"])
ax[1, 0].axhspan(128.4, 131.0, color="#e6f4ea", label="레이더 기록 범위(128.4~131.0)")
ax[1, 0].set_title("C. 필터 차단주파수에 따른 BR 대리 속도 (오프셋 15cm), km/h"); ax[1, 0].legend(fontsize=8, ncol=2); ax[1, 0].grid(alpha=.3)
offs = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25"]
for n in names:
    ax[1, 1].plot([float(o) * 100 for o in offs], [s[n]["offset_at_20Hz"][o] for o in offs], "-o", c=col[n], label=n)
ax[1, 1].axhspan(128.4, 131.0, color="#e6f4ea"); ax[1, 1].axvline(15, c="k", ls="--", lw=1)
ax[1, 1].text(15.5, 70, "현재 가정 15cm", fontsize=9)
ax[1, 1].set_xlabel("손목 중점에서 공 중심까지 가정한 거리 (cm)")
ax[1, 1].set_title("D. 손 오프셋 가정에 따른 BR 대리 속도 (20Hz), km/h: 5cm당 약 15km/h"); ax[1, 1].grid(alpha=.3)
fig.suptitle("RFIN 이상의 영향과 다른 가정(필터, 손 오프셋)의 영향 비교  (어떤 값도 레이더에 맞춰 고르지 않았음)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig("rfin_gate_vs_assumptions.png", dpi=105)
