import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for n in ("Malgun Gothic", "NanumGothic"):
    if any(n == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = n; break
plt.rcParams["axes.unicode_minus"] = False
d = json.load(open("offset_cv_results.json"))
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
names = ["3034_1", "3034_2", "3034_3", "3034_4", "3034_5"]
r = d["fc20_raw"]; radar = [130.196, 131.001, 129.230, 128.426, 128.587]
e_cal = [r["heldout_err_kmh"][n] for n in names]
e_mean = [np.mean([x for j, x in enumerate(radar) if j != i]) - radar[i] for i in range(5)]
x = np.arange(5); w = .38
ax[0].bar(x - w/2, e_cal, w, color="#1a73e8", label="오프셋 보정 모델 (평균 절대 오차 %.2f)" % r["MAE_calibrated"])
ax[0].bar(x + w/2, e_mean, w, color="#bdc1c6", label="나머지 4투구 레이더 평균 (평균 절대 오차 %.2f)" % r["MAE_train_mean_radar"])
ax[0].axhline(0, c="k", lw=.8); ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].legend(fontsize=9)
ax[0].set_title("A. 보정에 쓰지 않은 투구의 구속 예측 오차 (km/h, 20Hz)\n보정 오프셋 20.4~21.0cm로 안정적이지만 평균 예측보다 나쁨")
keys = ["fc13_raw", "fc20_raw", "fc30_raw", "fc50_raw"]
ax[1].plot(range(4), [d[k]["MAE_calibrated"] for k in keys], "-o", c="#1a73e8", label="오프셋 보정 모델")
ax[1].plot(range(4), [d[k]["MAE_train_mean_radar"] for k in keys], "-o", c="#80868b", label="레이더 평균 예측")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["13Hz", "20Hz", "30Hz", "50Hz"]); ax[1].set_ylim(0, 6); ax[1].legend()
ax[1].set_title("B. 필터에 따른 평균 절대 오차 (km/h): 모든 설정에서 평균 예측이 더 낫다"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("offset_cv.png", dpi=105)
