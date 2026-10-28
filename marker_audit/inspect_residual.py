import sys, numpy as np, ezc3d
c = ezc3d.c3d(sys.argv[1])
m = c["data"]["meta_points"]
print("meta keys:", list(m.keys()))
for k, v in m.items():
    v = np.asarray(v); print(k, v.shape, v.dtype, "min", v.min(), "max", v.max())
L = [l.strip() for l in c["parameters"]["POINT"]["LABELS"]["value"]]
res = np.asarray(m["residuals"])[0]
for name in ("RWRB", "RFIN", "RWRA", "RUPA"):
    r = res[L.index(name)]
    print(name, "resid mean %.4f p50 %.4f p99 %.4f max %.4f" % (r.mean(), np.median(r), np.percentile(r, 99), r.max()))
print("POINT params:", [k for k in c["parameters"]["POINT"].keys()])
