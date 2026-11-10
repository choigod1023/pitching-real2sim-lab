"""Render the recorded TrackMan polynomial and simulated paths at the same plate plane."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


def render(report, output, font_path):
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    figure.subplots_adjust(top=.75, bottom=.22, left=.11, right=.95, hspace=.39)
    figure.patch.set_facecolor("#fbfcfb")
    ink, faint = "#182722", "#55645d"
    colors = ("#18578e", "#a94422")
    settings = report.get("overlay", {})
    figure.text(.085, .94, settings.get("title", "TrackMan 기록과 비행 모델의 포심 궤적"),
                fontsize=23, weight="bold", color=ink)
    first, second = report["examples"]
    gap = abs(first["speed_kmh"] - second["speed_kmh"])
    prefix = settings.get("subtitle", "같은 경기의 연속 두 구")
    figure.text(.087, .886, f"{prefix}  ·  구속 차이 {gap:.2f} km/h  ·  홈플레이트 앞면",
                color=faint)
    styles = [("reference_xyz_m", "-", True, settings.get("reference_label", "TrackMan 적합 궤적")),
              ("model_xyz_m", "--", False, settings.get("model_label", "시뮬레이션"))]
    if all("backspin_xyz_m" in pitch for pitch in report["examples"]):
        styles.append(("backspin_xyz_m", ":", False, "백스핀 고정"))

    for axis, coordinate, label in zip(axes, (2, 0), ("측면에서 본 높이 (m)", "위에서 본 좌우 위치 (m)")):
        axis.set_title(label, loc="left", fontsize=12, weight="bold", color=ink, pad=8)
        axis.axvline(0, color="#687c70", linewidth=1, zorder=1)
        axis.grid(axis="both", color="#e2e9e5", linewidth=.8, zorder=0)
        axis.set_axisbelow(True)
        for color, pitch in zip(colors, report["examples"]):
            for key, dash, filled, _ in styles:
                path = np.array(pitch[key])
                axis.plot(path[:, 1], path[:, coordinate], linestyle=dash, color=color, lw=2)
                axis.plot(0, path[-1, coordinate], "o", color=color, markersize=5,
                          markerfacecolor=color if filled else "white", zorder=4)
    axes[-1].set_xlabel("홈플레이트 앞면 기준 전방 위치 (m)", labelpad=8)
    axes[-1].set_xlim(-16.8, .8)
    items = [Line2D([0], [0], lw=3, color=color, label=row.get("label", f"{row['pitch_number']}번"))
             for color, row in zip(colors, report["examples"])]
    items += [Line2D([0], [0], lw=2, color=faint, linestyle=style, label=label)
              for _, style, _, label in styles]
    figure.legend(handles=items, ncol=len(items), frameon=False, loc="upper left",
                  bbox_to_anchor=(.075, .859), handlelength=2.6)
    label = settings.get("delta_label", f"{first['pitch_number']}→{second['pitch_number']}번")
    figure.text(.087, .10, f"{label} 수직 무브먼트(IVB) 변화"
                f"   기록 {second['recorded_ivb_cm'] - first['recorded_ivb_cm']:+.2f} cm"
                f"   모델 {second['model_ivb_cm'] - first['model_ivb_cm']:+.2f} cm", color=ink, weight="bold")
    note = settings.get("note", "실선은 장비의 궤적 다항식입니다. 점선 모델에는 같은 비행에서 역산한 회전축이 들어갑니다.")
    figure.text(.087, .052, note, color=faint, fontsize=10)
    figure.savefig(output, dpi=155, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--font", type=Path, required=True, help="A font with Korean glyphs")
    args = parser.parse_args()
    render(json.loads(args.report.read_text(encoding="utf-8")), args.output, args.font)
