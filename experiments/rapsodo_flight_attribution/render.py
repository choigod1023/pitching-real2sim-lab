"""Overlay Rapsodo flight assumptions on source frames and in field coordinates."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
import numpy as np


STYLES = {
    "before": ("#6b6e73", "--", "카메라 보정 전"),
    "calibrated": ("#ac442a", "-", "카메라 보정 후"),
    "recorded_tilt": ("#7149a2", ":", "회전축 시계각만 교체"),
    "lyu_lift": ("#007969", "-.", "포심 양력식만 교체"),
}
INK = "#172925"
PLATE_COLOR = "#e8d7b6"


def plot_paths(axis, row, detail):
    modes = row["modes"]
    near, far = modes["near_plate"], modes["far_plate"]
    axis.axvspan(near["distance_m"], far["distance_m"], color=PLATE_COLOR, alpha=.28, zorder=0)
    axis.axvline(modes["calibrated"]["distance_m"], color="#9d8e71", lw=.9, zorder=0)
    for mode, (color, style, label) in STYLES.items():
        path = np.asarray(modes["far_plate" if mode == "calibrated" else mode]["path_m"])
        axis.plot(path[:, 1], path[:, 2] * 100, color=color, ls=style, lw=2., label=label)
        end = modes[mode]
        axis.plot(end["distance_m"], end["plate_height_cm"], "o", color=color, ms=3.5)
    early = np.asarray(row["video_path_yz_m_registered"])
    late = np.asarray(row["late_video_path_yz_m_registered"])
    axis.scatter(early[::6, 0], early[::6, 1] * 100, s=8, color="#718280", alpha=.68,
                 label="영상 공 중심에서 복원")
    axis.scatter(late[:, 0], late[:, 1] * 100, s=16, color="#718280", marker="+")
    axis.scatter([near["distance_m"], far["distance_m"]], [near["plate_height_cm"], far["plate_height_cm"]],
                 facecolors="#fbfcfa", edgecolors=STYLES["calibrated"][0], marker="s", s=24, zorder=5,
                 label="같은 궤적의 앞·뒤 평면")
    axis.plot(16.8, row["reference"]["plate_height_cm"], "x", color=INK, mew=2, ms=7, zorder=6,
              label="Rapsodo 통과 높이")
    if detail:
        axis.set_xlim(15.65, 17.48)
        limits = [modes[name]["plate_height_cm"] for name in modes] + [row["reference"]["plate_height_cm"]]
        axis.set_ylim(min(limits) - 7, max(limits) + 14)
    else:
        axis.set_xlim(-.15, 17.5)
        axis.set_ylim(50, 172)
    axis.set_xlabel("릴리스 이후 거리 (m)")
    axis.set_ylabel("공 높이 (cm)")
    axis.grid(color="#e2e9e4", alpha=.85)


def render_figure(report, output):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.), facecolor="#fafcf9")
    fig.subplots_adjust(left=.085, right=.97, bottom=.13, top=.77, hspace=.48, wspace=.24)
    fig.text(.085, .937, "공의 궤적과 통과 높이를 바꾼 요인", fontsize=25, weight="bold", color=INK)
    fig.text(.085, .894, "같은 포심 두 구에서 회전 방향, 양력식, 플레이트 위치를 하나씩 비교했습니다.", color="#4d6157")
    for i, (row, pair) in enumerate(zip(report["pitches"], axes)):
        for detail, axis in enumerate(pair):
            plot_paths(axis, row, detail)
            suffix = "홈플레이트 근처 확대" if detail else "전체 비행"
            axis.set_title(f"#{row['pitch_id']}  {row['speed_kmh']:.2f} km/h · {suffix}", loc="left",
                           fontsize=11, weight="bold", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper left", bbox_to_anchor=(.079, .872),
               frameon=False, columnspacing=1.4)
    fig.text(.085, .059, "장비 높이 표식은 비교를 위해 16.8m에 놓았습니다. 영상으로 복원한 깊이는 조건부입니다.", color="#4d6157")
    fig.text(.085, .030, "무회전 기준을 중력만 받는 공으로 바꿔도 그려진 실제 비행 모델 궤적은 같습니다.", color="#4d6157")
    fig.savefig(output, dpi=145)
    plt.close(fig)


def render_video(report, videos, traces, output, font):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    width, panel_height, header_height, plot_height = 1600, 450, 70, 290
    frame_height = panel_height + header_height + plot_height
    fig, axes = plt.subplots(1, 2, figsize=(16, 2.9), dpi=100, facecolor="#fafcf9")
    fig.subplots_adjust(left=.052, right=.985, bottom=.19, top=.94, wspace=.19)
    for row, axis in zip(report["pitches"], axes):
        plot_paths(axis, row, detail=True)
    fig.canvas.draw()
    plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    header = Image.new("RGB", (width, header_height), INK)
    draw = ImageDraw.Draw(header)
    draw.text((17, 6), "Rapsodo 포심 두 구 · 회전 방향과 양력식의 영향",
              font=ImageFont.truetype(str(font), 23), fill="white")
    palette = {"calibrated": "#ffa470", "recorded_tilt": "#e5b5ff", "lyu_lift": "#5ef0d5"}
    for x, label, color in ((18, "영상 공 중심", "#fff8ed"), (147, "보정 후", palette["calibrated"]),
                            (245, "시계각 교체", palette["recorded_tilt"]),
                            (371, "양력식 교체", palette["lyu_lift"])):
        draw.text((x, 42), label, font=ImageFont.truetype(str(font), 16), fill=color)

    with ExitStack() as stack:
        readers = [cv2.VideoCapture(str(video)) for video in videos]
        for reader, trace in zip(readers, traces):
            stack.callback(reader.release)
            reader.set(cv2.CAP_PROP_POS_FRAMES, trace[0][0])
        writer = imageio_ffmpeg.write_frames(str(output), (width, frame_height), fps=40, codec="libx264",
                                            pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2)
        writer.send(None)
        stack.callback(writer.close)
        end = int(max(row["late_projection"]["frames"][-1] - trace[0][0]
                      for row, trace in zip(report["pitches"], traces))) + 6
        for offset in range(end):
            panels = []
            for row, reader, trace in zip(report["pitches"], readers, traces):
                ok, frame = reader.read()
                if not ok:
                    raise ValueError(f"Source decode ended at offset {offset}")
                panel = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (800, panel_height),
                                   interpolation=cv2.INTER_AREA)
                current = trace[0][0] + offset
                late = row["late_projection"]
                tracks = [
                    (tuple(round(255 * v) for v in to_rgb(color)), late["modes"][mode]["projected_px"])
                    for mode, color in palette.items()
                ]
                tracks.append(((255, 248, 237), late["observed_px"]))
                for color, points in tracks:
                    visible = [point for frame, point in zip(late["frames"], points) if frame <= current]
                    if color == (255, 248, 237):
                        visible = [point[1:3] for point in trace if point[0] <= current] + visible
                    if visible:
                        xy = np.rint(np.asarray(visible) * (800 / 1920)).astype(np.int32)
                        cv2.polylines(panel, [xy], False, color, 1, cv2.LINE_AA)
                        cv2.circle(panel, tuple(xy[-1]), 4, color, 1, cv2.LINE_AA)
                label = f"#{row['pitch_id']}  {row['speed_kmh']:.2f} km/h  |  F{int(current)}"
                cv2.rectangle(panel, (6, 7), (305, 40), (20, 35, 30), -1)
                cv2.putText(panel, label, (13, 31), cv2.FONT_HERSHEY_SIMPLEX, .63, (245, 249, 246), 1, cv2.LINE_AA)
                panels.append(panel)
            writer.send(np.vstack((np.asarray(header), np.hstack(panels), plot)).tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--font", required=True, type=Path)
    parser.add_argument("--video", type=Path, nargs=5, metavar=("SOURCE4", "TRACE4", "SOURCE8", "TRACE8", "OUTPUT"))
    args = parser.parse_args()
    font_manager.fontManager.addfont(str(args.font))
    family = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    render_figure(report, args.output)
    if args.video is not None:
        source4, trace4, source8, trace8, output = args.video
        traces = [json.loads(trace.read_text(encoding="utf-8-sig"))["ball_trace"] for trace in (trace4, trace8)]
        render_video(report, [source4, source8], traces, output, args.font)
