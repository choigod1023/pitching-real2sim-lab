"""Overlay the source-image track and before/after field-space trajectories."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def render_video(report, videos, tracks, output, font):
    """Render the sparse holdout on the original synchronized-to-first-center clips."""
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1600, 526
    title = Image.new("RGB", (width, 76), "#172820")
    draw = ImageDraw.Draw(title)
    draw.text((22, 9), "Rapsodo 후방 고속 영상 · 공 중심 재투영", font=ImageFont.truetype(str(font), 23), fill="white")
    for x, label, color in ((22, "원영상 공 중심", "#edfff7"), (205, "보정 전", "#ff9a71"),
                            (324, "회전·발사 방향 보정", "#63e4ff")):
        draw.text((x, 45), label, font=ImageFont.truetype(str(font), 16), fill=color)
    with ExitStack() as stack:
        clips = [cv2.VideoCapture(str(video)) for video in videos]
        for clip, trace in zip(clips, tracks):
            stack.callback(clip.release)
            clip.set(cv2.CAP_PROP_POS_FRAMES, trace["ball_trace"][0][0])
        writer = imageio_ffmpeg.write_frames(str(output), (width, height), fps=40, codec="libx264",
                                            pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2)
        writer.send(None)
        stack.callback(writer.close)
        frame_count = int(max(row["late"]["frames"][-1] - trace["ball_trace"][0][0]
                              for row, trace in zip(report["pitches"], tracks))) + 6
        for offset in range(frame_count):
            panels = []
            for row, trace, clip in zip(report["pitches"], tracks, clips):
                ok, frame = clip.read()
                if not ok:
                    raise ValueError(f"Video decode failed at offset {offset}")
                panel = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (800, 450), interpolation=cv2.INTER_AREA)
                current = trace["ball_trace"][0][0] + offset
                late = row["late"]
                first = [point[1:3] for point in trace["ball_trace"] if point[0] <= current]
                last = [point for frame, point in zip(late["frames"], late["observed_centers_px"]) if frame <= current]
                overlays = [((237, 255, 247), first + last)]
                for mode, color in (("before", (255, 154, 113)), ("rotation_and_launch", (99, 228, 255))):
                    overlays.append((color, [point for frame, point in
                                     zip(late["frames"], late["scores"][mode]["projected_px"])
                                     if frame <= current and point is not None]))
                for color, points in overlays:
                    if points:
                        xy = np.round(np.asarray(points) * (800 / 1920)).astype(np.int32)
                        cv2.polylines(panel, [xy], False, color, 2, cv2.LINE_AA)
                        cv2.circle(panel, tuple(xy[-1]), 4, color, 1, cv2.LINE_AA)
                cv2.rectangle(panel, (6, 7), (306, 40), (20, 35, 30), -1)
                subtitle = f"#{row['pitch_id']}  {row['recorded_speed_kmh']:.2f} km/h  |  F{int(current)}"
                cv2.putText(panel, subtitle, (13, 31), cv2.FONT_HERSHEY_SIMPLEX, .63, (245, 249, 246), 1, cv2.LINE_AA)
                panels.append(panel)
            writer.send(np.vstack((np.asarray(title), np.hstack(panels))).tobytes())


def render(report, geometry, videos, tracks, output, font):
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 9.1), facecolor="#fafbf9")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1), hspace=.35, wspace=.26,
                          left=.07, right=.96, top=.83, bottom=.20)
    ink, muted = "#172820", "#4b6358"
    fig.text(.07, .943, "후방 고속 영상으로 공의 진행 방향을 보정하면", fontsize=23, weight="bold", color=ink)
    fig.text(.07, .901, "Rapsodo 포심 두 구  ·  동일 투구의 보정 전후 궤적과 홈플레이트 기록 비교", color=muted)
    for i, (row, video, trace) in enumerate(zip(report["pitches"], videos, tracks)):
        key = str(row["pitch_id"])
        frame = cv2.VideoCapture(str(video))
        try:
            frame.set(cv2.CAP_PROP_POS_FRAMES, 45)
            ok, image = frame.read()
        finally:
            frame.release()
        if not ok:
            raise ValueError(f"Cannot read frame 45 from {video.name}")
        scene = geometry["scenes"][key]
        axis = fig.add_subplot(grid[0, i])
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for name, color in (("plate_and_device_px", "#ffd475"), ("wall_primary_px", "#ffd475"),
                            ("wall_heldout_px", "#fffae8"), ("vertical_primary_px", "#ffd475"),
                            ("vertical_heldout_px", "#fffae8")):
            xy = np.asarray(scene[name])
            if name == "plate_and_device_px":
                xy = np.vstack((row["forward_vanishing_point_px"], xy))
            axis.plot(*xy.T, color=color, lw=1.5, marker="o", ms=3)
        centers = np.asarray(trace["ball_trace"])[:, 1:3]
        axis.plot(*centers.T, color="#79f9d9", lw=1.25)
        axis.scatter(*centers[::7].T, s=9, c="#79f9d9", edgecolors="#20362e", linewidths=.35)
        if "late" in row:
            late = row["late"]
            late_uv = np.asarray(late["observed_centers_px"])
            for mode, color, style in (("before", "#ff9a71", "--"),
                                       ("rotation_and_launch", "#63e4ff", "-")):
                pixels = np.asarray([point for point in late["scores"][mode]["projected_px"] if point is not None])
                if len(pixels):
                    axis.plot(*pixels.T, color=color, ls=style, lw=1.3, marker="o", ms=2.1)
            axis.plot(*late_uv.T, color="#e0fbf2", ls=":", lw=1.3, marker="o", ms=2.6)
        axis.set_title(f"#{key}  {row['recorded_speed_kmh']:.2f} km/h · 원본 F45",
                       color=ink, loc="left", weight="bold")
        axis.axis("off")
        axis = fig.add_subplot(grid[1, i])
        for mode, color, style, label in (
            ("before", "#794321", "--", "보정 전"),
            ("rotation_only", "#7654a0", ":", "회전만 보정"),
            ("rotation_and_launch", "#076c68", "-", "회전·발사 방향 보정"),
        ):
            path = np.asarray(row["flights"][mode]["path_yz_m"])
            axis.plot(path[:, 0], path[:, 1], color=color, ls=style, lw=2.2, label=label)
        path = np.asarray(row["video_path_yz_m_registered"])
        axis.scatter(path[::3, 0], path[::3, 1], color="#778880", s=7, alpha=.72,
                     label="영상 중심 복원 구간")
        if "late" in row:
            late_path = np.asarray(late["video_path_yz_m_registered"])
            axis.scatter(*late_path.T, facecolor="none", edgecolor="#43594e", s=21, linewidth=.85,
                         label="후반 공 윤곽 복원")
        axis.plot(16.8, row["recorded_plate_height_cm"] / 100, "o", color=ink, ms=6,
                  label="Rapsodo 통과 높이", zorder=6)
        axis.axvline(16.8, color="#9baea3", lw=1)
        axis.grid(color="#e0e9e3", lw=.8)
        axis.set_xlim(-.2, 17.8)
        axis.set_ylim(.5, 1.85)
        axis.set_xlabel("릴리스 이후 전방 거리 (m)")
        axis.set_ylabel("공 높이 (m)")
        if i == 0:
            handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower left", bbox_to_anchor=(.065, .082),
               frameon=False, columnspacing=1.35)
    fig.text(.07, .051, "원영상에서 흰 점은 후반 공 중심, 주황색과 하늘색은 보정 전후 재투영입니다. 노란 선은 배경 기준입니다.", color=muted)
    fig.text(.07, .026, "영상 복원은 첫 검출 중심을 기록 릴리스 높이에 맞췄습니다. 후반의 빈 원은 공 윤곽으로 얻은 추정치입니다.",
             color=muted)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "video_4", "track_4", "video_8", "track_8", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--font", required=True, type=Path)
    parser.add_argument("--video-output", type=Path)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))

    report = read(args.report)
    videos = [args.video_4, args.video_8]
    tracks = [read(args.track_4), read(args.track_8)]
    render(report, read(Path(__file__).with_name("scene.json")), videos, tracks, args.output, args.font)
    if args.video_output is not None:
        render_video(report, videos, tracks, args.video_output, args.font)
