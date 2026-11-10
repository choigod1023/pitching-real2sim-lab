"""Show measured image texture, simulated flights and the recorded plate endpoints."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from experiments.seam_axis_flight.rotation import sphere_images


def texture_overlay(video, tracking, result):
    trace = tracking["ball_trace"]
    frame = result["test_source_frames"][0]
    consecutive = [point for point in trace if point[0] in (frame, frame + 1)]
    source = sphere_images(video, consecutive[:1])[0]
    _, x, y, radius, _ = consecutive[-1]
    size = 181
    u, down = np.meshgrid(np.linspace(-1.22, 1.22, size, dtype=np.float32),
                          np.linspace(-1.22, 1.22, size, dtype=np.float32))
    with_frame = cv2.VideoCapture(str(video))
    try:
        with_frame.set(cv2.CAP_PROP_POS_FRAMES, frame + 1)
        ok, image = with_frame.read()
        if not ok:
            raise ValueError(f"Cannot read video frame {frame + 1}")
    finally:
        with_frame.release()
    background = cv2.remap(image, x + radius * u, y + radius * down, cv2.INTER_LINEAR)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    inside = u*u + down*down < .96**2
    p = np.column_stack((u[inside], -down[inside], np.sqrt(1-u[inside]**2-down[inside]**2)))
    axis = np.asarray(result["camera_axes"]["fit_0.60"])
    earlier = Rotation.from_rotvec(-result["radians_per_frame"] * axis).apply(p)
    valid = (earlier[:, 2] > 0) & (np.linalg.norm(earlier[:, :2], axis=1) <= .60)
    detail = map_coordinates(source, [32*(1-earlier[:, 1]), 32*(1+earlier[:, 0])], order=1)
    marked = valid & (detail < np.percentile(detail[valid], 18))
    pixels = np.flatnonzero(inside)[marked]
    overlay = background.copy()
    flat = overlay.reshape(-1, 3)
    flat[pixels] = .52 * flat[pixels] + .48 * np.array([111, 245, 191])
    return background, overlay, frame


def render(report, rotations, sources, output, font):
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 8.6), facecolor="#fbfcfb")
    grid = fig.add_gridspec(2, 4, height_ratios=(1.13, 1), hspace=.48, wspace=.34,
                          left=.07, right=.965, top=.78, bottom=.15)
    ink, muted = "#172820", "#4b6358"
    fig.text(.07, .938, "고속 영상에서 추정한 회전과 포심 비행", fontsize=24, weight="bold", color=ink)
    fig.text(.07, .89, "구속 차이 2.41 km/h인 공개 포심 두 구  ·  실밥 무늬 정합과 홈 앞면 높이 비교", color=muted)
    for i, (pitch, (video, track)) in enumerate(zip(report["pitches"], sources)):
        number = str(pitch["pitch_id"])
        raw, overlay, frame = texture_overlay(video, track, rotations[number])
        for k, picture in enumerate((raw, overlay)):
            axis = fig.add_subplot(grid[0, 2*i+k])
            axis.imshow(picture)
            axis.set_title(f"#{number}  F{frame+1} 원본" if not k else f"F{frame}→{frame+1} 무늬 정합",
                           color=ink, fontsize=11, pad=7)
            axis.axis("off")
        axis = fig.add_subplot(grid[1, 2*i:2*i+2])
        for field, color, style, label in (
            ("shared_efficiency_recorded_direction", "#9d501e", ":", "기록 축·공통 효율"),
            ("video_direction_efficiency", "#176b74", "-", "영상 축·영상 효율"),
        ):
            path = np.asarray(pitch["flights"][field]["path_yz_m"])
            axis.plot(path[:, 0], path[:, 1], color=color, ls=style, lw=2, label=label)
        axis.plot(16.8, pitch["recorded_plate_height_cm"] / 100, "o", color=ink, ms=5,
                  label="Rapsodo 통과 높이", zorder=5)
        axis.axvline(16.8, color="#9baea3", lw=1)
        axis.grid(color="#e0e9e3", lw=.8)
        axis.set_xlim(-.2, 17.8)
        axis.set_xlabel("릴리스 이후 전방 거리 (m)")
        axis.set_ylabel("공 높이 (m)")
        axis.set_title(f"#{number}  {pitch['speed_kmh']:.2f} km/h", loc="left", weight="bold", color=ink)
        if i == 0:
            handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper left", bbox_to_anchor=(.064, .857), frameon=False)
    fig.text(.07, .073, "민트색은 이전 프레임의 어두운 표면 무늬를 추정 회전으로 옮긴 위치입니다.", color=muted)
    fig.text(.07, .042, "선은 16.8 m를 가정한 모델 궤적입니다. 검은 점은 기록 높이이며 연속 계측 궤적은 공개되지 않았습니다.",
             color=muted)
    fig.savefig(output, dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("flight", "rotations", "video_4", "track_4", "video_8", "track_8", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--font", required=True, type=Path)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    render(read(args.flight), read(args.rotations),
           [(args.video_4, read(args.track_4)), (args.video_8, read(args.track_8))], args.output, args.font)
