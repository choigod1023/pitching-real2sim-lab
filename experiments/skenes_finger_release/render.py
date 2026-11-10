"""Render broadcast evidence and the modeled contact/flight in separate coordinates."""
import argparse
import hashlib
import json
from math import cos, radians, sin
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.skenes_finger_release.study import PRIMARY_SHIFT_MM, RADIUS

WIDTH, HEIGHT, FPS = 1440, 896, 30
BG, INK, MUTED, GRID = "#11202a", "#f2f6f9", "#bacbd6", "#4f6676"
BASE, CANDIDATE, RECORDED = "#6fe2f2", "#a0e2a5", "#ffca84"
CROP = (570, 320, 800, 460)
FREE = (182, 180)
VISIBLE_BALL = (
    {182: (728,382), 186: (696,374), 187: (689,372)},
    {180: (739,383), 181: (729,377), 182: (723,373), 183: (716,369),
     184: (708,365), 185: (701,362), 186: (696,357), 187: (691,354)},
)


def read_crop(path, source_hash, release):
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise ValueError(f"Different source for manually marked frames: {path.name}")
    capture = cv2.VideoCapture(str(path))
    frames = {}
    x0,y0,x1,y1 = CROP
    for i in range(release+9):
        ok, frame = capture.read()
        if not ok or frame.shape[:2] != (720,1280):
            raise ValueError(f"Cannot decode source frame {i}")
        if i >= release-10:
            pixels = cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
            frames[i] = Image.fromarray(pixels).resize((690,420),Image.Resampling.NEAREST)
    capture.release()
    return frames


def frame_image(offset, state, clips, fonts):
    canvas = Image.new("RGB",(WIDTH,HEIGHT),BG)
    d = ImageDraw.Draw(canvas)
    d.text((23,13),"스킨스 포심의 릴리스 · 실제 영상과 손–공 접촉 가정",font=fonts[30],fill=INK)
    d.text((24,58),"2024년 5월 11일, 같은 타석 두 공 · 원본 픽셀 확대 · 첫 자유 공 후보에 맞춤",font=fonts[19],fill=MUTED)
    obs = state["observed"]
    for i, color in enumerate((BASE,RECORDED)):
        x,y = 24+702*i,126
        frame = FREE[i]+offset
        canvas.paste(clips[i][frame],(x,y))
        label = f"{obs['speed_kmh'][i]:.2f} km/h  ·  {obs['spin_rpm'][i]:,.0f} RPM  ·  수직 {obs['vertical_break_cm'][i]:.1f} cm"
        d.text((x,91),label,font=fonts[23],fill=color)
        d.rectangle((x+9,y+10,x+208,y+44),fill=BG)
        d.text((x+16,y+15),f"원본 프레임 {frame}",font=fonts[18],fill=INK)
        if frame in VISIBLE_BALL[i]:
            u,v = VISIBLE_BALL[i][frame]
            px,py = x+(u-CROP[0])*3,y+(v-CROP[1])*3
            d.ellipse((px-12,py-12,px+12,py+12),outline="#f9ffff",width=3)
        d.rectangle((x+9,y+376,x+430,y+409),fill=BG)
        caption = "공이 분리돼 보이는 후보" if frame == FREE[i] else "개별 손가락과 마지막 접촉점은 판독 불가"
        d.text((x+16,y+381),caption,font=fonts[17],fill=INK)

    d.line((24,564,1416,564),fill=GRID)
    d.text((24,578),"공 표면 힘 작용점 +4 mm",font=fonts[25],fill=INK)
    d.text((24,613),"같은 합력 · 접촉 단면은 모델 좌표",font=fonts[17],fill=MUTED)
    cx,cy,r = 282,750,91
    d.ellipse((cx-r,cy-r,cx+r,cy+r),outline=GRID,width=2)
    d.line((cx-5,cy,cx+5,cy),fill=MUTED)
    d.line((cx,cy-5,cx,cy+5),fill=MUTED)
    for angle,color in ((radians(12),BASE),(radians(12)+PRIMARY_SHIFT_MM/(1000*RADIUS),CANDIDATE)):
        px,py = cx-r*cos(angle),cy-r*sin(angle)
        d.line((cx,cy,px,py),fill=color,width=2)
        d.ellipse((px-5,py-5,px+5,py+5),fill=color)
        d.line((px,py,px+54,py),fill=color,width=2)
        d.polygon(((px+54,py),(px+47,py-4),(px+47,py+4)),fill=color)
    d.text((25,811),"● 기준 작용점",font=fonts[17],fill=BASE)
    d.text((25,836),"● 변경 작용점",font=fonts[17],fill=CANDIDATE)

    d.rectangle((431,580,1416,847),outline=GRID,width=1)
    delta = state["delta"]
    base,candidate = (state["nominal"][k] for k in ("baseline","candidate"))
    phase = int(np.clip((offset+5)*16/5,0,16))
    d.text((453,595),f"같은 공의 최종 8 ms 접촉 계산 · 모델 {base['t_ms'][phase]:+.1f} ms",font=fonts[22],fill=INK)
    d.text((453,636),f"{base['rpm_trace'][phase]:,.0f} → {candidate['rpm_trace'][phase]:,.0f} RPM",font=fonts[29],fill=CANDIDATE)
    d.text((453,680),f"접촉 종료 +{delta['rpm']:.1f} RPM  ·  구속 차이 {delta['release_speed_kmh']:+.2f} km/h",font=fonts[19],fill=INK)
    d.text((453,716),f"수직 양력 변위 {delta['lift_xz_cm'][1]:+.2f} cm",font=fonts[23],fill=INK)
    d.text((453,754),f"회전 에너지 전달량 {delta['rotational_work_j']:+.3f} J",font=fonts[19],fill=MUTED)
    d.text((453,810),"기준 RPM은 실제 기록으로 초기화",font=fonts[17],fill=MUTED)
    x,y = 1238,745
    d.line((1155,y,1357,y),fill=GRID)
    d.line((x,686,x,804),fill=GRID)
    d.text((1038,599),"통과점 주변 확대 · 포수 시점",font=fonts[17],fill=MUTED)
    dx,dz = delta["plate_xz_cm"]
    d.text((1038,637),f"좌우 {dx:+.2f} cm · 높이 {dz:+.2f} cm",font=fonts[20],fill=INK)
    u,v = x+dx*122,y-dz*122
    d.line((x,y,u,v),fill=CANDIDATE,width=2)
    d.ellipse((x-7,y-7,x+7,y+7),fill=BASE)
    d.ellipse((u-7,v-7,u+7,v+7),fill=CANDIDATE)
    d.text((24,864),"위  MLB 원영상에서 판독한 공   /   아래  힘을 가정한 모형 · 원영상과 시간·좌표는 별도",font=fonts[17],fill=MUTED)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.results.read_bytes())
    keys = [key[:8] for key in state["play_ids"]]
    if keys != ["132a6129","98d35e2d"]:
        raise ValueError("Manual image marks are for the representative May 11 case")
    provenance = json.loads((args.source/"pose-provenance.json").read_bytes())
    clips = [read_crop(args.source/"clips"/f"{key}.mp4",provenance["video_hashes"][key],release)
             for key,release in zip(keys,FREE,strict=True)]
    fonts = {n:ImageFont.truetype(str(args.font),n) for n in (17,18,19,20,22,23,25,29,30)}
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg,"-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{WIDTH}x{HEIGHT}","-r",str(FPS),
           "-i","-","-an","-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(args.output)]
    count = 0
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as process:
        for offset in range(-10,9):
            canvas = frame_image(offset,state,clips,fonts)
            repeat = 45 if offset in (-10,0,8) else 5
            if offset == 0:
                canvas.save(args.output.with_suffix(".png"))
            process.stdin.write(canvas.tobytes()*repeat)
            count += repeat
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Overlay encoding failed")
    subprocess.run([ffmpeg,"-v","error","-xerror","-i",str(args.output),"-f","null","-"],check=True)
    print(f"Fully decoded {count} rendered frames")


if __name__ == "__main__":
    main()
