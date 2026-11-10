"""Compare official source clips and a conditional flight on separate axes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = (1440, 898)
FPS = 30
VIDEO_Y = 141
COLORS = ("#77e2f3", "#ffcb81", "#8ee5a3")
BG, INK, MUTED, GRID = "#101b24", "#edf4f8", "#b6c6d0", "#3b5261"
BALL = (
    {182: (728, 382), 186: (696, 374), 187: (689, 372)},
    {180: (739, 383), 181: (729, 377), 182: (723, 373), 183: (716, 369),
     184: (708, 365), 185: (701, 362), 186: (696, 357), 187: (691, 354)},
)


def load_clip(source, pose, key, release):
    path = source / "clips" / f"{key}.mp4"
    provenance = json.loads((source / "pose-provenance.json").read_bytes())
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["video_hashes"][key]:
        raise ValueError(f"Raw video does not match pose input: {key}")
    cap = cv2.VideoCapture(str(path))
    frames = {}
    lookup = {int(f): i for i, f in enumerate(pose[key+"_frame"])}
    for i in range(release+19):
        ok, img = cap.read()
        if not ok:
            raise ValueError(f"Missing source frame {i}: {key}")
        if i >= release-21:
            j = lookup[i]
            if abs(cap.get(cv2.CAP_PROP_POS_MSEC)-pose[key+"_time"][j]) > .01:
                raise ValueError(f"Pose/video PTS mismatch {key}:{i}")
            if img.shape[:2] != (720,1280):
                raise ValueError("Manual reference points require 1280×720 source")
            img = cv2.cvtColor(cv2.resize(img,(696,392),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            frames[i] = Image.fromarray(img), pose[key+"_point"][j], pose[key+"_score"][j]
    cap.release()
    return frames


def draw_clip(canvas, draw, clip, i, panel, color, fonts):
    x = 16+panel*712
    image, points, scores = clip[i]
    canvas.paste(image,(x,VIDEO_Y))

    def px(v):
        return x + float(v[0]) * 696/1280, VIDEO_Y + float(v[1]) * 392/720

    for a,b in ((11,12),(11,23),(12,24),(23,24),(12,14),(14,16),(23,25),(25,27)):
        if min(scores[a],scores[b]) >= .5 and np.isfinite(points[[a,b]]).all():
            u,v=px(points[a]),px(points[b])
            if 5 <= np.linalg.norm(points[a]-points[b]) <= 105:
                draw.line((u,v),fill=color,width=3)
                for tx,ty in (u,v):
                    draw.ellipse((tx-3,ty-3,tx+3,ty+3),fill=color)
    last = [px(point) for frame,point in BALL[panel].items() if frame <= i]
    for u,v in last:
        draw.ellipse((u-4,v-4,u+4,v+4),outline="#fcfafa",width=2)
    if i in BALL[panel]:
        u,v = px(BALL[panel][i])
        draw.ellipse((u-8,v-8,u+8,v+8),outline="#ffffff",width=2)
    draw.rectangle((x+12,VIDEO_Y+12,x+191,VIDEO_Y+42),fill=BG)
    draw.text((x+20,VIDEO_Y+15),f"원본 프레임 {i}",font=fonts[16],fill=INK)


def draw_flight(draw, result, time, fonts):
    x0,y0,x1,y1 = 32,626,900,808
    draw.text((22,563),"같은 기준 투구 · 팔 슬롯만 +1.4°",font=fonts[27],fill=INK)
    draw.text((23,600),"모델 좌표의 측면 비행 · 두 조건의 구속·회전수는 동일",font=fonts[17],fill=MUTED)
    base = result["nominal"]["baseline"]
    variant = result["nominal"]["slot_only"]
    dist = base["path_xyz_m"][-1][1]
    for z in (.5,1.0,1.5):
        y = y1-(y1-y0)*(z-.15)/1.7
        draw.line((x0,y,x1,y),fill=GRID,width=1)
    draw.line((x1,y0,x1,y1),fill=GRID,width=1)
    draw.text((x0,813),"이탈",font=fonts[17],fill=MUTED)
    draw.text((x1-80,813),"플레이트",font=fonts[17],fill=MUTED)
    for data,color in ((base,COLORS[0]),(variant,COLORS[2])):
        path = np.asarray(data["path_xyz_m"])
        pixels = np.column_stack((x0+(x1-x0)*path[:,1]/dist, y1-(y1-y0)*(path[:,2]-.15)/1.7))
        draw.line([tuple(v) for v in pixels[::5]],fill=GRID,width=2)
        k = min(int(np.searchsorted(data["t_s"],time)),len(pixels)-1)
        sampled = [tuple(v) for v in pixels[:k:4]]+[tuple(pixels[k])]
        if len(sampled)>1:
            draw.line(sampled,fill=color,width=2)
        u,v = pixels[k]
        draw.ellipse((u-4,v-4,u+4,v+4),outline=color,width=2)

    draw.rectangle((935,563,1420,831),fill="#172735",outline=GRID,width=1)
    dx,dz = result["delta"]["slot_only"]["plate_xz_cm"]
    draw.text((953,579),"플레이트 통과점 차이",font=fonts[23],fill=INK)
    draw.text((953,613),f"좌우 {dx:+.2f} cm  ·  높이 {dz:+.2f} cm",font=fonts[27],fill=COLORS[2])
    draw.text((953,655),"기준점 주변 확대 · 포수 시점",font=fonts[16],fill=MUTED)
    ox,oy = 1058,770
    draw.line((991,oy,1143,oy),fill=GRID)
    draw.line((ox,690,ox,806),fill=GRID)
    draw.ellipse((ox-6,oy-6,ox+6,oy+6),fill=COLORS[0])
    u,v = ox+dx*36,oy-dz*36
    draw.line((ox,oy,u,v),fill=COLORS[2],width=2)
    draw.ellipse((u-6,v-6,u+6,v+6),fill=COLORS[2])
    draw.text((1163,711),"● 원래 투구의 공 상태",font=fonts[16],fill=COLORS[0])
    draw.text((1163,749),"● 슬롯 기하 변경",font=fonts[16],fill=COLORS[2])
    draw.text((1163,786),"RPM·수직 양력 변화 0",font=fonts[16],fill=MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_bytes())
    keys = tuple(p[:8] for p in result["play_ids"])
    if keys != ("132a6129","98d35e2d") or abs(result["slot_change_deg"]-1.4) > 1e-6:
        raise ValueError("Manual video marks are scoped to the fixed representative case")
    releases = 182,180
    with np.load(args.source/"pose.npz",allow_pickle=False) as pose:
        clips = [load_clip(args.source,pose,key,release) for key,release in zip(keys,releases)]
    fonts = {n: ImageFont.truetype(str(args.font),n) for n in (16,17,20,23,27,31)}
    template = Image.new("RGB",SIZE,BG)
    text = ImageDraw.Draw(template)
    text.text((22,13),"스킨스 포심 · 비슷한 구속, 다른 회전과 수직 무브먼트",font=fonts[31],fill=INK)
    text.text((24,56),"2024년 5월 11일 · 1회 같은 타석 연속 두 공 · 첫 자유 공 후보로 정렬",font=fonts[20],fill=MUTED)
    for i in range(2):
        stat = result["observed"]
        label = f"{stat['release_speed_kmh'][i]:.2f} km/h  ·  {stat['rpm'][i]:,.0f} RPM  ·  수직 {stat['pfx_z_cm'][i]:.1f} cm"
        text.text((22+i*712,86),label,font=fonts[23],fill=COLORS[i])
        text.text((24+i*712,114),f"Statcast 팔 슬롯 {stat['arm_angle_deg'][i]:.1f}°  ·  화면 추적선은 2D",font=fonts[16],fill=MUTED)
    text.line((22,548,1418,548),fill=GRID)
    text.text((22,853),"위  MLB 원본·관측 2D 관절과 공   /   아래  별도 물리 좌표의 조건부 비행 · 방송 카메라와 3D 정렬하지 않음",font=fonts[16],fill=MUTED)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg,"-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{SIZE[0]}x{SIZE[1]}",
           "-r",str(FPS),"-i","-","-an","-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(args.output)]
    count = 0
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for offset in range(-21,19):
            im = template.copy()
            draw = ImageDraw.Draw(im)
            for i in range(2):
                draw_clip(im,draw,clips[i],releases[i]+offset,i,COLORS[i],fonts)
            # Illustrative release alignment; model is drawn in its own time and coordinates.
            time = max(0.,offset/18*result["nominal"]["baseline"]["flight_s"])
            draw_flight(draw,result,time,fonts)
            if offset == 12:
                im.save(args.output.with_suffix(".png"))
            repeat = 3 if offset != 18 else 63
            proc.stdin.write(im.tobytes()*repeat)
            count += repeat
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Overlay encoding failed")
    subprocess.run([ffmpeg,"-v","error","-xerror","-i",str(args.output),"-f","null","-"],check=True)
    print(f"Fully decoded {count} rendered frames")


if __name__ == "__main__":
    main()
