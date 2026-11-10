"""Review matched and rejected tracking hypotheses on the original baseball frames."""
import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.shared_ball_tracking.study import decode


PANELS = (
    ("academy-000003", "아카데미 · 야외 포수 시점"),
    ("delft-02", "TU Delft · 실내 측면"),
    ("delft-01", "TU Delft · 초점 흐림 검사"),
    ("mlb-low", "MLB · 중계"),
    ("prime-19", "OBP Prime · 실험실"),
    ("rapsodo-4", "Rapsodo · 후방 고속 영상"),
)
STYLES = (
    ("app", (255, 187, 92), "기존 앱"),
    ("round_contrast", (101, 226, 255), "원형 대비"),
    ("proposal_union", (232, 151, 255), "후보 결합"),
)


def render(manifest_path, report_path, output, font_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = {row["id"]:row for row in manifest["clips"]}
    report = {row["id"]:row for row in json.loads(report_path.read_text(encoding="utf-8"))["clips"]}
    regular = ImageFont.truetype(str(font_path), 15)
    large = ImageFont.truetype(str(font_path), 24)
    width, height = 1440, 728
    chrome = Image.new("RGB", (width,height), "#162625")
    label = ImageDraw.Draw(chrome)
    label.text((18,8),"같은 공 추적 모델을 여러 야구 영상에 적용하면",font=large,fill="white")
    for i, (_,color,name) in enumerate(STYLES):
        label.text((19+125*i,45),name,font=regular,fill=color)
    label.text((420,45),"흰색 ×  영상 기준점",font=regular,fill="white")
    label.text((18,697),"각 패널은 검토 구간을 같은 길이로 재생합니다. 좌표는 원영상에서 옮겼고 장비 구속은 사용하지 않았습니다.",
               font=regular,fill="#c5d5cf")
    panels = []
    for index,(key,name) in enumerate(PANELS):
        source, scores = clips[key], report[key]
        images, frames, factors, _ = decode(manifest_path.parent/source["path"],source["window"])
        left = index%3*480
        top = 76 + index//3*306
        n = len(source["positive"])
        numbers = " / ".join(str(scores["models"][mode]["score"]["matched"]) for mode,_,_ in STYLES)
        label.text((left+12,top+6),f"{name}  {numbers} · 기준 {n}",font=regular,fill="white")
        lookup = {frame:np.asarray([x,y])*factors for frame,x,y in source["positive"]}
        panels.append((images,frames,scores,lookup,left,top+36))
    writer = imageio_ffmpeg.write_frames(str(output),(width,height),fps=25,codec="libx264",
                                        pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=2)
    writer.send(None)
    try:
        for i in range(125):
            progress = min(i,99)/99
            screen = np.asarray(chrome).copy()
            for images,frames,scores,lookup,left,top in panels:
                j = round(progress*(len(images)-1))
                frame = frames[j]["frame"]
                picture = cv2.resize(cv2.cvtColor(images[j],cv2.COLOR_BGR2RGB),(480,270),interpolation=cv2.INTER_AREA)
                sx, sy = 480/scores["width"],270/scores["height"]
                for index,(mode,color,_) in enumerate(STYLES):
                    nearby = [p for p in scores["models"][mode]["points"] if frame-5 <= p["frame"] <= frame]
                    for point in nearby:
                        xy = round(point["x"]*sx),round(point["y"]*sy)
                        cv2.circle(picture,xy,4+index*2 if point["frame"]==frame else 2,color,
                                   1 if point["frame"]==frame else -1,cv2.LINE_AA)
                if frame in lookup:
                    x,y = lookup[frame]*[sx,sy]
                    cv2.drawMarker(picture,(round(x),round(y)),(255,255,255),cv2.MARKER_TILTED_CROSS,9,1,cv2.LINE_AA)
                cv2.rectangle(picture,(6,6),(77,27),(22,38,37),-1)
                cv2.putText(picture,f"F{frame}",(10,22),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,246,243),1,cv2.LINE_AA)
                screen[top:top+270,left:left+480] = picture
            writer.send(screen.tobytes())
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","report","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--font",required=True,type=Path)
    args = parser.parse_args()
    render(args.manifest,args.report,args.output,args.font)
