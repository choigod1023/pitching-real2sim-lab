"""Render release intervals, accepted ball positions and raw reference points."""
import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.shared_ball_tracking.study import decode


CASES = (
    ("academy-000001", "야외 아카데미 · 미사용 투구"),
    ("mlb-778343-low", "MLB · 미사용 경기"),
    ("prime-22", "OBP Prime · 같은 투구, 다른 카메라"),
    ("delft-01", "TU Delft · 흐린 영상, 개발 자료"),
)


def render(manifest_path, report_path, destination, font):
    source = {clip["id"]:clip for clip in json.loads(manifest_path.read_text(encoding="utf-8"))["clips"]}
    report = {clip["id"]:clip for clip in json.loads(report_path.read_text(encoding="utf-8"))["clips"]}
    normal = ImageFont.truetype(str(font), 18)
    title = ImageFont.truetype(str(font), 27)
    base = Image.new("RGB",(1280,888),"#172927")
    labels = ImageDraw.Draw(base)
    labels.text((16,7),"릴리스 표식 뒤 자유 비행 공 추적",font=title,fill="white")
    labels.text((16,46),"주황 기존 앱     하늘색 새 연결     흰색 × 원영상 기준점",font=normal,fill="#dae8e1")
    labels.text((16,853),"각 화면은 검토 구간을 같은 길이로 재생합니다. 숫자는 기준 위치와 일치한 프레임 수입니다.",
                font=normal,fill="#dae8e1")
    panels = []
    for index,(key,name) in enumerate(CASES):
        item, row = source[key], report[key]
        images, frames, scale, _ = decode(manifest_path.parent/item["path"],item["window"])
        left, top = index%2*640, 70+index//2*392
        old = row["models"]["app"]["score"]["matched"]
        new = row["models"]["release_beam"]["score"]["matched"]
        labels.text((left+12,top+4),f"{name}   {old} → {new} / {len(item['positive'])}",
                    font=normal,fill="white")
        references = {frame:np.asarray([x,y])*scale for frame,x,y in item["positive"]}
        panels.append((item,row,images,frames,references,left,top+32))
    writer = imageio_ffmpeg.write_frames(str(destination),(1280,888),fps=25,codec="libx264",
                                        pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=2)
    writer.send(None)
    try:
        for index in range(150):
            progress = min(index,124)/124
            board = np.asarray(base).copy()
            for item,row,images,frames,reference,left,top in panels:
                i = round(progress*(len(images)-1))
                frame = frames[i]["frame"]
                picture = cv2.resize(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB),(640,360),interpolation=cv2.INTER_AREA)
                scale = np.array([640/row["width"],360/row["height"]])
                for mode,color,radius in (("app",(255,185,92),6),("release_beam",(83,225,252),9)):
                    for point in row["models"][mode]["points"]:
                        if frame-3 <= point["frame"] <= frame:
                            xy = tuple(np.rint(np.array([point["x"],point["y"]])*scale).astype(int))
                            cv2.circle(picture,xy,radius if point["frame"]==frame else 2,color,
                                       1 if point["frame"]==frame else -1,cv2.LINE_AA)
                if frame in reference:
                    xy = tuple(np.rint(reference[frame]*scale).astype(int))
                    cv2.drawMarker(picture,xy,(255,255,255),cv2.MARKER_TILTED_CROSS,11,1,cv2.LINE_AA)
                low, high = item["release"]
                status = "PRE" if frame<low else "CONTACT?" if frame<=high else "POST"
                if frame in row["models"]["release_beam"]["ambiguousFrames"]:
                    status += " / HOLD"
                cv2.rectangle(picture,(6,6),(170,30),(20,39,35),-1)
                cv2.putText(picture,f"F{frame}   {status}",(12,22),cv2.FONT_HERSHEY_SIMPLEX,
                            .48,(243,249,246),1,cv2.LINE_AA)
                board[top:top+360,left:left+640] = picture
            writer.send(board.tobytes())
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","report","destination"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--font",required=True,type=Path)
    args = parser.parse_args()
    render(args.manifest,args.report,args.destination,args.font)
