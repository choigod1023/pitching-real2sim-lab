"""New camera17 hand-observation sheet; raw pixels, no model projection/seeds."""
import hashlib
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
from covariant_release import LAB, OUT


def run():
    path = LAB/'release-window-validation-20261028/artifacts/release-window/camera17.mp4'
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('Missing verified additional-view video')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 971)
    # Fixed visible-arm rectangle from raw frame976; no projected skeleton/ball input.
    roi = (350, 260, 700, 530)
    sheet = Image.new('RGB', (1500, 1230), '#101923')
    draw = ImageDraw.Draw(sheet); font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 21)
    frames = []
    for index, frame in enumerate(range(971, 980)):
        ok, raw = cap.read()
        if not ok:
            raise RuntimeError('Native-frame decode failed')
        im = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        dest = OUT/f'camera17_hand_raw_{frame}.png'; im.save(dest)
        crop = im.crop(roi).resize((490, 378))
        x, y = (index % 3)*500, (index//3)*410
        sheet.paste(crop, (x+5, y+32))
        draw.text((x+7, y+4), f'camera17 · decoded {frame} / 영상표기 {frame+1}', font=font, fill='white')
        frames.append({'frame': frame, 'path': str(dest.resolve()), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    cap.release()
    p = OUT/'camera17_hand_observation_sheet.png'; sheet.save(p)
    manifest = {'camera': 17, 'source_path': str(path.resolve()), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'decoded_frames': list(range(971, 980)), 'crop_native_pixels': list(roi), 'model_projected': False,
                'annotations_created': False, 'exact_joint_or_release_truth': False, 'frames': frames,
                'next': 'Annotate only visible wrist/elbow regions before viewing projections; assess repeatability/occlusion and timing across existing cameras. Do not turn image silhouettes into exact anatomical centres.'}
    (OUT/'hand_observation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Raw additional-view hand observation sheet ready.')


if __name__ == '__main__':
    run()
