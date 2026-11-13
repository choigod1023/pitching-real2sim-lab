"""Frozen-ROI 21-point hand inference on existing baseball recordings."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

MODEL_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'


def sha256(path):
    """Hash a local source without modifying it."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def square_roi(anchor, width, image_shape):
    """Keep the complete fixed-width ROI in the source image."""
    h, w = image_shape[:2]
    if width <= 0 or width > min(h, w) or not np.isfinite(anchor).all():
        raise ValueError('Invalid square ROI')
    x, y = np.clip(np.rint(anchor).astype(int) - width // 2, 0, [w-width, h-width])
    return [int(x), int(y), int(width), int(width)]


def upright_rotation(vector):
    """OpenCV rotation that points an observed proximal-to-distal vector upward."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (2,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
        raise ValueError('A finite nonzero image direction is required')
    return float((np.degrees(np.arctan2(vector[1], vector[0])) + 90. + 180.) % 360. - 180.)


def transform_input(frame, roi, angle, side=256):
    """Crop, rotate and resize; return the exact original-to-input affine map."""
    x, y, w, h = map(int, roi)
    if w != h or w <= 0 or side < 1 or x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        raise ValueError('Square ROI must lie inside the image')
    crop = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    center = ((w-1)/2., (h-1)/2.)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.)
    rotated = cv2.warpAffine(crop, rotation, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    resized = cv2.resize(rotated, (side, side), interpolation=cv2.INTER_CUBIC)
    shift = np.array([[1., 0., -x], [0., 1., -y], [0., 0., 1.]])
    rotate = np.vstack((rotation, [0., 0., 1.]))
    # cv2.resize maps pixel centres, not the pixel-corner coordinate system.
    scale = side / w
    resize = np.array([[scale, 0., .5*scale-.5], [0., scale, .5*scale-.5], [0., 0., 1.]])
    return resized, resize @ rotate @ shift


def restore_pixels(normalized, affine, side=256):
    """Invert the image preprocessing for normalized image landmarks only."""
    p = np.asarray(normalized, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError('Expected 21 finite image landmarks')
    homogeneous = np.column_stack((p[:, :2]*side, np.ones(21)))
    result = homogeneous @ np.linalg.inv(np.asarray(affine)).T
    return result[:, :2]/result[:, 2:3]


def select_candidate(candidates, sample, image_shape):
    """Preserve all detections and select only an on-image wrist near the target."""
    h, w = image_shape[:2]
    anchor = np.asarray(sample['wrist_anchor_px'], dtype=float)
    accepted = []
    for index, candidate in enumerate(candidates):
        points = np.asarray(candidate['pixel21'], dtype=float)
        reasons = []
        if points.shape != (21, 2) or not np.isfinite(points).all():
            reasons.append('nonfinite_or_wrong_shape')
            distance = None
        else:
            distance = float(np.linalg.norm(points[0]-anchor))
            if distance > sample['wrist_gate_px']:
                reasons.append('wrist_outside_frozen_anchor_gate')
            if not (0 <= points[0, 0] < w and 0 <= points[0, 1] < h):
                reasons.append('wrist_outside_source_image')
        if sample['role'] != 'target':
            reasons.append('negative_control_no_throwing_hand')
        candidate.update(wrist_distance_px=distance, eligible=not reasons, reasons=reasons,
                         identity_verified=False,
                         opposite_hand_possible=True)
        if not reasons:
            accepted.append((distance, index))
    return min(accepted)[1] if accepted else None


def longest_run(frames):
    """Count genuinely consecutive source frames, not adjacent records."""
    best = count = 0
    previous = None
    for value in sorted(set(frames)):
        count = count+1 if previous is not None and value == previous+1 else 1
        best, previous = max(best, count), value
    return best


def decode_selected(path, indices):
    """Sequentially decode selected source frames and fail on missing required data."""
    wanted = set(indices)
    if not wanted or min(wanted) < 0:
        raise ValueError('Nonempty nonnegative source indices required')
    if Path(path).suffix.lower() == '.gif':
        from PIL import Image, ImageSequence
        with Image.open(path) as video:
            if max(wanted) >= video.n_frames:
                raise ValueError('Required GIF frame is absent')
            for frame, image in enumerate(ImageSequence.Iterator(video)):
                if frame in wanted:
                    yield frame, cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
                if frame >= max(wanted):
                    break
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Could not open source video')
    try:
        for frame in range(max(wanted)+1):
            ok, raw = cap.read()
            if not ok:
                raise ValueError(f'Required source frame {frame} could not be decoded')
            if frame in wanted:
                yield frame, raw
    finally:
        cap.release()


def summarize(clips):
    """Describe support and input disagreement without calling them accuracy."""
    summary = []
    for clip in clips:
        item = dict(clip_id=clip['clip_id'], methods={})
        selected = {}
        for method, records in clip['methods'].items():
            rows = [r for r in records if r['role'] == 'target']
            valid = {r['frame']: np.asarray(r['candidates'][r['selected_index']]['pixel21'])
                     for r in rows if r['selected_index'] is not None}
            selected[method] = valid
            item['methods'][method] = dict(target_frames=len(rows), selected_frames=len(valid),
                selected_fraction=len(valid)/len(rows) if rows else None,
                longest_source_frame_run=longest_run(valid),
                negative_control_roi_samples=sum(r['role'] != 'target' for r in records),
                negative_control_unique_frames=len({r['frame'] for r in records if r['role'] != 'target'}),
                negative_control_detector_hits=sum(bool(r['candidates']) for r in records if r['role'] != 'target'),
                near_release_frames=[f for f in sorted(valid) if f >= clip.get('release_near_first', math.inf)])
        common = sorted(selected['raw'].keys() & selected['rotated'].keys())
        delta = np.array([np.linalg.norm(selected['raw'][f]-selected['rotated'][f], axis=1) for f in common])
        item['two_input_agreement'] = dict(common_frames=common,
            wrist_median_px=float(np.median(delta[:, 0])) if len(delta) else None,
            fingertips_median_px=float(np.median(delta[:, [4, 8, 12, 16, 20]])) if len(delta) else None,
            all21_median_px=float(np.median(delta)) if len(delta) else None,
            all21_max_px=float(np.max(delta)) if len(delta) else None,
            is_ground_truth_accuracy=False)
        summary.append(item)
    return summary


def prepare(lab, out):
    """Freeze inspected sources and raw-image-defined crops before inference."""
    dest = out/'input_manifest.private.json'
    if dest.exists():
        raise FileExistsError('The input manifest is immutable; do not overwrite it')
    prior = lab/'prime-hand-direction-20261107'
    sys.path.insert(0, str(prior))
    from experiments.prime_bridge.cuda_reconstruction import read_geometry
    from multiview_release_validation import project
    source, cameras = read_geometry(lab)
    anchors_file = prior/'artifacts/prime-hand-direction/anchors.private.npy'
    anchors = np.load(anchors_file)
    clips = []
    for view, camera in enumerate((17, 20, 21)):
        path = (lab/'release-window-validation-20261028/artifacts/release-window/camera17.mp4'
                if camera == 17 else lab/f'speed-journey/artifacts/release-observation/camera{camera}.mp4')
        elbow = project(source['origins']['l_larm'], cameras[camera])
        samples = [dict(frame=frame, roi_xywh=square_roi(anchors[view, j], 144, (720, 1280)),
                        wrist_anchor_px=anchors[view, j].tolist(), wrist_gate_px=35., role='target',
                        rotation_deg=upright_rotation(anchors[view, j]-elbow[j]))
                   for j, frame in enumerate(range(940, 977))]
        clips.append(dict(clip_id=f'prime{camera}', domain='prime_same_pitch_3view', path=str(path),
                          camera=camera, source_sha256=sha256(path), encoded_fps=360.,
                          capture_fps_nominal=360., capture_clock_independently_verified=False,
                          frame_shape=[720, 1280, 3], release_near_first=973,
                          visibility='Original frames 940/950/960/973/976 inspected. Hands small, ball occludes fingers; no fingertip ground truth.',
                          preprocessing_basis='Existing Theia wrist anchor and projected forearm direction; provisional alignment; not outcome fitted.',
                          manual_inspection_hand_span_px=[20, 45], samples=samples))
    highspeed = lab/'non-skenes-paired-source-audit-20261110'
    keys = {
        'Branson4S1': [[0, 752, 728, 835, 635], [4, 754, 744, 809, 680],
                      [8, 683, 833, 749, 833], [12, 624, 875, 660, 925], [15, 570, 910, 585, 965]],
        'Branson4S2': [[0, 782, 761, 861, 650], [4, 790, 781, 855, 666],
                      [8, 743, 872, 790, 817], [12, 674, 903, 733, 944], [15, 622, 930, 664, 985]],
    }
    for name, keypoints in keys.items():
        path = highspeed/(name+'.mp4')
        values = np.asarray(keypoints, dtype=float)
        sample_frames = np.arange(16)
        interpolated = np.array([np.interp(sample_frames, values[:, 0], values[:, j]) for j in range(1, 5)]).T
        samples = [dict(frame=int(frame), roi_xywh=square_roi(row[:2], 256, (1080, 1920)),
                        wrist_anchor_px=row[:2].tolist(), wrist_gate_px=70., role='target',
                        rotation_deg=upright_rotation(row[2:]-row[:2]))
                   for frame, row in zip(sample_frames, interpolated, strict=True)]
        for frame in (30, 40, 50):
            for region, anchor in (('former_throwing_hand', [630., 830.]), ('free_ball', [850., 610.])):
                samples.append(dict(frame=frame, sample_id=f'{frame}_{region}',
                    roi_xywh=square_roi(anchor, 160, (1080, 1920)), wrist_anchor_px=anchor,
                    wrist_gate_px=70., role='no_throwing_hand_control', control_region=region,
                    rotation_deg=0.))
        cap = cv2.VideoCapture(str(path))
        encoded = cap.get(cv2.CAP_PROP_FPS)
        header_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        clips.append(dict(clip_id=name, domain='rapsodo_linked_single_view', path=str(path),
            source_sha256=sha256(path), encoded_fps=encoded, header_frames=header_frames,
            capture_fps_nominal=500., capture_clock_independently_verified=False,
            frame_shape=[1080, 1920, 3], release_near_first=0,
            manual_inspection_hand_span_px=[70, 140], raw_wrist_distal_keyframes=keypoints,
            preprocessing_basis='Raw RGB wrist and hand direction approximations before inference; 70px gate is engineering support, not ground truth.',
            visibility='Frames 0/4/8/12/15 inspected: fingers visible but partially ball-occluded, after frame8 blurred. Frames30/40/50 throwing hand absent from both frozen control ROIs.',
            record_link={'workbook':'example-rapsodo.xlsx', 'pitch_number':4 if name.endswith('1') else 8,
                         'same_pitcher_verified':False, 'synchronized_side_video':False,
                         'measured_3d_joints':False}, samples=samples))
    model = prior/'artifacts/prime-hand-direction/hand_landmarker.task'
    gif = lab/'skenes-hand-source-audit-20261110/driveline-four-seam.gif'
    gif_keys = np.array([[8, 58, 185, 74, 161], [12, 80, 165, 103, 132],
                         [16, 109, 152, 135, 121], [20, 75, 186, 89, 168]], dtype=float)
    gif_frames = np.arange(8, 21)
    gif_points = np.array([np.interp(gif_frames, gif_keys[:, 0], gif_keys[:, j]) for j in range(1, 5)]).T
    gif_samples = [dict(frame=int(f), roi_xywh=square_roi(p[:2], 120, (207, 260)),
        wrist_anchor_px=p[:2].tolist(), wrist_gate_px=30., role='target',
        rotation_deg=upright_rotation(p[2:]-p[:2])) for f, p in zip(gif_frames, gif_points, strict=True)]
    clips.append(dict(clip_id='driveline_four_seam_demo', domain='unlinked_demo', path=str(gif),
        source_sha256=sha256(gif), encoded_fps=10., capture_fps_nominal=None,
        capture_clock_independently_verified=False, frame_shape=[207, 260, 3],
        release_near_first=16, manual_inspection_hand_span_px=[25, 45],
        raw_wrist_distal_keyframes=gif_keys.tolist(),
        preprocessing_basis='RGB wrist/hand direction frozen at F8/12/16/20; interpolation only controls crop, never fills detections.',
        visibility='Raw GIF F8/12/16/20/24 inspected: compressed small image; fingertips partly visible at F12/16; no same-pitch records, clock or 3D.',
        record_link=None, samples=gif_samples))
    manifest = dict(schema_version=1, model_path=str(model), model_sha256=sha256(model),
        preparation='Raw-image visibility review and preregistered protocol; no model output used.',
        input_side=256, num_hands=2, running_mode='IMAGE', delegate='CPU',
        min_detection_confidence=.5, min_presence_confidence=.5, min_tracking_confidence=.5,
        baseline_source=str(prior/'artifacts/prime-hand-direction/hand_direction.json'),
        anchors_sha256=sha256(anchors_file), clips=clips)
    out.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(dest), sha256=sha256(dest), samples=sum(len(c['samples']) for c in clips))))


def run(manifest_path, out):
    """Run the frozen official model, retaining rejected and ambiguous candidates."""
    import mediapipe as mp
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if sha256(manifest['model_path']) != MODEL_SHA256 or manifest['model_sha256'] != MODEL_SHA256:
        raise ValueError('Wrong official hand model')
    output = out/'inference.private.json'
    if output.exists():
        raise FileExistsError('Inference output already exists')
    cv2.setNumThreads(2)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=manifest['model_path']),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=manifest['num_hands'],
        min_hand_detection_confidence=manifest['min_detection_confidence'],
        min_hand_presence_confidence=manifest['min_presence_confidence'],
        min_tracking_confidence=manifest['min_tracking_confidence'])
    clips = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for clip in manifest['clips']:
            if sha256(clip['path']) != clip['source_sha256']:
                raise ValueError('Video source changed')
            results = dict(clip_id=clip['clip_id'], domain=clip['domain'],
                           release_near_first=clip['release_near_first'], methods={'raw':[], 'rotated':[]})
            frames = dict(decode_selected(clip['path'], [s['frame'] for s in clip['samples']]))
            for sample in clip['samples']:
                raw = frames[sample['frame']]
                for method in ('raw', 'rotated'):
                    angle = 0. if method == 'raw' else sample['rotation_deg']
                    rgb, affine = transform_input(raw, sample['roi_xywh'], angle, manifest['input_side'])
                    detected = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    candidates = []
                    for index, landmarks in enumerate(detected.hand_landmarks):
                        normalized = [[p.x, p.y, p.z] for p in landmarks]
                        world = [[p.x, p.y, p.z] for p in detected.hand_world_landmarks[index]]
                        handedness = [dict(label=p.category_name, score=p.score) for p in detected.handedness[index]]
                        candidates.append(dict(normalized21=normalized, world21=world,
                            world_frame='model-relative hand-centred coordinates in preprocessed view; not calibrated body/world',
                            pixel21=restore_pixels(normalized, affine, manifest['input_side']).tolist(),
                            handedness=handedness))
                    selected = select_candidate(candidates, sample, raw.shape)
                    results['methods'][method].append(dict(frame=sample['frame'],
                        sample_id=sample.get('sample_id', str(sample['frame'])), role=sample['role'],
                        angle_deg=angle, original_to_input=affine.tolist(),
                        candidates=candidates, selected_index=selected))
            clips.append(results)
            print(json.dumps({'finished':clip['clip_id'], 'support':summarize([results])}), flush=True)
    report = dict(schema_version=1, manifest_sha256=sha256(manifest_path), model_sha256=MODEL_SHA256,
                  mediapipe_version=mp.__version__, delegate='CPU', clips=clips)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out/'inference-summary.json').write_text(json.dumps(summarize(clips), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--lab', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.lab is None:
            parser.error('--lab is required')
        prepare(args.lab, args.out)
    else:
        if args.manifest is None:
            parser.error('--manifest is required')
        run(args.manifest, args.out)


if __name__ == '__main__':
    main()
