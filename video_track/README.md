# video_track: ball tracking and hand-to-ball geometry from OBP-CV

Measures where the ball is, and how far it is from the hand, in the public OBP-CV recording of one throw (`OBP_movement_BaseballThrow_001`, 62.7 mph, eight synchronized OptiTrack Prime cameras at 360 fps). The distance is then used as the wrist-to-ball distance of the five session-3034 pitches. **The radar speed is never used to choose any value**, only to compare.

## Data you need (not included)

- OBP-CV videos and Theia3D output (`OBP CV Files/compressed_optitrack`, `OBP CV Files/theia3d_output`), see the links on https://openbiomechanics.org
- The OBP repository (`computer_vision/calibration/results`, `computer_vision/4d_reconstruction/theia_alignment.json`) and the OBP pitching data (`baseball_pitching/data`, with C3D files)

## Run

From the repository root (replace the placeholders):

```bash
python -m video_track.track_ball --cv-root "OBP CV Files" --cal-dir <obp>/computer_vision/calibration/results \
  --theia-c3d "OBP CV Files/theia3d_output/c3d/OBP_movement_BaseballThrow_001.c3d" \
  --alignment <obp>/computer_vision/4d_reconstruction/theia_alignment.json --out out/track
python -m video_track.hand_ball_distance --track out/track/track.json --theia-c3d <same c3d> --alignment <same alignment> --out out/track
python -m video_track.error_budget --obp-data <obp>/baseball_pitching/data --track out/track/track.json --theia-c3d <same c3d> --alignment <same alignment> --out out
python -m video_track.simulate_grip --obp-data <obp>/baseball_pitching/data --work-dir out/sim
python -m unittest test_video_track
```

### 3D hand-relative ball motion

`relative_motion` extends the earlier `hand_ball_observation.py` / `relative_ball_state.py`
diagnostics to the new eight-camera track, including Theia hand **rotation**, not just
wrist-to-ball distance. It decomposes observed velocity into translation, rotation and
relative motion. It does not predict unseen ball motion or validate a finger-contact model.

```bash
python -m video_track.relative_motion --track video_track/results/track.json \
  --theia-c3d <same c3d> --alignment <same alignment> --out artifacts/hand-ball-relative
python -m video_track.render_relative_motion --analysis artifacts/hand-ball-relative \
  --cv-root "OBP CV Files" --cal-dir <obp>/computer_vision/calibration/results \
  --out artifacts/hand-ball-relative
python -m unittest test_video_track test_relative_motion
```

The initial local ball vector uses only frames 956–963 and is frozen for the diagnostic
evaluation at 964–973. These are analysis windows, **not established contact/release labels**.
All C3D offsets 0/1/2 are reported; offset 1 is the contributor's nominal alignment, not
selected by residual or radar speed. Full observed reconstruction is an algebraic identity.
The supplied alignment is contributor-fitted, trial-specific and provisional. The backward
ball track also used a hand-origin prior for its search windows. Full results/NPZ/video
are generated locally, and a compact result is in `results/relative_motion_summary.json`.
See [the Korean findings](RELATIVE_MOTION_FINDINGS.md) for interpretation and limitations.

## Modules

| File | Purpose |
|---|---|
| `camera.py` | Load the published pinhole calibration, project and triangulate |
| `detect.py` | Read video frames and find the ball as a bright blob near an expected pixel |
| `theia.py` | Theia hand origin placed in the camera frame with the published trial alignment |
| `track_ball.py` | Forward tracking from three image-inspected seeds, then backward into the hand-contact frames |
| `hand_ball_distance.py` | Distance ball to hand origin per frame, ball speed profile |
| `pitches.py` | The five 3034 pitches: markers, release time, recorded speed |
| `error_budget.py` | How the 3034 bias changes with release frame, release time and filter |
| `simulate_grip.py` | Force-based rollout of the observed arm with different grip distances |

## Results (`results/`)

- **Tracking:** frames 956 to 1003, 4 to 8 cameras per frame, reprojection error median 1.55 px (95 percent: 3.9 px). Ball speed 97.0 to 101.6 km/h depending on the fitting window (99.1 km/h linear fit over frames 974 to 1003), recorded 100.9 km/h. The quadratic fits give accelerations of 28 to 51 m/s^2 instead of about 10, so the fit noise (3.6 to 4.8 mm) limits the speed accuracy to a few km/h.
- **Hand to ball:** about 14 cm at frame 956, 17 to 19 cm at frames 972 to 974, 19.3 cm at 974 and 21.1 cm at 975 (C3D offset 1). With the other C3D frame offsets (0 and 2) the values differ by 1 to 2 cm near release and by up to 6 cm in the early contact frames, so the early-frame distance is not reliable; the Theia-to-camera alignment also has a median residual of 1.7 cm.
- **Error budget (5 pitches, bias of the marker proxy against radar):** fixed 15 cm gives -17.3 km/h; with the video-derived distance the bias is -10 to +1 km/h for release frames 973 to 975, and one frame of release time moves it by 6 to 10 km/h.
- **Five-pitch simulation** (error against radar, km/h):

| Pitch | Radar | 15.0 cm | 19.3 cm | 20.0 cm | 21.1 cm |
|---|---|---|---|---|---|
| 3034_1 | 130.2 | 113.7 (-16.5) | 127.7 (-2.4) | 130.1 (-0.1) | 133.7 (+3.5) |
| 3034_2 | 131.0 | 114.4 (-16.6) | 127.8 (-3.2) | 130.0 (-1.0) | 133.4 (+2.4) |
| 3034_3 | 129.2 | 113.1 (-16.1) | 126.3 (-2.9) | 128.5 (-0.7) | 132.0 (+2.7) |
| 3034_4 | 128.4 | 114.4 (-14.1) | 127.9 (-0.5) | 130.2 (+1.7) | 133.7 (+5.3) |
| 3034_5 | 128.6 | 109.2 (-19.4) | 121.6 (-7.0) | 123.7 (-4.9) | 126.9 (-1.7) |

Mean bias: -16.6 (15 cm), -3.2 (19.3 cm), -1.0 (20.0 cm), +2.4 km/h (21.1 cm). After removing the mean bias the error is 1.5 to 1.7 km/h, worse than the 0.89 km/h of simply predicting the mean recorded speed.

## Limits

- One throw of one athlete (about 101 km/h), applied to a different athlete throwing about 130 km/h. Hand size, grip and finger action may differ.
- The hand origin is Theia's estimated wrist, aligned to the cameras only for this trial. The release frame cannot be identified from the data alone; the ball leaves the hand over several frames.
- Camera synchronization was not validated. The calibration is provisional.
- The seeds of the forward track are image-inspected pixel positions of cameras 19 and 22 at frames 980, 981 and 985.
- Results are development results on seen data. They show that the average speed level can be reproduced with a measured distance, not that changes of about 1 km/h can be predicted.

Data derived from OBP-CV and OBP is CC BY-NC-SA 4.0, see `NOTICE.md`.
