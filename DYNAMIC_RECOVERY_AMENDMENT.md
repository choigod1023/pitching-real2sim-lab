# Recovery amendment after detector failure

The predeclared 4x crop detector found only 3/47 camera1 frames and 10/47 camera4 frames. Their 3-arm same-frame intersection had zero frames, so no stereo fit was possible. Original detections and empty split are retained. This is a failed processing arm, not a lack of public video data.

An actual camera1 frame0 probe tested a larger border. Native adjacent corner spacing was 35.90–40.72px. Increasing crop margin from30 to91px did **not** make 4x SB detect the board. On the same 91px crop, scale1 and scale2 detected it, while scale4 did not. Therefore border clipping alone is not established as the cause. Explicit 4x enlargement plus `CALIB_CB_ACCURACY` can fail on this observed target; the OpenCV implementation's precise failure mechanism has not been isolated.

The amended arm uses a fixed 2x enlargement, with margin `2*max_observed_corner_spacing+10px` to keep the surrounding board/white border. The rule is applied to every frame without looking at calibration fit, known lengths or speed. Baseline and native observations are reused byte-for-byte, not recomputed or moved. Preserve original 4x failures. Use the same frozen temporal train/heldout rule and >=8/5/2 counts; do not change metric thresholds. Report this as an evidence-driven recovery experiment, not the originally frozen 4x arm.

Official guidance: SB directly returns subpixel corners and `CALIB_CB_ACCURACY` itself upsamples; an adequate white border helps detection. [OpenCV reference](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html). No claim that enlargement creates new information or fixes physical speed accuracy.
