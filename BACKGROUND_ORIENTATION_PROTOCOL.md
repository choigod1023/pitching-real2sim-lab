# Independent background orientation diagnostic

Motivation: board association reduced native/source training RMS to5.199/2.316px, yet frozen cube correspondence phase1 failed. Testing all cube phases conditionally gave substantially different reprojections. No phase is accepted from its known-length agreement.

Before running, freeze an independent image-based diagnostic using static room features rather than cube/CS-200 corners or moving board frames. Use source static cube images at frames0/120/240. Mask the whole cube region in each view. Camera1–3 is the target pair;1–4 is the already fitted control.

- SIFT up to5000 features, bidirectional Lowe ratio0.75, mutual matches only; no target pixel coordinates fit pose.
- Original fixed intrinsics undistort points to normalized rays. RANSAC essential estimation, probability0.999, threshold equivalent to1pixel at the mean focal length; recover relative rotation and translation **direction**, not metric scale.
- Report every pair/frame, raw/mutual matches, essential inliers, positive-depth cheirality and homography support. Require >=20mutual matches,>=20essential inliers,>=15cheirality inliers for a descriptive rotation comparison. If >=90%mutual matches support a single homography, mark planar degeneracy; do not accept the orientation from that case alone.
- Compare rotation with all four unchanged cube-PnP horizontal phases and with the board-derived rig. Do not use cube lengths or radar to select SIFT parameters. Replicate frames from stationary cameras are repeatability checks, not independent athlete observations.
- Even a valid rotation comparison does not solve metric scale, native clock, CS-200 identities, same-pitch speed definitions, scientific data gate or biomechanical prediction. Failed/ambiguous cases remain reported.
