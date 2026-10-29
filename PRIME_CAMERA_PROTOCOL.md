# Prime calibration audit before changing the pitching model

Reuse upstream camera intrinsics/pose graph unchanged. Raw checkerboard17/20/21
downloads must match upstream SHA256. Cube videos use observed source hashes;
the publisher did not provide an independent cube-video checksum. Dataset index
documents100mm checkerboard squares and698.5mm outer red-centre cube edges with
349.25mm midpoint spacing. Source capture synchronization is documented, but the
360Hz capture-to-export clock and Theia offset remain separate unknowns.

## Unchanged-rig measurements

Before detection, choose checkerboard evaluation frames
1013,1413,1813,2213,2613,3013,7413,7813,8213. They are not stride40 intrinsic-fit
samples, though nearby poses are correlated and this is not an independent
capture session. Detect7x4 inner corners using the upstream algorithm. Cache
successes/failures. Keep28-point order and explicitly test both180-degree order
possibilities by reprojection only, never by reconstructed length. Do not choose
a camera/timing change from speed. Report every pair, every frame and failure.
Triangulate using upstream fixed camera matrices, check positive depth, report
reprojection and known100mm edge lengths plus diagonals. No new calibration fit
is implicit in a small reprojection residual. Board lengths were used upstream;
this is a new-view consistency check, not an independent target-scale validation.

For the independent static cube, inspect raw frame0 before model projections.
Identify red joint components with frozen HSV hue<12 or>168, saturation>80,
value>35, connected area2..500 and fixed crop ROIs. Define visible outer corners
and mid-edge labels from the raw image topology. Record uncertain correspondence
instead of silently choosing by known length. Freeze all marker/edge associations
before triangulation or PnP. Start17/20 whose faces can be associated visually;
camera21 may require explicit alternative cube symmetries, all reported. Do not
call camera21 correspondence ground truth if its symmetry is unresolved.
Raw cube-marker centre/surface ambiguity remains, even though physical edge
definitions are documented. Original cube marker dimensions have manufacturing
and annotation uncertainty, not a supplied statistical CI.

No code retunes the athlete's root, strength, grip, release or recorded speed.
Any corrective calibration requires a separately frozen follow-up protocol with
fit/evaluation marker separation and actual held-out geometry validation. Do not
apply provisional correction to pitching before those checks. Existing data gate
criteria/state remain unchanged; no actual athlete effect is inferred.
