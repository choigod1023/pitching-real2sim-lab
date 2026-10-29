# Follow-up calibration correction, frozen after unchanged-rig cube failure

The original rig yields outer cube-edge MAE28.884mm in17/20 despite0.712px
reprojection. Do not choose a physical scale from pitching velocity. Test one
explicit correction: solve each camera's extrinsics from the8 cube outer corners
using documented698.5mm cube geometry and fixed upstream intrinsics/zero distortion.
Camera21 correspondence uses the previously selected top-pixel-only cyclic shift1;
keep all previous alternatives in the baseline artifact. There is no intrinsics,
body, timing, ball or strength fit. A cube reference frame replaces camera19 here;
it is not automatically compatible with the Theia alignment.

One cv2.solvePnP ITERATIVE per camera, no evaluation-driven solver/model choice.
Hold out each of4 vertical midpoints from extrinsic fitting. Before/after
midpoint reprojection uses one rigid cube-chart-to-old-rig registration derived
from the8 triangulated outer corners; scale stays1. Report the frame change.
Also triangulate midpoints and report349.25mm half-edge errors. Outer698.5mm
edges were used for the correction, so their post-fit error is not independent
scale validation. Re-evaluate the9 predeclared checkerboard frames with the same
28-corner orientation decisions as baseline; retain all failures.

Adoption requires positive depth, nonworsening held-out midpoint reprojection
in every camera, nonworsening half-edge MAE in every camera pair, and nonworsening
checkerboard reprojection and100mm-edge MAE in every camera pair. These are
engineering consistency checks, not clinical accuracy/uncertainty or a data-gate
replacement. Do not apply the new camera matrices to pitching unless all checks
pass and coordinate/time/body alignment is separately validated. Report a
cross-target conflict if any metric worsens; do not tune scale/labels to remove it.
