# Independent route after cube audit failure

Frozen after rounds 1 and 2, before solving static checkerboard poses. This adds a separate method; the original cube method remains failed, and scientific gate criteria are unchanged.

The source workbook Readme B17 gives 5 x 8 squares, 100 mm spacing (4 x 7 internal corners). Static checkerboard video frame 120, cropped and enlarged four times for detection, supplies observed corners. Enlargement supplies no new resolution. Use alternating corner indices (even) for PnP and odd indices as withheld reprojection checks. The board remains stationary; other frames only assess repeatability.

Fit pose with original checkerboard intrinsics, not cube or throwing speed. Enumerate the four grid reversals, retain positive-depth, common-facing plane normals, and resolve 180-degree grid registration using cube reprojection from the reference-camera cube pose. Cube observations participate in discrete registration, not continuous camera fitting. Preserve competing scores. Absolute lab axes and body/C3D alignment remain unverified.

Retain the original limits: <=2 px fit/withheld projection RMS, <=1% on all CS-200 lengths, positive depth. Also report all independently triangulated cube edges (698.5 mm) without adjusting them to their known lengths. Discrete registration and single-analyst cube annotation make these conditional checks, not external validation. The source CS-200 drawing is only a valid metric check if the actual image points can be identified as its specified marker mounting centres; otherwise mark that check UNKNOWN, not PASS.

No throwing 3D reconstruction is accepted until the engineering checks pass. If the checkerboard route exposes incorrect cube correspondence or target geometry assumptions, retain the failed assumptions and quantify the discrepancy before any further reconstruction.
