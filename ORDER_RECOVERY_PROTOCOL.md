# Recovery after fixed-order stereo failure

The dense native detector supplied24 common frames (19train/5heldout), but all four **global** grid reversal choices failed (best training RMS18.208px). The source detector supplied23frames (20/3), best16.405px. These failed calculations remain intact.

Public upstream `computer_vision/calibration/README.md` explicitly warns that checkerboard180-degree ambiguity makes trusting every frame's corner order unsafe. Representative frame576 images show different image-grid origins across views; this alone does not establish the correct physical correspondence or timing.

Before testing the next model, freeze a training-only correspondence recovery: enumerate per-frame grid reversals; construct candidate rig transforms from each training frame's PnP; choose initialization by the median training image reprojection residual, never static target lengths or radar. Alternate training frame association and fixed-intrinsic stereo fitting for at most5 iterations. Use the trained rig and camera1-only PnP to associate withheld frame indices; report all winning/runner-up image costs and describe this data association limitation. No cube or speed chooses a frame orientation. Original frame IDs, modulo5 split and >=8/5/2 counts remain unchanged.

This is a new method after the failed global-order method. Test exactly the same heldout board and unchanged static target coordinates. All original <=2px,<=1%,positive-depth checks still apply. Lower training error alone cannot accept a rig. A conditional recovered link does not establish absolute lab axes, capture timing, CS-200 point identities or athlete benefits.

After this method also failed metric checks, compare all four horizontal cube correspondence rotations as a **diagnostic**, preserving the original phase1 result. This tests whether provisional cube correspondence conflicts with board-derived camera geometry. Report all phases and do not select by matching known lengths. Any alternative phase requires image/topology verification and remains conditional because cube observations participate in registration.
