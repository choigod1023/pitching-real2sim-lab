# Post-primary diagnostic (no refitting)

The primary region reconstruction reduced ball-speed discrepancy but worsened
held-out image regions and saturated one root bound. It is rejected. Diagnose
without modifying regions, fitted parameters or regularization:

1. Separate held-out camera17 at fitting times from other early frames and
release frames. This tests temporal transfer, not a new fit.
2. Check simultaneous 3D region feasibility at new940/960 observations with
existing camera17 using the existing LP; no joint length prior. Minimal common
padding is an engineering compatibility number, not an error interval.
3. At the unchanged973..976 ball fitting window, substitute reconstructed wrist
only with original hand rotation, then reconstructed rotation only with original
wrist. Compare to both changes and source. These are attribution controls, not
new candidates, biomechanical interventions or a selection procedure.
4. Keep existing rejection and scientific gate unchanged regardless of controls.
