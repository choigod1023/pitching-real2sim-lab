# Follow-up: observation-point definition versus joint movement

After rejection of the free-arm region IK, test a separate explanation: the
visible surface region and the exported joint origin can have different point
definitions. Do not change skeletal joint positions, hand orientation, timing,
ball prediction, camera calibration or existing labels.

Use the same12 fitting regions and40 held-out regions as the primary protocol.
Fit one constant local-frame observation offset for the elbow (l_larm frame) and
one for the wrist (l_hand frame), six parameters total. Use source rotation at
video+1. Each local axis is bounded +/-30mm, an engineering stress range, not
measured skin thickness or an anatomical prior. One zero initialization,
least_squares with max_nfev1000 and xtol/ftol/gtol1e-12. Objective is normalized
outside-region image residual plus sqrt(.001) times local offset/10mm.

Record all fit/evaluation cameras and +/-3px boundary sensitivity. Do not choose
this as the correct point definition merely because it fits. Acceptance for this
observation diagnostic requires lower held-out mean, nonworsening every camera
and every boundary sensitivity, and no bound saturation. Even acceptance would
not establish actual bone-centre accuracy. Source B speed stays unchanged by
construction and must be recomputed/checked. This experiment tests explanation,
not a speed-improving athlete intervention or a replacement for hand-ball physics.

The split remains development data; previous held-out results informed the new
hypothesis. No post-result label or penalty changes are permitted.
