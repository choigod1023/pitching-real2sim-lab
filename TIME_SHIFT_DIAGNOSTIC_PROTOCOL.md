# +/-1 frame engineering diagnostic

Frozen after the0-shift per-frame-order model failed. Run camera3 frame f+shift against camera1 f for shifts-1,0,+1. Do not interpret this as capture uncertainty bounds or pick a shift by cube/radar agreement.

For each detector use the **same camera1 frame IDs in all three shifts**, intersecting its valid points with camera3 validity at all f-1/f/f+1. The original540..612 acquisition bounds apply. Keep modulo5 heldout rule and >=8/5/2 minimum counts; insufficient cases stay insufficient. Fit the training-only correspondence recovery identically for every shift, keep each output, and report withheld board reprojection and all conditional static geometry. Keep original cube phase1 as primary; phase3 is an explicitly unvalidated correspondence diagnostic. No shift is automatically adopted. Data gate remains unpassed.
