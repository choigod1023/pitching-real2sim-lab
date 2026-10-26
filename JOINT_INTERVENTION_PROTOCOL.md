# Joint intervention demand-screen protocol

Frozen before this run. Target original3034_1 force-driven arm/hand/ball reference, not Prime ball tracking. Same observed initial state, timing, generic mass/inertia,15cm grip and BR event. Existing16.73km/h record gap remains unvalidated.

Question: what conditional ball-speed change follows a small joint trajectory change, and does the required force/work violate the baseline mechanical envelope?

Primary cases: no change plus signed1deg and2deg modifications in each of4model coordinates: elbow hinge(dof6), wrist marker local axesX/Y/Z(dofs8/9/10). These wristaxes are not calibrated anatomicalflexion/deviation. All17cases reported.

Window: final30ms before fixedBR. Angle correction theta(t)=angle * clip((t-(BR-.030))/.030,0,1)^3. Smooth initialzero angle/velocity/acceleration; terminalangle andangularrate change. No postBR followthrough claim. Apply quaternion changes with mj_integratePos, notcomponentaddition. No release/grip/strength search. This is a prescribed movement-demand experiment, not an autonomous learned controller.

Recompute required inverse forces for each declared modified trajectory and apply them in actualRK4 forward integration. Explicitly NOTfixedforces/equalability. Reject candidates with increasedperDOFpositive/negativework, forcepeak, jointtorquenormpeak, shouldertranslation/rotationnetwork under existingstrictmechanicalgate. Recordinitialenergy/state equality, fulltrajectory rotationdeviation andactualendpointtracking. Use max3deg actualrotationdeviation atalljoints as an engineering small-change screen, not comfort/physiologicalROM. Baselinecomparison excludesunusedterminalforce row.

Numericalcheck: same reference-derived force per sourceinterval, integrate atdt anddt/2; controlsheldconstant. This checksintegratorconvergence, notalltrajectoryderivativeuncertainty. Reportallfailures andsignedspeedchanges; none automaticallyrecommended. Existinggate is coordinate-dependent conservative screening, notvalidatedmusclecapacity.

Further target-free sensitivity: evaluate declared endpointjointchanges atgrip14/15/16cm withsamekinematicstates. Label kinematicmapping only, notsameabilitycounterfactual. Comparepaired changes within eachgrip, notabsolutebestcondition.

A ball-trajectory-only forecast with identical preintervention ballhistory has no jointinput and cannot predict these actioneffects. Do not treat itszero sensitivity as evidence that physicalmodelchanges are correct. Realinterventiontruth is unavailable, so actionpredictionvalidity remainsunverified.
