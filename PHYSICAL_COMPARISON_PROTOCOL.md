# Exploratory physical continuation, fixed before its own results

After inspecting the first kinematic comparison, its test interval contains separated-ball outlines, so continuously attaching the ball to the future hand is not an adequate free-flight baseline. Do not present that improvement as a physics-model gain. Run this additional control without selecting parameters from its results:
- Forecast start is frame976, the last training frame, not a verified release event.
- A_release: evaluate fitted constant-local-offset model on all four TRAINING hand states. Initial position is its final position, initial velocity the finite difference of its last two training positions.
- B_release: evaluate fitted affine-local-offset model on all four TRAINING hand states. Same position/velocity rule; no evaluation hand state is consumed.
- C_release: initial position and velocity from the training-only affine world trajectory. Same forecasting horizon.
- Each is a145g free ball in MuJoCo with RK4, gravity=provided alignment rotation times lab[0,0,-9.81], no drag/contact, no state resets. Lab Z-up is a stated engineering assumption; this is not release identification. Integrate1/360s per frame; compare dt andhalfdt for convergence. Evaluate same future frames977-979 and report all9 settings, same default80/+1. No radar fit.
- This checks observed initial-state correction followed by physical integration, not strength-matched full athlete simulation or actual training gain. Record all outcomes even if A/B worsens.
