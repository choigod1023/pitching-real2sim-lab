# Post-correction diagnostic: cross-recording background coordinates

Cube-only extrinsics improved excluded cube midpoints but worsened the separate
checkerboard. Before choosing camera movement as the cause, test unchanged
background image coordinates across recordings of the same camera.

Compare cube frame0 with checkerboard1013 and baseball940 in each17/20/21.
Use SIFT3000 with the central rectangle x350..1000,y200..650 masked out in both
images to exclude most person/target content. Mutual identity is not assumed:
Lowe ratio0.75, cv2 homography RANSAC2px, seed0, max2000iterations. Report total
features/matches/inliers, raw matched displacement median/p95 and residual after
the fitted homography. Backgrounds are not a camera-pose ground truth. Small
displacements constrain gross image change in matched regions but do not prove
correct intrinsics, physical scale, target centre definitions, time or anatomy.
No homography or camera correction is applied to the pitching model.
