import {FilesetResolver, PoseLandmarker} from '@mediapipe/tasks-vision';
import {normalizePose} from '@pitch/core';

let model;
self.onmessage = async ({data}) => {
  const {id, bitmap, timestamp} = data;
  try {
    if (!model) {
      const files = await FilesetResolver.forVisionTasks(data.base + 'wasm');
      model = await PoseLandmarker.createFromOptions(files, {
        baseOptions:{modelAssetPath:data.base + 'models/pose_landmarker_full.task'},
        runningMode:'VIDEO', numPoses:1,
        minPoseDetectionConfidence:.45, minPosePresenceConfidence:.5, minTrackingConfidence:.5,
      });
    }
    const result = model.detectForVideo(bitmap, timestamp);
    self.postMessage({id, points:normalizePose(result.landmarks[0], data.region)});
  } catch (error) {
    self.postMessage({id, error:error.message || String(error)});
  } finally {
    bitmap?.close();
  }
};
