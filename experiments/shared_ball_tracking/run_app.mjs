import {readFileSync} from 'node:fs';
import {OfflineBallReview} from '../../apps/pitch-capture/core/offline-ball.mjs';

const config = JSON.parse(readFileSync(process.argv[2],'utf8'));
const {width,height,anchors,frames,candidates} = config;
const rgba = readFileSync(0);
const stride = width*height*4;
if (rgba.length !== stride*frames.length || candidates.length !== frames.length) throw new Error('Frame mismatch');
const run = (alternative) => {
  const review = new OfflineBallReview(width,height,...anchors.map(([x,y])=>({x,y})));
  const frameByTime = new Map();
  for (const [index,{frame,time}] of frames.entries()) {
    frameByTime.set(time,frame);
    if (alternative) review.tracker.candidates = () => candidates[index];
    review.push(rgba.subarray(index*stride,(index+1)*stride),time);
  }
  const result = review.finish();
  return {...result,points:result.points.map(point=>({...point,frame:frameByTime.get(point.time)}))};
};
process.stdout.write(JSON.stringify({app:run(false),round_contrast:run(true)}));
