import {readFileSync} from 'node:fs';
import {OfflineBallReview} from '../../apps/pitch-capture/core/offline-ball.mjs';
import {releaseTrack} from '../release_conditioned_ball/track.mjs';
import {preservePath} from './preserve.mjs';

const {width,height,anchors,frames,candidates,release} = JSON.parse(readFileSync(process.argv[2],'utf8'));
const rgba = readFileSync(0);
const stride = width*height*4;
if (rgba.length !== stride*frames.length || candidates.length !== frames.length) throw new Error('Frame mismatch');
const baseline = new OfflineBallReview(width,height,...anchors.map(([x,y])=>({x,y})));
const original = baseline.tracker.candidates.bind(baseline.tracker);
const frameByTime = new Map();
const evidence = [];
for (const [index,{frame,time}] of frames.entries()) {
  frameByTime.set(time,frame);
  let app = [];
  baseline.tracker.candidates = () => (app = original());
  baseline.push(rgba.subarray(index*stride,(index+1)*stride),time);
  const round = candidates[index];
  const extra = round.filter(point => app.every(other =>
    Math.hypot(point.x-other.x,point.y-other.y)>2*width/960));
  evidence.push({frame,time,app,round,proposals:[...app,...extra]});
}
const result = baseline.finish();
const candidateByFrame = new Map(evidence.map(row=>[row.frame,row.app]));
const points = result.points.map(point=>{
  const frame = frameByTime.get(point.time);
  const original = candidateByFrame.get(frame)?.find(other=>other.x===point.x && other.y===point.y);
  if (!original) throw new Error('Could not recover the baseline observation geometry');
  return {...original,...point,frame};
});
const length = Math.hypot(anchors[1][0]-anchors[0][0],anchors[1][1]-anchors[0][1]);
process.stdout.write(JSON.stringify({proposalEvidence:evidence.map(({frame,app,round})=>({
  frame,app:app.map(({x,y})=>[x,y]),round:round.map(({x,y})=>[x,y]),
})),models:{
  app:{...result,points}, release_beam:releaseTrack(evidence,release,length,width),
  preserved:preservePath(points,evidence,release,width,length),
  anchored:preservePath(points,evidence,release,width,length,{anchorGuard:true}),
}}));
