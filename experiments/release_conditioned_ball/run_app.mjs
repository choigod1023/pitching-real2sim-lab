import {readFileSync} from 'node:fs';
import {OfflineBallReview} from '../../apps/pitch-capture/core/offline-ball.mjs';
import {releaseTrack} from './track.mjs';

const config = JSON.parse(readFileSync(process.argv[2],'utf8'));
const {width,height,anchors,frames,candidates,release} = config;
const rgba = readFileSync(0);
const stride = width*height*4;
if (rgba.length !== stride*frames.length || candidates.length !== frames.length) throw new Error('Frame mismatch');

const baseline = new OfflineBallReview(width,height,...anchors.map(([x,y])=>({x,y})));
const original = baseline.tracker.candidates.bind(baseline.tracker);
const frameByTime = new Map();
const evidence = [];
for (const [index,{frame,time}] of frames.entries()) {
  frameByTime.set(time,frame);
  let raw = [];
  baseline.tracker.candidates = () => (raw = original());
  baseline.push(rgba.subarray(index*stride,(index+1)*stride),time);
  const extra = candidates[index].filter(point => raw.every(other =>
    Math.hypot(point.x-other.x,point.y-other.y) > 2*width/960));
  evidence.push({frame,time,app:raw,round:candidates[index],proposals:[...raw,...extra]});
}
const result = baseline.finish();
const points = result.points.map(point=>({...point,frame:frameByTime.get(point.time)}));
const length = Math.hypot(anchors[1][0]-anchors[0][0],anchors[1][1]-anchors[0][1]);
const run = shift => releaseTrack(evidence,release.map(frame=>frame+shift),length,width);
process.stdout.write(JSON.stringify({proposalEvidence:evidence.map(({frame,app,round})=>({
  frame,app:app.map(({x,y})=>[x,y]),round:round.map(({x,y})=>[x,y]),
})),models:{
  app:{...result,points}, mask_only:{points:points.filter(point=>point.frame>release[1])},
  release_beam:run(0), one_frame_early:run(-1), one_frame_late:run(1),
  wide_post_only:releaseTrack(evidence,release,length,width,{seedFrames:5}),
  wide_bracket:releaseTrack(evidence,release,length,width,{startInBracket:true,seedFrames:5}),
}}));
