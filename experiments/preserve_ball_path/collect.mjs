import {readFileSync} from 'node:fs';
import {OfflineBallReview} from '../../apps/pitch-capture/core/offline-ball.mjs';

export function collectReview(configPath, rgba) {
  const {width,height,anchors,frames,candidates,release} = JSON.parse(readFileSync(configPath,'utf8'));
  const stride = width*height*4;
  if (rgba.length !== stride*frames.length || candidates.length !== frames.length) throw new Error('Frame mismatch');
  const review = new OfflineBallReview(width,height,...anchors.map(([x,y])=>({x,y})));
  const detect = review.tracker.candidates.bind(review.tracker);
  const frameByTime = new Map();
  const evidence = [];
  for (const [index,{frame,time}] of frames.entries()) {
    frameByTime.set(time,frame);
    let app = [];
    review.tracker.candidates = () => (app = detect());
    review.push(rgba.subarray(index*stride,(index+1)*stride),time);
    const round = candidates[index];
    const extra = round.filter(point => app.every(other =>
      Math.hypot(point.x-other.x,point.y-other.y)>2*width/960));
    evidence.push({frame,time,app,round,proposals:[...app,...extra]});
  }
  const result = review.finish();
  const candidateByFrame = new Map(evidence.map(row=>[row.frame,row.app]));
  const points = result.points.map(point=>{
    const frame = frameByTime.get(point.time);
    const original = candidateByFrame.get(frame)?.find(other=>other.x===point.x && other.y===point.y);
    if (!original) throw new Error('Could not recover the baseline observation geometry');
    return {...original,...point,frame};
  });
  const length = Math.hypot(anchors[1][0]-anchors[0][0],anchors[1][1]-anchors[0][1]);
  const proposalEvidence = evidence.map(({frame,app,round})=>({
    frame,app:app.map(({x,y})=>[x,y]),round:round.map(({x,y})=>[x,y]),
  }));
  return {app:{...result,points},evidence,proposalEvidence,release,width,length};
}
