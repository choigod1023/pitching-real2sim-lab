import assert from 'node:assert/strict';
import test from 'node:test';
import {preservePath} from './preserve.mjs';

const point = (frame,y=50) => ({x:40+10*frame,y,s:10+10*frame,v:y-50,frame,time:frame/60});
const original = (frame,y=50,usedForFit=true) => ({...point(frame,y),usedForFit});
const video = selector => Array.from({length:12},(_,frame)=>{
  const observations = selector(frame);
  return {frame,time:frame/60,app:observations,round:observations};
});

test('removes definite pre-contact output while keeping fitted and uncertain points', () => {
  const baseline = [original(0),original(1),original(3,66),original(4),original(5),original(6)];
  const frames = video(frame=>[point(frame)]);
  const result = preservePath(baseline,frames,[1,2],960,200);
  assert.deepEqual(result.changed.suppressedPreRelease,[0]);
  for (const old of baseline.slice(1)) assert.deepEqual(result.points.find(p=>p.frame===old.frame),old);
});

test('fills a gap with an actual candidate after matching separate future frames', () => {
  const frames = video(frame=>[point(frame)]);
  const baseline = [original(1),original(2),original(4),original(5),original(6),original(7),original(8)];
  const result = preservePath(baseline,frames,[1,2],960,200);
  assert.deepEqual(result.changed.added,[3]);
  const added = result.points.find(p=>p.frame===3);
  assert.equal(added.x,point(3).x);
  assert.equal(added.futureHits,3);
});

test('changes an unfit existing point only when later observations support a closer path', () => {
  const baseline = [original(2,70,false),original(3,70,false),...Array.from({length:6},(_,i)=>original(4+i))];
  const frames = video(frame=>[point(frame)]);
  const result = preservePath(baseline,frames,[0,1],960,200);
  assert.ok(result.changed.replaced.includes(2));
  assert.equal(result.points.find(p=>p.frame===2).y,50);
  const sparse = video(frame=>frame<=3 ? [point(frame)] : []);
  assert.equal(preservePath(baseline,sparse,[0,1],960,200).points.find(p=>p.frame===2).y,70);
});

test('abstains when separate paths have equal corroboration', () => {
  const frames = video(frame=>[point(frame,45),point(frame,65)]);
  const result = preservePath([],frames,[1,2],960,200);
  assert.deepEqual(result.points,[]);
  assert.deepEqual(result.changed.ambiguous,[3,4,5,6,7]);
});
