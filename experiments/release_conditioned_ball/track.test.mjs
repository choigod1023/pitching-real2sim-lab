import assert from 'node:assert/strict';
import test from 'node:test';
import {releaseTrack} from './track.mjs';

const point = (frame, y, s = 10+(frame-4)*9) => ({x:80+s,y,s,v:y-50,diameter:5,strength:100});
const sequence = fn => Array.from({length:12},(_,frame)=>({frame,time:frame/60,proposals:fn(frame)}));

test('the contact bracket excludes early distractors and preserves observations after release', () => {
  const frames = sequence(frame => frame<4 ? [point(frame,61,frame*6)] :
    frame<=10 ? [point(frame,50)] : []);
  const result = releaseTrack(frames,[2,3],160,960);
  assert.equal(result.status,'observed');
  assert.deepEqual(result.points.map(p=>p.frame),[4,5,6,7,8,9,10]);
  assert.equal(result.points.every(p=>p.y===50),true);
});

test('two equally supported paths leave their disputed coordinates blank', () => {
  const frames = sequence(frame => frame>=4 && frame<=10 ? [point(frame,40),point(frame,60)] : []);
  const result = releaseTrack(frames,[2,3],160,960);
  assert.equal(result.status,'ambiguous');
  assert.equal(result.points.length,0);
  assert.equal(result.hypothesisCount>=2,true);
  assert.deepEqual(result.ambiguousFrames,[4,5,6,7,8,9,10]);
});

test('a skipped observation is not inserted and short tracks are insufficient', () => {
  const frames = sequence(frame => frame>=4 && frame<=10 && frame!==6 ? [point(frame,50)] : []);
  const result = releaseTrack(frames,[3,3],160,960);
  assert.deepEqual(result.points.map(p=>p.frame),[4,5,7,8,9,10]);
  assert.equal(releaseTrack(frames.slice(0,8),[3,3],160,960).status,'insufficient_support');
  assert.throws(()=>releaseTrack(frames,[4,3],160,960),/ordered release/);
});
