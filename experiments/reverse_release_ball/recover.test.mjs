import assert from 'node:assert/strict';
import test from 'node:test';
import {recoverRelease} from './recover.mjs';

const point = (frame, y=10, usedForFit=true) => ({frame,time:frame/60,x:frame*6,y,s:frame*6,usedForFit});
const clip = (candidateFrames={3:[{x:19,y:10}]}) =>
  Array.from({length:9},(_,frame)=>({frame,time:frame/60,app:candidateFrames[frame] ?? []}));
const baseline = [4,5,6,7].map(frame=>point(frame));

test('returns the observed centre and preserves every existing observation verbatim', () => {
  const old = [point(1,4,false),...baseline,point(8,10,false)];
  const result = recoverRelease(old,clip(),[1,2],960);
  assert.deepEqual(result.checks.map(check=>check.status),['recovered','existing']);
  assert.deepEqual(result.points.filter(p=>p.frame!==3),old);
  assert.deepEqual(result.points[1],{x:19,y:10,frame:3,time:3/60,usedForFit:false,pathDecision:'reverse_observed'});
});

test('only fills the first two post-release frames and never synthesizes a centre', () => {
  const result = recoverRelease(baseline,clip({2:[{x:12,y:10}]}),[0,1],960);
  assert.deepEqual(result.points.map(p=>p.frame),[2,4,5,6,7]);
  assert.equal(result.checks[1].status,'no_image_candidate');
  const contact = recoverRelease(baseline,clip(),[1,3],960);
  assert.deepEqual(contact.points,baseline);
});

test('rejects a conflicting future path, a missing fit point and a distant fit', () => {
  const conflict = [point(4),point(5),point(6),point(7,16)];
  assert.equal(recoverRelease(conflict,clip(),[1,2],960).checks[0].status,'followup_mismatch');
  assert.equal(recoverRelease([point(4),point(5),point(7)],clip(),[1,2],960).checks[0].status,'nonconsecutive_fit');
  assert.equal(recoverRelease(baseline,clip(),[0,0],960).checks[0].status,'no_nearby_fit');
  const backwards = baseline.map(p=>({...p,s:-p.s}));
  assert.equal(recoverRelease(backwards,clip(),[1,2],960).checks[0].status,'inconsistent_direction');
});

test('rejects competing centres and tests the distance in width-normalized pixels', () => {
  const ambiguous = recoverRelease(baseline,clip({3:[{x:18,y:12},{x:18,y:8}]}),[1,2],960);
  assert.equal(ambiguous.checks[0].status,'ambiguous');
  const tooFar = recoverRelease(baseline,clip({3:[{x:21,y:10}]}),[1,2],480);
  assert.equal(tooFar.checks[0].status,'no_image_candidate');
});

test('uses decoded times and declines irregular frame spacing', () => {
  const evidence = clip({3:[{x:18,y:10}]});
  const times = [0,16,32,48,66,82,98,114,130].map(ms=>ms/1000);
  evidence.forEach((row,i)=>{row.time=times[i]});
  const anchors = baseline.map(p=>({...p,time:times[p.frame]}));
  const result = recoverRelease(anchors,evidence,[1,2],960);
  assert.equal(result.checks[0].status,'recovered');
  assert.ok(Math.abs(result.checks[0].nearestPx960-.75)<1e-8);
  evidence[3].time=.022;
  assert.equal(recoverRelease(anchors,evidence,[1,2],960).checks[0].status,'irregular_pts');
});
