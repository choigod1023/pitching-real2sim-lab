import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PitchDetector, comparisonError, markEvent, nearestPose, phaseTime, normalizePose} from './pitch.mjs';

const pitch = {id:'pitch-1',start:0,end:2,landing:null,release:null};
const session = {fingerprint:'original',side:'right',aspect:16/9};

test('event order and a distinct pair are required for phase comparison', () => {
  const first = markEvent(markEvent(pitch,'landing',.7),'release',1.1);
  assert.throws(() => markEvent(first,'release',.6), /착지/);
  assert.throws(() => markEvent(first,'landing',2.5), /밖/);
  assert.match(comparisonError({pitch:first,session},{pitch:{...first,id:'another'},session},true), /같은 원본/);
  const b = {pitch:{...first,id:'second'},session:{...session,fingerprint:'other'}};
  assert.match(comparisonError({pitch:first,session},b,false), /시야/);
  assert.equal(comparisonError({pitch:first,session},b,true),'');
  assert.ok(Math.abs(phaseTime(first,.5)-.9) < 1e-12);
});

test('missing or out-of-frame points remain missing in the replay', () => {
  const landmarks = Array(33).fill(null);
  landmarks[15] = {x:.2,y:.3,visibility:.4};
  landmarks[16] = {x:.2,y:.3,visibility:.9};
  landmarks[13] = {x:-.1,y:.3,visibility:.9};
  const pose = normalizePose(landmarks,[.3,.2,.5,.5]);
  assert.deepEqual(pose[16],[.4,.35]);
  assert.equal(pose[15],null);
  assert.equal(pose[13],null);
  assert.equal(nearestPose([{time:.2,points:pose}],.4),null);
});

test('a stationary player and a single erratic sample do not create a saved pitch', () => {
  const detector = new PitchDetector('right',1);
  for (let i=0;i<25;i++) {
    const points=Array(33).fill(null);
    points[12]=[.50,.45]; points[24]=[.50,.65]; points[16]=[i===10 ? .95 : .5,.4];
    detector.add(i*.05,points);
  }
  assert.deepEqual(detector.finish(1.25),[]);
});
