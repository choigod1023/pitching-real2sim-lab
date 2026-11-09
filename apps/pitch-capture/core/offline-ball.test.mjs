import test from 'node:test';
import assert from 'node:assert/strict';
import {OfflineBallReview} from './offline-ball.mjs';

function recording({metres = null, startX = 122, step = 21, range = [135,303], gates = null} = {}) {
  const width = 640, height = 360;
  const onPath = x => ({x,y:180});
  const [a,b] = range.map(onPath);
  const review = new OfflineBallReview(width,height,a,b,metres,{gates:gates?.map(onPath)});
  for (let i=0; i<=10; i++) {
    const pixels = new Uint8Array(width*height*4);
    if (i > 0) {
      const x = startX+(i-1)*step;
      for (let row=177; row<=183; row++) for (let col=x-3; col<=x+3; col++) {
        if ((row-180)**2+(col-x)**2 <= 9) pixels.fill(255,4*(row*width+col),4*(row*width+col)+4);
      }
    }
    review.push(pixels,i/60);
  }
  return review.finish();
}

test('recorded centres and times support calibrated gate-to-gate speed without extrapolation', () => {
  const result = recording({metres:4});
  assert.equal(result.frameCount,11);
  assert.equal(result.points.length,10);
  assert.equal(result.points[0].time,1/60);
  assert.equal(result.fit.kind,'linear');
  assert.ok(result.release.before < result.release.estimate && result.release.estimate < result.release.after);
  assert.equal(result.speedStatus,'available');
  assert.ok(Math.abs(result.kmh-108) < 1e-4);
  assert.ok(result.speedBoundsKmh[0] < result.kmh && result.kmh < result.speedBoundsKmh[1]);
});

test('an uncalibrated or unobserved gate cannot yield km/h', () => {
  const uncalibrated = recording();
  assert.equal(uncalibrated.speedStatus,'uncalibrated');
  assert.equal(uncalibrated.kmh,null);
  assert.ok(uncalibrated.imageSpeedPxS > 0);
  const firstGateMissed = recording({metres:4,startX:146});
  assert.equal(firstGateMissed.release,null);
  assert.equal(firstGateMissed.speedStatus,'unbracketed_gates');
  assert.equal(firstGateMissed.kmh,null);
});

test('a recording without a coherent ball track retains the frame count and withholds metrics', () => {
  const review = new OfflineBallReview(640,360,{x:135,y:180},{x:303,y:180},4);
  const blank = new Uint8Array(640*360*4);
  for (let i=0; i<4; i++) review.push(blank,i/60);
  const result = review.finish();
  assert.equal(result.frameCount,4);
  assert.deepEqual(result.points,[]);
  assert.equal(result.speedStatus,'too_few_centres');
  assert.equal(result.release,null);
});

test('wide detection retains fast samples when the calibrated segment is too short for tracking', () => {
  const input = {startX:80,step:60,metres:1.3};
  const coupled = recording({...input,range:[165,295]});
  const separated = recording({...input,range:[100,510],gates:[165,295]});
  const longer = recording({...input,range:[100,510],gates:[165,415],metres:2.5});
  assert.equal(coupled.kmh,null);
  assert.ok(Math.abs(separated.kmh-129.6) < 1e-4);
  assert.deepEqual(separated.points,longer.points);
  assert.equal(separated.baseline.kmh,null);
  assert.ok(Math.abs(longer.kmh-129.6) < 1e-4);
  assert.equal(separated.measurement.start.estimate,longer.measurement.start.estimate);
  const [shortMin,shortMax] = separated.speedBoundsKmh;
  const [longMin,longMax] = longer.speedBoundsKmh;
  assert.ok(longMax-longMin < shortMax-shortMin);
});

test('ordered measurement points must be in the detection corridor and separately bracketed', () => {
  const input = {startX:80,step:60,range:[100,510],metres:1};
  assert.throws(() => recording({...input,gates:[300,200]}),/순서대로/);
  assert.throws(() => recording({...input,gates:[200,600]}),/검출 범위/);
  const missed = recording({...input,gates:[200,550]});
  assert.equal(missed.speedStatus,'unbracketed_gates');
  assert.equal(missed.measurement.finish,null);
  assert.equal(missed.kmh,null);
  const adjacent = recording({...input,gates:[165,225]});
  assert.equal(adjacent.speedStatus,'overlapping_gate_brackets');
  assert.ok(adjacent.measurement.finish.estimate > adjacent.measurement.start.estimate);
  assert.equal(adjacent.kmh,null);
});
