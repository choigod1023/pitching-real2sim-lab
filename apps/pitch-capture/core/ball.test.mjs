import assert from 'node:assert/strict';
import test from 'node:test';
import {BallTracker, ballMetrics} from './ball.mjs';

const width = 640, height = 240;
function image(x, y, phase = null) {
  const rgba = new Uint8ClampedArray(width*height*4).fill(24);
  if (x === null) return rgba;
  for (let row = y-11; row <= y+11; row++) {
    for (let col = x-11; col <= x+11; col++) {
      if ((col-x)**2+(row-y)**2 > 121) continue;
      const i = (row*width+col)*4;
      const blue = phase !== null && (col-x-7*Math.cos(phase))**2+(row-y-7*Math.sin(phase))**2 < 6;
      rgba[i] = blue ? 30 : 232;
      rgba[i+1] = blue ? 62 : 232;
      rgba[i+2] = blue ? 206 : 232;
    }
  }
  return rgba;
}

test('image-based tracker uses observed moving centers and recovers marked-ball RPM', () => {
  const tracker = new BallTracker(width,height,{x:80,y:100},{x:510,y:100},{metres:2.15,markedBall:true});
  tracker.push(image(null),0);
  for (let i = 1; i <= 13; i++) tracker.push(image(55+28*i,100,i*Math.PI/4),i/240);
  const result = tracker.visible;
  assert.equal(result.points.length,13);
  assert.ok(Math.abs(result.kmh-120.96) < .4, JSON.stringify(result));
  assert.ok(Math.abs(result.rpm-1800) < 22, JSON.stringify(result));
  assert.ok(Math.abs(result.bendCm) < .4);
  const held = tracker.push(image(null),14/240);
  assert.equal(held.updatedAt,13/240);
  assert.equal(tracker.push(image(null),2),null);
});

test('low sample rate, absent marker, and absent scale leave metrics unavailable', () => {
  const tracker = new BallTracker(width,height,{x:80,y:100},{x:500,y:100},{markedBall:true});
  tracker.push(image(null),0);
  for (let i = 1; i < 10; i++) tracker.push(image(53+35*i,100,i*Math.PI/4),i/60);
  assert.ok(tracker.visible.points.length >= 8);
  assert.equal(tracker.visible.kmh,null);
  assert.equal(tracker.visible.rpm,null);
  const points = [0,1,2,3,4].map(i => ({x:100+i*10,y:i===2?105:100,time:i/60}));
  assert.equal(ballMetrics(points,.01).rpm,null);
  assert.equal(ballMetrics(points,.01).bendCm,5);
  assert.equal(ballMetrics(points.map(p => ({...p,x:200-p.x})),.01).bendCm,5);
});

test('a fixed bright blob and a one-frame flash do not create a pitch path', () => {
  const tracker = new BallTracker(width,height,{x:80,y:100},{x:500,y:100});
  for (let i = 0; i < 12; i++) tracker.push(i === 6 ? image(250,110) : image(110,100),i/60);
  assert.equal(tracker.visible,null);
});
