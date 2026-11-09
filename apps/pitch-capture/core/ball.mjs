const bounded = (value, low, high) => Math.max(low, Math.min(high, value));
const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));

// All image coordinates use square pixels at the analysis resolution.
export class BallTracker {
  constructor(width, height, start, end, {metres = null, markedBall = false} = {}) {
    this.width = width;
    this.height = height;
    this.start = start;
    const dx = end.x - start.x, dy = end.y - start.y;
    this.length = Math.hypot(dx, dy);
    if (!Number.isFinite(this.length) || this.length < width * .10) {
      throw new Error('추적 구간의 두 점을 더 멀리 지정하세요.');
    }
    this.ux = dx / this.length;
    this.uy = dy / this.length;
    this.band = Math.max(12, this.length * .13);
    this.metres = Number.isFinite(metres) && metres > 0 ? metres : null;
    this.markedBall = markedBall;
    const margin = this.band + .12 * this.length;
    this.left = bounded(Math.floor(Math.min(start.x, end.x) - margin), 2, width - 3);
    this.right = bounded(Math.ceil(Math.max(start.x, end.x) + margin), 2, width - 3);
    this.top = bounded(Math.floor(Math.min(start.y, end.y) - margin), 2, height - 3);
    this.bottom = bounded(Math.ceil(Math.max(start.y, end.y) + margin), 2, height - 3);
    this.gray = new Uint8Array(width * height);
    this.previous = new Uint8Array(width * height);
    this.mask = new Uint8Array(width * height);
    this.reset();
  }

  reset() {
    this.lastTime = null;
    this.hypotheses = [];
    this.visible = null;
    this.intervals = [];
  }

  push(rgba, time) {
    if (!Number.isFinite(time) || rgba.length !== this.gray.length * 4) return null;
    if (this.lastTime !== null && time <= this.lastTime) {
      if (time < this.lastTime) this.reset();
      else return this.visible;
    }
    const interval = this.lastTime === null ? Infinity : time - this.lastTime;
    for (let y = this.top - 2; y <= this.bottom + 2; y++) {
      for (let x = this.left - 2; x <= this.right + 2; x++) {
        const index = y * this.width + x;
        this.gray[index] = (77*rgba[4*index] + 150*rgba[4*index+1] + 29*rgba[4*index+2]) >> 8;
      }
    }
    this.lastTime = time;
    if (interval <= .085) {
      this.intervals.push(interval);
      if (this.intervals.length > 12) this.intervals.shift();
      this.advance(this.candidates(rgba), time);
    } else {
      this.hypotheses = [];
      this.intervals = [];
      this.visible = null;
    }
    [this.gray, this.previous] = [this.previous, this.gray];
    return this.visible;
  }

  candidates(rgba) {
    const {width, gray, previous, mask} = this;
    mask.fill(0);
    let motion = 0;
    for (let y = this.top; y <= this.bottom; y++) {
      for (let x = this.left; x <= this.right; x++) {
        const dx = x - this.start.x, dy = y - this.start.y;
        const s = dx*this.ux + dy*this.uy, v = dx*(-this.uy) + dy*this.ux;
        if (s < -.12*this.length || s > 1.12*this.length || Math.abs(v) > this.band) continue;
        const i = y*width+x, luminance = gray[i];
        if (luminance >= 85 && luminance-previous[i] >= 17) {
          mask[i] = 1;
          motion++;
        }
      }
    }
    // A moving/zooming camera or an obstruction can light up the whole corridor.
    if (motion > 2*this.band*this.length*.12) return [];
    const candidates = [];
    const stack = [];
    for (let y = this.top; y <= this.bottom; y++) {
      for (let x = this.left; x <= this.right; x++) {
        const index = y*width+x;
        if (!mask[index]) continue;
        let count = 0, cx = 0, cy = 0, total = 0, minX=x, maxX=x, minY=y, maxY=y;
        stack.push(index);
        mask[index] = 0;
        while (stack.length) {
          const i = stack.pop(), px = i % width, py = Math.floor(i / width);
          const weight = gray[i]-previous[i];
          count++;
          cx += px*weight;
          cy += py*weight;
          total += weight;
          minX = Math.min(minX, px); maxX = Math.max(maxX, px);
          minY = Math.min(minY, py); maxY = Math.max(maxY, py);
          for (const near of [i-1,i+1,i-width,i+width]) {
            if (!mask[near]) continue;
            mask[near] = 0;
            stack.push(near);
          }
        }
        const boxW = maxX-minX+1, boxH = maxY-minY+1;
        const diameter = Math.max(boxW,boxH);
        if (count < 3 || diameter > Math.min(52, width*.065) ||
            diameter > 3*Math.min(boxW,boxH) || count/(boxW*boxH) < .27) continue;
        cx /= total;
        cy /= total;
        const dx = cx - this.start.x, dy = cy - this.start.y;
        const point = {x:cx, y:cy, s:dx*this.ux+dy*this.uy,
          v:-dx*this.uy+dy*this.ux, diameter, strength:total/count};
        if (this.markedBall && Math.min(boxW,boxH) >= 20 && boxW/boxH >= .8 && boxW/boxH <= 1.25) {
          point.marker = this.blueMarker(rgba, (maxX+minX)/2, (maxY+minY)/2, Math.min(boxW,boxH)/2);
        }
        candidates.push(point);
      }
    }
    return candidates.sort((a,b) => b.strength-a.strength).slice(0,24);
  }

  blueMarker(rgba, x, y, radius) {
    let mx = 0, my = 0, count = 0;
    for (let row = Math.floor(y-radius); row <= Math.ceil(y+radius); row++) {
      for (let col = Math.floor(x-radius); col <= Math.ceil(x+radius); col++) {
        if (row < 0 || row >= this.height || col < 0 || col >= this.width ||
            (col-x)**2+(row-y)**2 > (.91*radius)**2) continue;
        const i = 4*(row*this.width+col);
        if (rgba[i+2] < 85 || rgba[i+2]-rgba[i] < 40 || rgba[i+2]-rgba[i+1] < 25) continue;
        mx += col;
        my += row;
        count++;
      }
    }
    if (count < 4 || count > radius*radius*.55) return null;
    const dx = mx/count-x, dy = my/count-y;
    return {angle:Math.atan2(dy,dx), radial:Math.hypot(dx,dy)/radius};
  }

  advance(candidates, time) {
    const expectedInterval = this.intervals.slice().sort((a,b) => a-b)[this.intervals.length >> 1];
    const gap = bounded(expectedInterval*2.65, .014, .085);
    const next = [];
    for (const track of this.hypotheses) {
      const last = track.points.at(-1);
      const dt = time - last.time;
      if (dt > gap) continue;
      next.push(track);
      for (const point of candidates) {
        const ds = point.s - last.s;
        const dv = point.v - last.v;
        const velocity = ds/dt;
        if (ds < 1.05 || velocity < .60*this.length || velocity > 24*this.length ||
            Math.abs(dv) > 4+ds*.63) continue;
        const predictionError = track.velocity === null ? 0 :
          Math.abs(ds-track.velocity*dt) + .6*Math.abs(dv-track.lateral*dt);
        if (predictionError > Math.max(5, ds*.44)) continue;
        const sizeRatio = point.diameter/last.diameter;
        if (sizeRatio < .42 || sizeRatio > 2.4) continue;
        next.push({
          points:[...track.points,{...point,time}],
          velocity:track.velocity === null ? velocity : .68*track.velocity+.32*velocity,
          lateral:dv/dt,
          penalty:track.penalty + predictionError/(3+ds) + Math.abs(Math.log(sizeRatio))*.4,
        });
      }
    }
    for (const point of candidates) {
      if (point.s <= this.length*.42) next.push({points:[{...point,time}],velocity:null,lateral:0,penalty:0});
    }
    const rank = track => track.points.length*2 - track.penalty -
      1.1*((time-track.points.at(-1).time)/expectedInterval);
    // Keep one hypothesis per current center; stale alternatives only bridge short gaps.
    next.sort((a,b) => rank(b)-rank(a));
    const seen = new Set();
    this.hypotheses = next.filter(track => {
      const tip = track.points.at(-1), key = `${tip.time}|${tip.x}|${tip.y}`;
      if (seen.has(key) || time-track.points[0].time > 1.2) return false;
      seen.add(key);
      return true;
    }).slice(0,32);
    if (this.visible && time-this.visible.updatedAt > .85) this.visible = null;
    const best = this.hypotheses.find(track => track.points.length >= 3 && track.points.at(-1).time === time &&
      track.points.at(-1).s-track.points[0].s >= .045*this.length);
    if (best) {
      this.visible = {
        points:best.points,
        updatedAt:time,
        gap,
        ...ballMetrics(best.points, this.metres === null ? null : this.metres/this.length),
      };
    }
  }
}

function slope(samples, key) {
  const meanT = samples.reduce((sum,p) => sum+p.time,0)/samples.length;
  const meanY = samples.reduce((sum,p) => sum+p[key],0)/samples.length;
  const denom = samples.reduce((sum,p) => sum+(p.time-meanT)**2,0);
  if (!denom) return null;
  const rate = samples.reduce((sum,p) => sum+(p.time-meanT)*(p[key]-meanY),0)/denom;
  const residual = Math.sqrt(samples.reduce((sum,p) => sum+(p[key]-meanY-rate*(p.time-meanT))**2,0)/samples.length);
  return {rate,residual};
}

export function markedRPM(points) {
  const run = [];
  let accumulated = 0;
  for (const point of points) {
    const marker = point.marker;
    const last = run.at(-1);
    const dt = last ? point.time-last.time : 0;
    if (!marker || marker.radial < .33 || marker.radial > .87 ||
        (last && (dt < .002 || dt > .0051 || Math.abs(marker.radial-last.radial) > .13))) {
      run.length = 0;
      accumulated = 0;
      if (!marker || marker.radial < .33 || marker.radial > .87) continue;
    }
    if (run.length) {
      const delta = wrap(marker.angle-run.at(-1).angle);
      if (Math.abs(delta) > 2.36 || Math.abs(delta) < .045 || (run.length >= 2 && accumulated*delta <= 0)) {
        run.length = 0;
        accumulated = 0;
      } else accumulated += delta;
    }
    run.push({time:point.time, phase:accumulated, radial:marker.radial, angle:marker.angle});
    if (run.length >= 8 && Math.abs(accumulated) >= 2*Math.PI) {
      const fit = slope(run, 'phase');
      const rpm = Math.abs(fit.rate)*60/(2*Math.PI);
      const radials = run.map(p => p.radial);
      if (fit.residual <= .14 && Math.max(...radials)-Math.min(...radials) <= .15 &&
          rpm >= 300 && rpm <= 3300) return rpm;
    }
  }
  return null;
}

export function ballMetrics(points, metresPerPixel) {
  const first = points[0], last = points.at(-1);
  const fitPoints = points.filter(p => p.time-first.time <= .12).slice(0,10);
  const vx = slope(fitPoints,'x'), vy = slope(fitPoints,'y');
  const speed = vx && vy && fitPoints.length >= 4 && fitPoints.at(-1).time-first.time >= .016 &&
    Math.hypot(vx.residual,vy.residual) <= 3.5 ? Math.hypot(vx.rate,vy.rate) : null;
  const dx = last.x-first.x, dy = last.y-first.y, span = Math.hypot(dx,dy);
  let bendPx = null;
  if (points.length >= 5 && span >= 24) {
    const sign = dx < 0 ? -1 : 1;
    const signed = points.map(p => sign*(dx*(p.y-first.y)-dy*(p.x-first.x))/span);
    bendPx = signed.reduce((a,b) => Math.abs(a) > Math.abs(b) ? a : b);
  }
  return {
    kmh:speed !== null && metresPerPixel !== null ? speed*metresPerPixel*3.6 : null,
    bendPx,
    bendCm:bendPx !== null && metresPerPixel !== null ? bendPx*metresPerPixel*100 : null,
    rpm:markedRPM(points),
  };
}
