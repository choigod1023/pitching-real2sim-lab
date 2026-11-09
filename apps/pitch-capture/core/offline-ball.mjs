import {BallTracker} from './ball.mjs';

// A straight trajectory viewed obliquely has a shared projective denominator in x and y.
// Time is normalized to the observed span; the model is never extrapolated for a readout.
function fit(points, kind, span = points) {
  if (points.length < (kind === 'linear' ? 2 : 3)) return null;
  const start = span[0].time, duration = span.at(-1).time - start;
  if (!(duration > 0)) return null;
  const origin = span[0];
  const rows = points.map(p => ({t:(p.time-start)/duration, x:p.x-origin.x, y:p.y-origin.y}));
  let best = null;
  const bound = kind === 'linear' ? 0 : 50;
  for (let n=-bound; n<=bound; n++) {
    const c = n / 100;
    let w=0, t=0, tt=0, x=0, tx=0, y=0, ty=0;
    for (const row of rows) {
      const d = 1+c*row.t, weight = 1/(d*d);
      w += weight;
      t += row.t*weight;
      tt += row.t*row.t*weight;
      x += row.x*d*weight;
      tx += row.t*row.x*d*weight;
      y += row.y*d*weight;
      ty += row.t*row.y*d*weight;
    }
    const det = w*tt-t*t;
    if (det <= 1e-9) continue;
    const a = {x:(tt*x-t*tx)/det,y:(tt*y-t*ty)/det};
    const b = {x:(w*tx-t*x)/det,y:(w*ty-t*y)/det};
    const at = time => {
      const u = (time-start)/duration;
      return {x:origin.x+(a.x+b.x*u)/(1+c*u),y:origin.y+(a.y+b.y*u)/(1+c*u)};
    };
    const residual = Math.sqrt(points.reduce((sum,p) => {
      const v = at(p.time);
      return sum+(v.x-p.x)**2+(v.y-p.y)**2;
    },0)/points.length);
    if (!best || residual < best.residual) best = {at,c,residual};
  }
  return best;
}

function checkedFit(points, kind, limit) {
  const model = fit(points,kind);
  if (!model || model.residual > limit) return null;
  // Predict both edge samples without fitting them. In particular, three points alone
  // cannot provide an independent check on a three-parameter projective trace.
  const edgeErrors = [0,points.length-1].map(index => {
    const held = points[index];
    const partial = fit(points.filter((_,i) => i !== index),kind,points);
    if (!partial) return Infinity;
    const predicted = partial.at(held.time);
    return Math.hypot(predicted.x-held.x,predicted.y-held.y);
  });
  const edgeError = Math.max(...edgeErrors);
  return edgeError <= limit*2 ? {...model,edgeError,kind} : null;
}

function crossing(points, signed, target, maxGap) {
  for (let i=1; i<points.length; i++) {
    const before = points[i-1], after = points[i];
    const a = signed(before), b = signed(after);
    if (a <= target && b >= target && b > a && after.time-before.time <= maxGap) {
      return {before:before.time,after:after.time};
    }
  }
  return null;
}

function crossingTime(model, bracket, signed, target) {
  let {before,after} = bracket;
  if (signed(model.at(before)) > target || signed(model.at(after)) < target) return null;
  for (let i=0; i<28; i++) {
    const middle = (before+after)/2;
    if (signed(model.at(middle)) < target) before = middle;
    else after = middle;
  }
  return (before+after)/2;
}

export class OfflineBallReview {
  constructor(width, height, start, end, metres = null) {
    this.tracker = new BallTracker(width,height,start,end,{metres});
    this.times = [];
    this.track = null;
    this.baseline = null;
  }

  push(rgba, time) {
    if (!Number.isFinite(time) || time <= (this.times.at(-1) ?? -Infinity)) return;
    this.times.push(time);
    const result = this.tracker.push(rgba,time);
    if (result?.updatedAt !== time) return;
    if (!this.track || result.points.length > this.track.points.length) this.track = result;
    if (result.kmh !== null && (!this.baseline || result.points.length > this.baseline.points.length)) {
      this.baseline = result;
    }
  }

  finish() {
    const {width,height,start,length,ux,uy,band,metres} = this.tracker;
    const signed = p => (p.x-start.x)*ux + (p.y-start.y)*uy;
    const points = this.track?.points ?? [];
    const limit = 3.5 * width / 960;
    // Components cut by the search boundary can move their measured centroid.
    const usable = points.filter(p => p.s > -.12*length+p.diameter/2 &&
      p.s < 1.12*length-p.diameter/2 && Math.abs(p.v) < band-p.diameter/2);
    const linear = fit(usable,'linear');
    const model = usable.length < 4 ? null :
      checkedFit(usable,'linear',limit) ?? checkedFit(usable,'perspective',limit);
    const gap = Math.min(.05,this.track?.gap ?? .05);
    const releaseBracket = crossing(usable,signed,0,gap);
    const release = releaseBracket && model
      ? {...releaseBracket, estimate:crossingTime(model,releaseBracket,signed,0), method:'marked_start_crossing'}
      : null;
    let imageSpeedPxS = null, kmh = null, speedBoundsKmh = null;
    let speedStatus = model ? 'uncalibrated' : usable.length < 4 ? 'too_few_centres' : 'unreliable_fit';
    if (model) {
      const first = usable[0].time, last = usable.at(-1).time;
      const a = model.at(first), b = model.at(last);
      imageSpeedPxS = Math.hypot(b.x-a.x,b.y-a.y)/(last-first);
      if (metres !== null) {
        const finish = crossing(usable,signed,length,gap);
        const startTime = release?.estimate ?? null;
        const finishTime = finish && crossingTime(model,finish,signed,length);
        speedStatus = 'unbracketed_gates';
        const fastest = finish && release ? finish.before-release.after : null;
        if (startTime !== null && finishTime !== null && finishTime > startTime && fastest > 0) {
          speedStatus = 'available';
          kmh = metres/(finishTime-startTime)*3.6;
          // Range from observed frame brackets only. Camera calibration and detection errors are separate.
          speedBoundsKmh = [metres/(finish.after-release.before)*3.6, metres/fastest*3.6];
        }
      }
    }
    return {
      version:1,width,height,frameCount:this.times.length,
      points:points.map(({x,y,time})=>({x,y,time,usedForFit: Boolean(model && usable.some(p=>p.time===time))})),
      firstObservation:points[0]?.time ?? null,
      release:release?.estimate != null ? release : null,
      usableCentres:usable.length,
      linearResidualPx:linear?.residual ?? null,
      fit:model ? {kind:model.kind,residualPx:model.residual,edgeErrorPx:model.edgeError} : null,
      imageSpeedPxS,kmh,speedBoundsKmh,speedStatus,
      baseline:{pointCount:points.length,kmh:this.baseline?.kmh ?? null},
    };
  }
}
