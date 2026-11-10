/** Preserve fitted image points and require later observations before changing the path. */
const SUPPORT_DISTANCE = 5;
const IMPROVEMENT = 1.5;
const AMBIGUITY_MARGIN = 1;
const dist = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);

function evidenceScore(first, second, following, fixed, scale, length, strict) {
  const gap = second.frame-first.frame;
  const ds = second.s-first.s;
  if (gap<1 || gap>2 || ds<.35*scale || ds>.45*length*gap ||
    Math.abs(second.v-first.v)>5*scale+.90*ds || second.time<=first.time) return null;
  const anchor = fixed.get(second.frame);
  if (strict && anchor?.usedForFit && dist(second,anchor)>SUPPORT_DISTANCE*scale) return null;
  let hits = 0, error = 0, tested = 0;
  for (const frame of following) {
    const ratio = (frame.time-first.time)/(second.time-first.time);
    const prediction = {x:first.x+ratio*(second.x-first.x),y:first.y+ratio*(second.y-first.y)};
    const anchor = fixed.get(frame.frame);
    if (strict && anchor?.usedForFit && dist(prediction,anchor)>SUPPORT_DISTANCE*scale) return null;
    const observed = anchor?.usedForFit ? frame.agreed.filter(point=>dist(point,anchor)<=SUPPORT_DISTANCE*scale) : frame.agreed;
    const residual = Math.min(...observed.map(point=>dist(prediction,point)/scale), SUPPORT_DISTANCE+1);
    error += residual;
    tested++;
    if (residual<=SUPPORT_DISTANCE) hits++;
  }
  return tested>=2 ? {hits,error:error/tested} : null;
}

export function preservePath(baseline, frames, release, width, length, {anchorGuard=false}={}) {
  const [low,high] = release;
  const scale = width/960;
  const fixed = new Map(baseline.map(point=>[point.frame,point]));
  const prepared = frames.map(frame => {
    // Keep coordinates from actual observations, never the projected locations.
    const agreed = frame.app.filter(point => frame.round.some(other=>dist(point,other)<=4*scale));
    return {...frame,agreed};
  });
  const last = baseline.length ? baseline.at(-1).frame : high+5;
  const changed = {added:[],replaced:[],suppressedPreRelease:[],ambiguous:[]};
  const points = [];
  for (const [index,frame] of prepared.entries()) {
    const f = frame.frame;
    const original = fixed.get(f);
    if (f<low) {
      if (original) changed.suppressedPreRelease.push(f);
      continue;
    }
    if (f<=high || f>last || original?.usedForFit) {
      if (original) points.push(original);
      continue;
    }
    const next = prepared.slice(index+1,index+3).filter(row=>row.frame<=f+2);
    const following = prepared.slice(index+1,index+6).filter(row=>row.frame>=f+3 && row.frame<=f+5);
    const anchors = anchorGuard ? [...next,...following].map(row=>fixed.get(row.frame))
      .filter(point=>point?.usedForFit).slice(0,2) : [];
    if (anchorGuard && baseline.length && anchors.length<2) {
      if (original) points.push(original);
      continue;
    }
    let origins = frame.agreed;
    if (anchors.length===2) {
      const [a,b] = anchors;
      const ratio = (frame.time-a.time)/(b.time-a.time);
      const backward = {x:a.x+ratio*(b.x-a.x),y:a.y+ratio*(b.y-a.y)};
      origins = [...frame.app,...frame.round].filter(point=>dist(point,backward)<=8*scale);
    }
    const scored = (point, seeds, strict) => seeds.map(seed=>evidenceScore(point,seed,following,fixed,scale,length,strict))
      .filter(Boolean).sort((a,b)=>a.error-b.error);
    const withTime = (point,row) => ({...point,frame:row.frame,time:row.time});
    const seeds = next.flatMap(row=>row.agreed.map(point=>withTime(point,row)));
    const options = [];
    for (const proposal of origins) {
      const point = withTime(proposal,frame);
      const score = scored(point,seeds,true).find(value=>value.hits>=2);
      if (score) options.push({point,...score});
    }
    options.sort((a,b)=>a.error-b.error);
    const best = options[0];
    const ambiguous = best && options.some(other=>other.error-best.error<AMBIGUITY_MARGIN &&
      dist(other.point,best.point)>SUPPORT_DISTANCE*scale);
    if (ambiguous) changed.ambiguous.push(f);
    const prior = original && scored(original,next.map(row=>fixed.get(row.frame)).filter(Boolean),false)[0];
    const gain = prior ? prior.error-(best?.error ?? Infinity) : -Infinity;
    if (best && !ambiguous && (!original || (gain>=IMPROVEMENT && dist(original,best.point)>SUPPORT_DISTANCE*scale))) {
      const source = original ? 'replaced' : 'added';
      changed[source].push(f);
      points.push({...best.point,usedForFit:false,pathDecision:source,futureHits:best.hits,futureErrorPx960:best.error});
    } else if (original) {
      points.push(original);
    }
  }
  return {points,changed};
}
