const distance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);

// Back-project only to rank image detections. Never return a predicted coordinate.
export function recoverRelease(baseline, evidence, release, width) {
  const scale = width/960;
  const byFrame = new Map(baseline.map(point=>[point.frame,point]));
  const frames = new Map(evidence.map(row=>[row.frame,row]));
  const first = baseline.find(point=>point.usedForFit);
  const checks = [];
  const recovered = [];
  for (let frame=release[1]+1; frame<=release[1]+2; frame++) {
    const check = {frame,status:'unavailable'};
    checks.push(check);
    if (byFrame.has(frame)) {
      check.status = 'existing';
      continue;
    }
    if (!first || first.frame-frame<1 || first.frame-frame>2) {
      check.status = 'no_nearby_fit';
      continue;
    }
    const anchors = [0,1,2,3].map(offset=>byFrame.get(first.frame+offset));
    if (!anchors.every(point=>point?.usedForFit)) {
      check.status = 'nonconsecutive_fit';
      continue;
    }
    const times = Array.from({length:first.frame-frame+4},(_,index)=>frames.get(frame+index)?.time);
    const gaps = times.slice(1).map((time,index)=>time-times[index]);
    if (gaps.some(gap=>!(gap>0)) || Math.max(...gaps)>1.5*Math.min(...gaps)) {
      check.status = 'irregular_pts';
      continue;
    }
    const [a,b,...validation] = anchors;
    if (distance(a,b)<2*scale || anchors.slice(1).some((point,index)=>!(point.s>anchors[index].s))) {
      check.status = 'inconsistent_direction';
      continue;
    }
    const at = time => {
      const fraction = (time-a.time)/(b.time-a.time);
      return {x:a.x+fraction*(b.x-a.x),y:a.y+fraction*(b.y-a.y)};
    };
    check.followupErrorsPx960 = validation.map(point=>distance(at(point.time),point)/scale);
    if (check.followupErrorsPx960.some(error=>error>5)) {
      check.status = 'followup_mismatch';
      continue;
    }
    const {time,app} = frames.get(frame);
    const predicted = at(time);
    const candidates = app.map(point=>({point,error:distance(point,predicted)/scale}))
      .sort((left,right)=>left.error-right.error);
    const best = candidates[0];
    check.nearestPx960 = best?.error ?? null;
    if (!best || best.error>5) {
      check.status = 'no_image_candidate';
      continue;
    }
    if (candidates.slice(1).some(({point,error})=>error<=5 && error-best.error<=1 &&
        distance(point,best.point)>3*scale)) {
      check.status = 'ambiguous';
      continue;
    }
    check.status = 'recovered';
    recovered.push({...best.point,frame,time,usedForFit:false,pathDecision:'reverse_observed'});
  }
  return {points:[...baseline,...recovered].sort((a,b)=>a.frame-b.frame),checks};
}
