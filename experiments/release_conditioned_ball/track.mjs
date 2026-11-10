/** Free-flight hypotheses from observed image points after a release bracket. */
const START_FRAMES = 3;
const BEAM = 36;
const FRAME_GAP = 3;
const MIN_SUPPORT = 4;
const SCORE_MARGIN = .85;

function rank(track) {
  return track.points.length - .55*track.penalty - .18*track.delay;
}

function extension(track, point, length, scale) {
  const last = track.points.at(-1);
  const ds = point.s-last.s, dv = point.v-last.v;
  const gap = point.frame-last.frame;
  if (ds < .35*scale || ds > .45*length*gap || Math.abs(dv) > 5*scale+.90*ds) return null;
  const ratio = point.diameter/last.diameter;
  if (ratio < .32 || ratio > 3.2) return null;
  let penalty = .13*Math.abs(Math.log(ratio));
  const previous = track.points.at(-2);
  if (previous) {
    const timeRatio = (point.time-last.time)/(last.time-previous.time);
    const error = Math.hypot(ds-timeRatio*(last.s-previous.s),dv-timeRatio*(last.v-previous.v));
    const limit = Math.max(4*scale,.55*Math.hypot(ds,dv)+1.5*scale);
    if (error > limit) return null;
    penalty += error/limit;
  }
  return {...track,points:[...track.points,point],penalty:track.penalty+penalty};
}

function prune(tracks, scale) {
  const unique = new Map();
  const cell = point => [point.frame,Math.round(point.x/(3*scale)),Math.round(point.y/(3*scale))].join('/');
  for (const track of tracks.sort((a,b)=>rank(b)-rank(a))) {
    const points = track.points;
    const key = cell(points[0])+'|'+cell(points.at(-1));
    if (!unique.has(key)) unique.set(key,track);
    if (unique.size >= BEAM) break;
  }
  return [...unique.values()];
}

export function releaseTrack(frames, release, length, width) {
  if (!Array.isArray(release) || release.length !== 2 ||
    !release.every(Number.isInteger) || release[0] > release[1] ||
    !Number.isFinite(length) || length <= 0 || !Number.isFinite(width) || width <= 0) {
    throw new Error('Expected an ordered release bracket and positive image geometry');
  }
  const scale = width/960;
  const lastPossibleContact = release[1];
  let beam = [], finished = [];
  for (const {frame,time,proposals} of frames) {
    if (frame <= lastPossibleContact) continue;
    const points = proposals.map(point=>({...point,frame,time}));
    const next = [];
    for (const track of beam) {
      if (frame-track.points.at(-1).frame > FRAME_GAP) {
        finished.push(track);
        continue;
      }
      next.push(track);
      for (const point of points) {
        const extended = extension(track,point,length,scale);
        if (extended) next.push(extended);
      }
    }
    if (frame <= lastPossibleContact+START_FRAMES) {
      for (const point of points) {
        if (point.s <= .45*length) next.push({points:[point],delay:frame-lastPossibleContact-1,penalty:0});
      }
    }
    beam = prune(next,scale);
  }
  finished = prune([...finished,...beam].filter(track => {
    const points = track.points;
    return points.length >= MIN_SUPPORT && points.at(-1).s-points[0].s >= Math.max(4*scale,.04*length);
  }),scale);
  if (!finished.length) return {points:[],ambiguousFrames:[],hypothesisCount:0,status:'insufficient_support'};
  const best = finished[0];
  const rivals = finished.filter(track => rank(track) >= rank(best)-SCORE_MARGIN);
  const ambiguousFrames = [];
  const points = best.points.filter(point => {
    const alternatives = rivals.map(track=>track.points.find(row=>row.frame===point.frame)).filter(Boolean);
    const agree = alternatives.length/rivals.length >= 2/3 && alternatives.every(row =>
      Math.hypot(row.x-point.x,row.y-point.y) <= 5*scale);
    if (!agree) ambiguousFrames.push(point.frame);
    return agree;
  });
  return {points,ambiguousFrames,hypothesisCount:rivals.length,
    status:points.length ? 'observed' : 'ambiguous'};
}
