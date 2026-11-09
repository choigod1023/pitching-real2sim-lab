export const JOINTS = [11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32];
export const BONES = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[29,31],[24,26],[26,28],[28,30],[30,32]];

export function normalizePose(landmarks, region = [0,0,1,1]) {
  const [left, top, width, height] = region;
  return Array.from({length:33}, (_, index) => {
    if (!JOINTS.includes(index)) return null;
    const point = landmarks?.[index];
    const score = Math.min(point?.visibility ?? 1, point?.presence ?? 1);
    if (!point || score < .55 || ![point.x, point.y, score].every(Number.isFinite) ||
        point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) return null;
    return [left + width * point.x, top + height * point.y];
  });
}

const distance = (a, b, aspect) => Math.hypot((a[0]-b[0])*aspect, a[1]-b[1]);

export class PitchDetector {
  constructor(side = 'right', aspect = 1) {
    this.side = side;
    this.aspect = aspect;
    this.previous = null;
    this.burst = null;
    this.lastPeak = -Infinity;
    this.peaks = [];
  }

  add(time, points) {
    const [shoulderIndex, wristIndex, hipIndex] = this.side === 'left' ? [11,15,23] : [12,16,24];
    const shoulder = points[shoulderIndex], wrist = points[wristIndex], hip = points[hipIndex];
    const previous = this.previous;
    const dt = previous ? time - previous.time : 0;
    const torso = shoulder && hip ? distance(shoulder, hip, this.aspect) : 0;
    const speed = wrist && previous?.wrist && torso > .035 && dt >= .012 && dt <= .20
      ? distance(wrist, previous.wrist, this.aspect) / torso / dt : 0;
    this.previous = {time, wrist};
    if (this.burst && time - this.burst.last > .25) this.flush();
    if (speed >= 3.4 && speed <= 42 && wrist[1] <= shoulder[1] + .75 * torso &&
        time - this.lastPeak >= 1.7) {
      if (!this.burst) this.burst = {time, speed, count:0, last:time};
      this.burst.count++;
      this.burst.last = time;
      if (speed > this.burst.speed) Object.assign(this.burst, {time,speed});
    }
  }

  flush() {
    if (this.burst?.count >= 2) {
      this.lastPeak = this.burst.time;
      this.peaks.push(this.lastPeak);
    }
    this.burst = null;
  }

  finish(duration) {
    this.flush();
    return this.peaks.filter(time => time >= 0 && time <= duration).map((peak, index, peaks) => ({
      start:Math.max(0, peak-1.12, index ? (peaks[index-1]+peak)/2 : 0),
      end:Math.min(duration, peak+.78, index+1 < peaks.length ? (peaks[index+1]+peak)/2 : duration),
      peak, landing:null, release:null, kind:'auto',
    })).filter(pitch => pitch.end - pitch.start >= .25);
  }
}

export function nearestPose(samples, time, maxGap = .115) {
  if (!samples.length || !Number.isFinite(time)) return null;
  let low = 0, high = samples.length-1;
  while (low < high) {
    const middle = (low+high)>>1;
    if (samples[middle].time < time) low = middle+1;
    else high = middle;
  }
  const closest = low && time-samples[low-1].time < Math.abs(samples[low].time-time)
    ? samples[low-1] : samples[low];
  return Math.abs(closest.time-time) <= maxGap ? closest.points : null;
}

export function markEvent(pitch, event, time) {
  if (!['landing','release'].includes(event) || !Number.isFinite(time) ||
      time < pitch.start || time > pitch.end) throw new Error('현재 시점이 투구 구간 밖입니다.');
  const next = {...pitch, [event]:time};
  if (next.landing !== null && next.release !== null && next.landing >= next.release)
    throw new Error('착지는 릴리스보다 앞이어야 합니다.');
  return next;
}

export const confirmed = pitch => pitch?.landing != null && pitch?.release != null &&
  pitch.landing < pitch.release;

export function comparisonError(a, b, checkedView) {
  if (!a || !b || a.pitch.id === b.pitch.id) return '서로 다른 두 투구를 고르세요.';
  if (a.session.fingerprint === b.session.fingerprint &&
      Math.max(a.pitch.start,b.pitch.start) < Math.min(a.pitch.end,b.pitch.end))
    return '같은 원본에서 겹치는 구간입니다.';
  if (!confirmed(a.pitch) || !confirmed(b.pitch)) return '두 투구의 착지와 릴리스를 확인하세요.';
  if (a.session.side !== b.session.side || Math.abs(a.session.aspect/b.session.aspect - 1) > .025)
    return '투구 손과 영상의 가로세로 비율이 같아야 합니다.';
  if (!checkedView) return '같은 선수와 고정된 시야인지 확인하세요.';
  return '';
}

export function phaseTime(pitch, phase) {
  return pitch.landing + Math.max(0,Math.min(1,phase)) * (pitch.release-pitch.landing);
}
