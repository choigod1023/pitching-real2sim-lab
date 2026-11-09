import {BallTracker} from '@pitch/core/ball.mjs';

const byId = id => document.getElementById(id);
const number = (value, precision, unit) => value === null ? '—' : `${value.toFixed(precision)} ${unit}`;
const ink = '#ffe2a0';

export class LiveBall {
  constructor(video) {
    this.video = video;
    this.canvas = byId('ball-overlay');
    this.layer = this.canvas.getContext('2d');
    this.source = document.createElement('canvas');
    this.pixels = this.source.getContext('2d', {willReadFrequently:true});
    this.anchors = [];
    this.calibrating = false;
    this.tracker = null;
    this.last = null;
    byId('ball-calibrate').onclick = () => {
      this.reset();
      this.calibrating = true;
      byId('stage').classList.add('marking');
      this.status('공이 날아갈 구간의 시작점과 끝점을 차례로 누르세요.');
    };
    byId('ball-reset').onclick = () => this.reset();
    byId('stage').addEventListener('click', event => this.pick(event));
    byId('ball-metres').onchange = () => this.configure();
    byId('ball-marked').onchange = () => {
      this.configure();
      const track = video.srcObject?.getVideoTracks()[0];
      track?.applyConstraints({frameRate:{ideal:byId('ball-marked').checked ? 240 : 60}}).catch(() => {});
    };
  }

  active(enabled) {
    byId('ball-controls').hidden = !enabled;
    this.canvas.hidden = !enabled;
    if (!enabled) this.reset();
  }

  reset() {
    this.anchors = [];
    this.calibrating = false;
    this.tracker = null;
    this.last = null;
    byId('stage').classList.remove('marking');
    this.status('고정된 측면 영상에서 공이 날아가는 구간을 지정하세요.');
    this.report(null);
    this.paint(null, null);
  }

  geometry(width, height) {
    const ratio = this.video.videoWidth / this.video.videoHeight;
    if (!Number.isFinite(ratio) || ratio <= 0) return null;
    const w = Math.min(width,height*ratio), h = w/ratio;
    return {x:(width-w)/2, y:(height-h)/2, w, h};
  }

  pick(event) {
    if (!this.calibrating || this.canvas.hidden) return;
    const bounds = byId('stage').getBoundingClientRect();
    const fit = this.geometry(bounds.width, bounds.height);
    if (!fit) return;
    const x = (event.clientX - bounds.left - fit.x) / fit.w;
    const y = (event.clientY - bounds.top - fit.y) / fit.h;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    this.anchors.push({x,y});
    if (this.anchors.length === 1) {
      this.status('공이 진행하는 방향의 두 번째 점을 누르세요.');
      this.paint(null, null);
      return;
    }
    if (this.configure()) {
      this.calibrating = false;
      byId('stage').classList.remove('marking');
    } else this.anchors.pop();
  }

  configure() {
    if (this.anchors.length !== 2 || this.canvas.hidden || !this.video.videoWidth) return false;
    const factor = Math.min(1,960/Math.max(this.video.videoWidth,this.video.videoHeight));
    const width = Math.round(this.video.videoWidth*factor), height = Math.round(this.video.videoHeight*factor);
    const [a,b] = this.anchors.map(p => ({x:p.x*width, y:p.y*height}));
    const distance = byId('ball-metres');
    try {
      this.tracker = new BallTracker(width,height,a,b,{
        metres:distance.validity.valid ? distance.valueAsNumber : null, markedBall:byId('ball-marked').checked,
      });
    } catch (error) {
      this.status(error.message);
      return false;
    }
    this.source.width = width;
    this.source.height = height;
    this.last = null;
    this.status('공 관측 대기 · 카메라를 고정하세요.');
    this.report(null);
    return true;
  }

  frame(time) {
    if (!this.tracker) return;
    try {
      const {width,height} = this.source;
      this.pixels.drawImage(this.video,0,0,width,height);
      const data = this.pixels.getImageData(0,0,width,height).data;
      const latest = this.tracker.push(data,time);
      if (latest !== this.last) {
        this.last = latest;
        this.report(latest);
      }
      this.paint(latest,time);
      const fresh = latest && time-latest.updatedAt <= latest.gap;
      this.status(fresh ? `공 중심 후보 ${latest.points.length}개` :
        latest ? `최근 경로 · 공 중심 후보 ${latest.points.length}개` : '공 관측 대기 · 카메라를 고정하세요.');
    } catch {
      this.tracker = null;
      this.last = null;
      this.paint(null,null);
      this.report(null);
      this.status('공 추적을 다시 시작하려면 추적 구간을 지정하세요.');
    }
  }

  report(result) {
    byId('ball-speed').textContent = number(result?.kmh ?? null,1,'km/h');
    byId('ball-rpm').textContent = number(result?.rpm ?? null,0,'rpm');
    byId('ball-bend').textContent = result?.bendCm != null ? number(result.bendCm,1,'cm') :
      number(result?.bendPx ?? null,1,'px');
    byId('ball-speed-note').textContent = this.tracker?.metres ? '지정 평면 · 초기 관측 구간' : '실제 기준 거리 필요';
    byId('ball-rpm-note').textContent = byId('ball-marked').checked ?
      '약 200 fps 이상 · 파란 점이 보이는 훈련구' : '고속 표식 모드 필요';
  }

  status(message) {
    if (byId('ball-status').textContent !== message) byId('ball-status').textContent = message;
  }

  paint(result, time) {
    const scale = Math.min(devicePixelRatio || 1, 2);
    const rect = byId('stage').getBoundingClientRect();
    const width = Math.round(rect.width*scale), height = Math.round(rect.height*scale);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const ctx = this.layer;
    ctx.clearRect(0,0,width,height);
    const fit = this.geometry(width,height);
    if (!fit) return;
    const pixel = p => ({x:fit.x+p.x*fit.w, y:fit.y+p.y*fit.h});
    ctx.strokeStyle = ink;
    ctx.fillStyle = ink;
    ctx.lineWidth = 2.3*scale;
    if (this.anchors.length === 2) {
      const [a,b] = this.anchors.map(pixel);
      ctx.globalAlpha = .48;
      ctx.setLineDash([5*scale,5*scale]);
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    for (const point of this.anchors.map(pixel)) {
      ctx.strokeRect(point.x-3*scale,point.y-3*scale,6*scale,6*scale);
    }
    if (!result) return;
    const points = result.points.map(p => ({...pixel({x:p.x/this.source.width,y:p.y/this.source.height}), time:p.time}));
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (i === 0 || p.time-points[i-1].time > result.gap*.63) ctx.moveTo(p.x,p.y);
      else ctx.lineTo(p.x,p.y);
    }
    ctx.stroke();
    for (const p of points) {
      ctx.beginPath(); ctx.arc(p.x,p.y,2*scale,0,Math.PI*2); ctx.fill();
    }
    if (time-result.updatedAt <= result.gap) {
      const p = points.at(-1);
      ctx.beginPath(); ctx.arc(p.x,p.y,6*scale,0,Math.PI*2); ctx.stroke();
    }
  }
}
