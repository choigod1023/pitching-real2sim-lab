import './style.css';
import {BONES, JOINTS, PitchDetector, nearestPose, markEvent, confirmed,
  comparisonError, phaseTime} from '@pitch/core';
import {listSessions, saveSession} from './storage.js';
import {LiveBall} from './live-ball.js';

const byId = id => document.getElementById(id);
const video = byId('video'), canvas = byId('overlay'), ctx = canvas.getContext('2d');
const liveBall = new LiveBall(video);
const A = '#75e4d3', B = '#ffd17e';
const regions = {full:[0,0,1,1], mound:[.40,.24,.28,.63]};
let sessions = [], selected = null, comparison = null, mode = 'idle', stream, recorder;
let objectUrl, busy = false, loop = 0, pendingStart = null, recordingTimer;
let worker, requestId = 0, lastModelTime = -1, previewStarted = 0, previewCount = 0;
let frameTime = null, seekVersion = 0;

function tell(message, isError = false) {
  byId('notice').textContent = message;
  byId('notice').dataset.error = String(isError);
}

function workerPose(bitmap, region, mediaSeconds) {
  worker ??= new Worker(new URL('./pose-worker.js', import.meta.url));
  const id = ++requestId;
  const timestamp = lastModelTime = Math.max(lastModelTime+1, Math.round(mediaSeconds*1000));
  return new Promise((resolve, reject) => {
    const current = worker;
    const cleanup = () => {
      current.removeEventListener('message', receive);
      current.removeEventListener('error', fail);
      clearTimeout(timer);
    };
    const fail = error => {
      cleanup();
      current.terminate();
      if (worker === current) worker = undefined;
      reject(error instanceof Error ? error : new Error(error.message || '관절 추적을 시작할 수 없습니다.'));
    };
    const receive = ({data}) => {
      if (data.id !== id) return;
      if (data.error) return fail(new Error(data.error));
      cleanup();
      resolve(data.points);
    };
    const timer = setTimeout(() => fail(new Error('관절 추적 시간이 초과됐습니다.')), 20000);
    current.addEventListener('message', receive);
    current.addEventListener('error', fail);
    current.postMessage({id, bitmap, region, timestamp, base:new URL('./',location.href).href}, [bitmap]);
  });
}

async function track(mediaSeconds, region) {
  const [x,y,w,h] = region;
  const width = video.videoWidth, height = video.videoHeight;
  const bitmap = await createImageBitmap(video, Math.round(x*width), Math.round(y*height),
    Math.round(w*width), Math.round(h*height));
  return workerPose(bitmap, region, mediaSeconds);
}

function freshWorker() {
  worker?.terminate();
  worker = undefined;
  lastModelTime = -1;
}

function draw(layers = [], guide = null) {
  const stage = byId('stage').getBoundingClientRect();
  const scale = Math.min(devicePixelRatio || 1, 2);
  const width = Math.round(stage.width*scale), height = Math.round(stage.height*scale);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0,0,width,height);
  const ratio = video.videoWidth / video.videoHeight;
  if (!Number.isFinite(ratio) || ratio <= 0) return;
  const w = Math.min(width,height*ratio), h = w/ratio;
  const x = (width-w)/2, y = (height-h)/2;
  if (guide && guide[2] < 1) {
    ctx.strokeStyle = '#a1c8bc';
    ctx.setLineDash([7,6]);
    ctx.strokeRect(x+guide[0]*w,y+guide[1]*h,guide[2]*w,guide[3]*h);
    ctx.setLineDash([]);
  }
  for (const [points,color] of layers) {
    if (!points) continue;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.5*scale;
    for (const [first,second] of BONES) {
      const a = points[first], b = points[second];
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(x+a[0]*w,y+a[1]*h);
      ctx.lineTo(x+b[0]*w,y+b[1]*h);
      ctx.stroke();
    }
    for (const joint of JOINTS) {
      const point = points[joint];
      if (!point) continue;
      ctx.beginPath();
      ctx.arc(x+point[0]*w,y+point[1]*h,2.7*scale,0,2*Math.PI);
      ctx.fill();
    }
  }
}

function waitFor(event, timeout = 12000) {
  return new Promise((resolve, reject) => {
    function clear() {
      clearTimeout(timer);
      video.removeEventListener(event, success);
      video.removeEventListener('error', failure);
    }
    function success() { clear(); resolve(); }
    function failure() { clear(); reject(new Error('영상을 열거나 이동할 수 없습니다.')); }
    const timer = setTimeout(failure, timeout);
    video.addEventListener(event, success, {once:true});
    video.addEventListener('error', failure, {once:true});
  });
}

async function useBlob(blob) {
  video.pause();
  video.srcObject = null;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(blob);
  frameTime = null;
  const ready = waitFor('loadedmetadata');
  video.src = objectUrl;
  await ready;
  byId('empty').hidden = true;
}

function stopPreview() {
  loop++;
  stream?.getTracks().forEach(track => track.stop());
  stream = undefined;
  video.srcObject = null;
  byId('record').disabled = true;
  byId('camera').textContent = '카메라 켜기';
}

function showMode(next) {
  mode = next;
  byId('workspace').dataset.mode = next;
  liveBall.active(next === 'live');
  byId('review').hidden = next !== 'review';
  byId('compare-controls').hidden = next !== 'compare';
  byId('manual').hidden = next !== 'review';
  byId('empty').hidden = next !== 'idle';
  if (next === 'idle') {
    byId('view-title').textContent = '투구 화면';
    byId('quality').textContent = '입력 대기';
  }
}

function preview() {
  const token = ++loop;
  let working = false, poseFailed = false, last = -Infinity;
  previewCount = 0;
  previewStarted = performance.now();
  const step = async (_, meta) => {
    if (token !== loop || mode !== 'live') return;
    video.requestVideoFrameCallback(step);
    liveBall.frame(meta.mediaTime);
    if (poseFailed || working || meta.mediaTime-last < .055) return;
    last = meta.mediaTime;
    working = true;
    const region = regions[byId('region').value];
    const started = performance.now();
    try {
      const points = await track(last, region);
      if (token !== loop || mode !== 'live') return;
      previewCount++;
      draw([[points,A]],region);
      const count = points.filter(Boolean).length;
      const fps = previewCount/((performance.now()-previewStarted)/1000);
      byId('quality').textContent = count < 7 ? '투구자 가림 또는 화면 밖' :
        `추적 ${fps.toFixed(1)} fps · ${Math.round(performance.now()-started)} ms`;
    } catch (error) {
      if (token === loop) {
        poseFailed = true;
        tell(error.message,true);
        byId('quality').textContent = '관절 추적 중단';
      }
    } finally { working = false; }
  };
  video.requestVideoFrameCallback(step);
}

async function camera() {
  if (busy || recorder?.state === 'recording') return;
  if (stream) {
    stopPreview();
    showMode('idle');
    tell('카메라를 껐습니다.');
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !video.requestVideoFrameCallback)
    throw new Error('카메라 기능을 쓰려면 HTTPS 또는 이 기기의 localhost에서 지원 브라우저로 여세요.');
  byId('camera').disabled = true;
  let opened;
  try {
    opened = await navigator.mediaDevices.getUserMedia({
      audio:false, video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:60}},
    });
  } finally { byId('camera').disabled = false; }
  stream = opened;
  video.pause();
  video.removeAttribute('src');
  video.srcObject = stream;
  try { await video.play(); } catch (error) { stopPreview(); throw error; }
  freshWorker();
  comparison = null;
  pendingStart = null;
  showMode('live');
  byId('record').disabled = false;
  byId('camera').textContent = '카메라 끄기';
  byId('view-title').textContent = '실시간 촬영';
  byId('quality').textContent = '모델 준비 중';
  tell('전신이 화면에 들어오면 녹화를 시작하세요. 녹화가 끝나면 투구 후보를 자동으로 저장합니다.');
  preview();
}

function recording() {
  if (recorder?.state === 'recording') {
    byId('record').disabled = true;
    recorder.stop();
    return;
  }
  if (!stream || !globalThis.MediaRecorder) throw new Error('이 브라우저의 녹화 기능을 사용할 수 없습니다.');
  const type = ['video/mp4','video/webm;codecs=vp8','video/webm'].find(t => MediaRecorder.isTypeSupported(t));
  recorder = new MediaRecorder(stream, type ? {mimeType:type} : undefined);
  const pieces = [];
  const side = byId('side').value, region = regions[byId('region').value];
  recorder.ondataavailable = event => { if (event.data.size) pieces.push(event.data); };
  recorder.onerror = event => { clearTimeout(recordingTimer); tell(event.error?.message || '녹화 오류',true); };
  recorder.onstop = () => {
    clearTimeout(recordingTimer);
    const name = `촬영 ${new Date().toLocaleTimeString('ko-KR')}`;
    const blob = new Blob(pieces,{type:recorder.mimeType || pieces[0]?.type});
    byId('record').textContent = '녹화 시작';
    byId('camera').disabled = false;
    for (const id of ['upload','side','region']) byId(id).disabled = false;
    if (blob.size) handle(() => importVideos([{blob,name}],side,region));
    else { byId('record').disabled = false; tell('저장할 영상이 없습니다.',true); }
  };
  recorder.start(500);
  recordingTimer = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); }, 90000);
  byId('record').textContent = '녹화 끝내기';
  byId('camera').disabled = true;
  for (const id of ['upload','side','region']) byId(id).disabled = true;
  tell('녹화 중입니다. 최대 90초까지 저장합니다.');
}

async function analyze(blob, name, side, region) {
  if (!video.requestVideoFrameCallback) throw new Error('프레임 분석을 지원하는 브라우저로 여세요.');
  await useBlob(blob);
  if (Number.isFinite(video.duration) && video.duration > 90)
    throw new Error('90초 이내의 투구 영상을 선택하세요.');
  showMode('analysis');
  byId('view-title').textContent = name;
  freshWorker();
  const aspect = video.videoWidth/video.videoHeight;
  const detector = new PitchDetector(side,aspect);
  const samples = [], intervals = [];
  let previous = null, last = -Infinity, working = false, inFlight = Promise.resolve();
  let trackingError = null;
  video.playbackRate = .75;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error('영상 분석 시간이 초과됐습니다.')), 160000);
    let running = true;
    function finish(error) {
      if (!running) return;
      running = false;
      clearTimeout(timeout);
      video.removeEventListener('ended', end);
      video.removeEventListener('error', failed);
      if (error) { video.pause(); reject(error); }
      else resolve();
    }
    const end = () => finish();
    const failed = () => finish(new Error('원본 영상을 디코딩하지 못했습니다.'));
    function step(_, meta) {
      if (!running) return;
      video.requestVideoFrameCallback(step);
      const time = meta.mediaTime;
      if (previous !== null && time > previous) intervals.push(time-previous);
      previous = time;
      if (time > 90) { video.pause(); return finish(); }
      if (working || trackingError || time-last < .047) return;
      last = time;
      working = true;
      inFlight = track(time,region).then(points => {
        samples.push({time,points});
        detector.add(time,points);
        draw([[points,A]],region);
        byId('quality').textContent = `분석 ${time.toFixed(1)}초 · ${samples.length}표본`;
      }).catch(error => {
        trackingError = error.message;
        byId('quality').textContent = '관절 추적 실패 · 원본 저장 중';
      }).finally(() => { working = false; });
    }
    video.addEventListener('ended',end,{once:true});
    video.addEventListener('error',failed,{once:true});
    video.requestVideoFrameCallback(step);
    video.play().catch(finish);
  });
  await inFlight;
  video.playbackRate = 1;
  const duration = Math.min(90, Number.isFinite(video.duration) ? video.duration : previous || 0);
  const steps = intervals.sort((a,b)=>a-b);
  const digest = await crypto.subtle.digest('SHA-256',await blob.arrayBuffer());
  const fingerprint = Array.from(new Uint8Array(digest),b => b.toString(16).padStart(2,'0')).join('');
  return {id:crypto.randomUUID(),name,blob,aspect,side,samples,duration,fingerprint,trackingError,
    frameStep:steps[Math.floor(steps.length/2)] || 1/30,
    pitches:detector.finish(duration).map(pitch => ({...pitch,id:crypto.randomUUID()}))};
}

async function importVideos(items, side, region) {
  if (busy || recorder?.state === 'recording') return;
  busy = true;
  stopPreview();
  byId('camera').disabled = true;
  byId('upload').disabled = true;
  let last = null, lastPitch = null, added = 0, errors = [];
  try {
    for (const {blob,name} of items) {
      try {
        tell(`${name} 관절 분석 중…`);
        const session = await analyze(blob,name,side,region);
        await saveSession(session);
        sessions.push(session);
        last = session;
        lastPitch = session.pitches[0] || null;
        added += session.pitches.length;
        if (session.trackingError) errors.push(session.trackingError);
      } catch (error) { errors.push(`${name} — ${error.message}`); }
    }
    if (last) await review(last,lastPitch);
    else showMode('idle');
    tell(errors.length ? `자동 저장 ${added}구. ${errors.join(' / ')}` :
      `자동 저장 ${added}구. 목록에서 투구를 골라 착지와 릴리스를 확인하세요.`,Boolean(errors.length));
  } finally {
    busy = false;
    byId('camera').disabled = false;
    byId('upload').disabled = false;
    updateList();
  }
}

function allPitches() {
  return sessions.flatMap(session => session.pitches.map((pitch,index) =>
    ({session,pitch,label:`${session.name} · ${index+1}구`})));
}

function choice(id) { return allPitches().find(item => item.pitch.id === id); }

function updateList() {
  const items = allPitches();
  byId('count').textContent = `${items.length}구`;
  const list = byId('pitches');
  list.replaceChildren();
  if (!sessions.length) list.append(Object.assign(document.createElement('p'),{className:'hint',textContent:'아직 저장된 투구가 없습니다.'}));
  for (const [index, item] of items.entries()) {
    const button = document.createElement('button');
    button.className = 'pitch-row';
    button.dataset.pitchId = item.pitch.id;
    button.setAttribute('aria-pressed',String(selected?.pitch?.id === item.pitch.id));
    const number = document.createElement('span');
    number.className = 'pitch-index';
    number.setAttribute('aria-hidden','true');
    number.textContent = String(index + 1).padStart(2,'0');
    const text = document.createElement('span');
    text.className = 'pitch-info';
    const title = document.createElement('span');
    title.textContent = item.label;
    const detail = document.createElement('small');
    detail.textContent = `${item.pitch.kind === 'auto' ? '자동 저장' : '직접 추가'} · ${confirmed(item.pitch) ? '검수 완료' : '착지·릴리스 검수 대기'}`;
    text.append(title,detail);
    button.append(number,text);
    button.onclick = () => handle(() => { if (!busy && recorder?.state !== 'recording') return review(item.session,item.pitch); });
    list.append(button);
  }
  for (const session of sessions.filter(row => !row.pitches.length)) {
    const open = Object.assign(document.createElement('button'),{className:'pitch-row',textContent:`${session.name} · 원본 검수`});
    open.onclick = () => handle(() => { if (!busy && recorder?.state !== 'recording') return review(session,null); });
    list.append(open);
  }
  for (const id of ['pitch-a','pitch-b']) {
    const select = byId(id), old = select.value;
    select.replaceChildren(new Option('투구 선택',''),...items.map(({label,pitch}) => new Option(label,pitch.id)));
    if (items.some(row => row.pitch.id === old)) select.value = old;
  }
}

function refreshReview(time = video.currentTime) {
  if (mode !== 'review' || !selected) return;
  const {pitch,session} = selected;
  draw([[nearestPose(session.samples,time),A]]);
  byId('time').value = time;
  const start = Number(byId('time').min), end = Number(byId('time').max);
  for (const event of ['landing','release']) {
    const mark = byId(`${event}-mark`);
    const eventTime = pitch?.[event];
    mark.hidden = !Number.isFinite(eventTime) || eventTime < start || eventTime > end || end <= start;
    if (!mark.hidden) mark.style.left = `${(eventTime-start)/(end-start)*100}%`;
  }
  byId('time-read').textContent = `${time.toFixed(3)}초`;
  const label = value => value === null || value === undefined ? '미확인' : `${value.toFixed(3)}초`;
  byId('events').textContent = pitch ? `착지 ${label(pitch.landing)} · 릴리스 ${label(pitch.release)}` : '원본에서 투구 시작과 끝을 직접 표시할 수 있습니다.';
}

async function seek(time) {
  const version = ++seekVersion;
  if (Math.abs(video.currentTime-time) > .002) {
    const done = waitFor('seeked',4000);
    video.currentTime = time;
    await done;
  }
  if (version !== seekVersion) return null;
  frameTime = video.currentTime;
  return frameTime;
}

async function review(session, pitch) {
  stopPreview();
  comparison = null;
  pendingStart = null;
  const reuseVideo = selected?.session.id === session.id && objectUrl && mode === 'review';
  showMode('loading');
  selected = {session,pitch};
  if (!reuseVideo) await useBlob(session.blob);
  const time = await seek(pitch?.landing ?? pitch?.start ?? 0);
  showMode('review');
  byId('view-title').textContent = session.name;
  byId('quality').textContent = `${session.samples.length}개 추적 표본`;
  byId('time').min = pitch?.start ?? 0;
  byId('time').max = pitch?.end ?? session.duration;
  byId('landing').disabled = byId('release').disabled = !pitch;
  refreshReview(time ?? video.currentTime);
  updateList();
}

async function annotate(event) {
  if (!selected?.pitch || mode !== 'review') return;
  video.pause();
  const {pitch,session} = selected;
  const next = markEvent(pitch,event,frameTime ?? video.currentTime);
  const updated = {...session,pitches:session.pitches.map(row => row.id === next.id ? next : row)};
  await saveSession(updated);
  sessions = sessions.map(row => row.id === updated.id ? updated : row);
  selected = {session:updated,pitch:next};
  refreshReview();
  updateList();
  tell(confirmed(next) ? '착지와 릴리스를 저장했습니다.' : '선택한 사건을 저장했습니다.');
}

async function addManual() {
  if (!selected || mode !== 'review' || pendingStart === null) throw new Error('시작 시점을 먼저 지정하세요.');
  const end = frameTime ?? video.currentTime;
  if (end-pendingStart < .25) throw new Error('끝 시점은 시작보다 0.25초 이상 뒤여야 합니다.');
  const session = selected.session;
  if (session.pitches.some(pitch => Math.max(pitch.start,pendingStart) < Math.min(pitch.end,end)))
    throw new Error('기존 투구 구간과 겹칩니다.');
  const pitch = {id:crypto.randomUUID(),start:pendingStart,end,landing:null,release:null,kind:'manual'};
  const updated = {...session,pitches:[...session.pitches,pitch]};
  await saveSession(updated);
  sessions = sessions.map(row => row.id === updated.id ? updated : row);
  await review(updated,pitch);
  tell('투구 구간을 저장했습니다.');
}

async function compare() {
  if (busy || recorder?.state === 'recording') return;
  const a = choice(byId('pitch-a').value), b = choice(byId('pitch-b').value);
  const error = comparisonError(a,b,byId('view-check').checked);
  byId('compare-note').textContent = error;
  if (error) return;
  stopPreview();
  await useBlob(a.session.blob);
  comparison = {a,b};
  selected = {session:a.session,pitch:a.pitch};
  showMode('compare');
  byId('view-title').textContent = '착지–릴리스 겹쳐보기';
  byId('quality').textContent = '두 투구 · 2D';
  tell('A 영상 위에서 같은 진행률의 B 관절선을 확인하세요. 가려진 점은 표시하지 않습니다.');
  updateList();
  await updatePhase();
}

async function updatePhase() {
  if (!comparison || mode !== 'compare') return;
  const {a,b} = comparison, phase = Number(byId('phase').value)/1000;
  const timeA = phaseTime(a.pitch,phase), timeB = phaseTime(b.pitch,phase);
  const time = await seek(timeA);
  if (time === null || mode !== 'compare') return;
  byId('phase-read').textContent = `${Math.round(phase*100)}% · A ${timeA.toFixed(3)}초 / B ${timeB.toFixed(3)}초`;
  draw([[nearestPose(a.session.samples,timeA),A],[nearestPose(b.session.samples,timeB),B]]);
}

function handle(operation) {
  Promise.resolve().then(operation).catch(error => tell(error.message || String(error),true));
}

byId('camera').onclick = () => handle(camera);
byId('record').onclick = () => handle(recording);
byId('upload').onchange = event => {
  const files = Array.from(event.target.files, blob => ({blob,name:blob.name}));
  event.target.value = '';
  if (files.length) handle(() => importVideos(files,byId('side').value,regions[byId('region').value]));
};
byId('time').oninput = () => handle(async () => {
  video.pause();
  const time = await seek(Number(byId('time').value));
  if (time !== null) refreshReview(time);
});
for (const [id,sign] of [['back',-1],['next',1]]) byId(id).onclick = () => handle(async () => {
  video.pause();
  const step = selected.session.frameStep*sign;
  const time = await seek(Math.max(+byId('time').min,Math.min(+byId('time').max,video.currentTime+step)));
  if (time !== null) refreshReview(time);
});
byId('play').onclick = () => handle(async () => {
  if (!video.paused) return video.pause();
  if (selected?.pitch && video.currentTime >= selected.pitch.end-.025) await seek(selected.pitch.start);
  frameTime = null;
  await video.play();
});
video.ontimeupdate = () => {
  if (mode !== 'review') return;
  if (!video.paused) frameTime = video.currentTime;
  if (selected?.pitch && video.currentTime >= selected.pitch.end) video.pause();
  refreshReview();
};
video.onplay = () => { byId('play').textContent = '일시정지'; };
video.onpause = () => { byId('play').textContent = '재생'; };
byId('landing').onclick = () => handle(() => annotate('landing'));
byId('release').onclick = () => handle(() => annotate('release'));
byId('manual-start').onclick = () => { pendingStart = frameTime ?? video.currentTime; tell('시작을 표시했습니다. 끝 시점으로 이동하세요.'); };
byId('manual-end').onclick = () => handle(addManual);
byId('compare').onclick = () => handle(compare);
byId('phase').oninput = () => handle(updatePhase);
byId('leave-compare').onclick = () => handle(() => review(comparison.a.session,comparison.a.pitch));
new ResizeObserver(() => {
  if (mode === 'compare') handle(updatePhase);
  if (mode === 'review') refreshReview();
}).observe(byId('stage'));
handle(async () => { sessions = await listSessions(); updateList(); });
