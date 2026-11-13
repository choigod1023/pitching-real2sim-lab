const byId = id => document.getElementById(id);
const ui = Object.fromEntries([
  "folder","demo","notice","task-title","frame-id","viewport","picture","frame-image","point","empty",
  "prev","next","scrub","zoom","strip","progress","task","reader","controls","current","absent","unknown",
  "clear","contact","free","contact-value","free-value","uncertain","notes","download","readiness","save-status","retry","submit",
  "proposal-point","proposal-status","question","accept","reject","unseen","correction-hint",
  "release-assist","release-before","release-after","release-before-image","release-after-image",
  "release-before-label","release-after-label","release-help","release-yes","release-no","release-unknown","release-state","release-details",
  "portal","login-form","login-reader","access-code","login-submit","login-message","scene-hub","scene-list","scene-total",
  "account-name","change-reader","workbench","scene-status","inspect-issue","next-scene",
  "submit-all","scene-submit-help",
].map(id => [id, byId(id)]));
const {checkManifest, readiness, submission, reviewFrame, sceneProgress, submissionState} = PitchLabels;
const states = {ball:"공 중심 표시됨", absent:"자유 비행 없음", unknown:"판독 불가"};
let manifest, task, drafts, sources, index = 0, frameURL, currentFrameLoaded = false, loadTicket = 0;
let storageKey = "";
let storageFailed = false;
let session;
let sceneSubmissions = {}, submittingScene = false;
let frameRequest;
let correction = false, releaseReady = false, releaseTicket = 0, releaseURLs = [];
const accountStorageKey = "pitch-review-current-reader";
let token = new URLSearchParams(location.hash.slice(1)).get("key");
try { token ??= localStorage.getItem(accountStorageKey); } catch { /* Login remains usable. */ }

function showSaveStatus(message) {
  ui["save-status"].textContent = message;
  ui.retry.hidden = !message.startsWith("서버 저장 실패");
}

function save() {
  try {
    localStorage.setItem(storageKey, JSON.stringify({reader:ui.reader.value,drafts}));
  } catch {
    if (!storageFailed) ui.notice.textContent += " 브라우저 임시 저장을 쓸 수 없습니다. 이 탭에서 작업을 마치세요.";
    storageFailed = true;
  }
}

async function mount(data, files, online, remote) {
  checkManifest(data);
  if (files) {
    for (const task of data.tasks) {
      for (const frame of task.frames) {
        if (!files.has(frame.file)) throw new Error("작업 폴더에 필요한 프레임 파일이 없습니다.");
      }
    }
  }
  session = online;
  ui.portal.hidden = true;
  ui.workbench.hidden = false;
  ui["scene-hub"].hidden = false;
  ui["change-reader"].hidden = !online;
  ui["account-name"].hidden = !online;
  ui["account-name"].textContent = online ? remote.reader+" 작업 중" : "";
  ui["save-status"].hidden = !online;
  ui.retry.hidden = true;
  ui.submit.hidden = !online;
  ui["submit-all"].hidden = !online;
  ui.reader.readOnly = Boolean(online);
  manifest = data;
  sceneSubmissions = remote?.sceneSubmissions ?? {};
  sources = files;
  storageKey = "pitch-labels-v1:" + JSON.stringify([data.batch,data.tasks.map(t=>[t.id,t.sha256,t.frames[0].index,t.frames.at(-1).index])]);
  let previous;
  try { previous = JSON.parse(localStorage.getItem(storageKey) || "null"); } catch { previous = null; }
  drafts = Object.fromEntries(data.tasks.map(t=>[t.id,(online ? remote.drafts[t.id] : previous?.drafts?.[t.id]) ??
    {marks:{},contact:null,free:null,uncertain:false,notes:""}]));
  ui.reader.value = remote?.reader ?? previous?.reader ?? "";
  ui.task.replaceChildren(...data.tasks.map(t=>new Option(t.id+" · "+t.group,t.id)));
  ui.task.disabled = false;
  ui["scene-list"].replaceChildren(...data.tasks.map((t,i)=>{
    const button = document.createElement("button");
    const title = document.createElement("strong");
    title.textContent = "씬 "+(i+1)+" · "+t.group;
    const name = document.createElement("small");
    name.textContent = t.id;
    const progress = document.createElement("span");
    button.append(title,name,progress);
    button.addEventListener("click",()=>{
      select(t.id);
      ui.workbench.scrollIntoView({block:"start"});
    });
    return button;
  }));
  ui.notice.textContent = data.practice ? "연습용 합성 프레임입니다. 연습 결과는 연구 판독에 포함하지 않습니다." :
    online ? "자동 후보를 먼저 확인하세요. 맞음·틀림·안 보임으로 답하면 서버에 자동 저장됩니다." :
      "영상은 브라우저에서만 엽니다. 모든 작업을 판독한 뒤 결과 JSON을 의뢰자에게 보내세요.";
  select((data.tasks.find(t=>submissionState(drafts[t.id],sceneSubmissions[t.id]) !== "received") ?? data.tasks[0]).id);
  session?.report();
}

function select(id) {
  task = manifest.tasks.find(entry=>entry.id===id);
  ui.task.value = id;
  ui["task-title"].textContent = task.id + " · " + task.group;
  ui.scrub.min = 0;
  ui.scrub.max = task.frames.length - 1;
  ui.strip.replaceChildren(...task.frames.map(({index: frame}, position)=>{
    const button = document.createElement("button");
    button.textContent = frame;
    button.addEventListener("click",()=>seek(position));
    return button;
  }));
  const firstProposal = task.frames.findIndex(f=>task.assistance?.points[f.index] && !drafts[task.id].marks[f.index]);
  const firstMissing = task.frames.findIndex(f=>!drafts[task.id].marks[f.index]);
  const progress = sceneProgress(task,drafts[task.id]);
  seek(progress.status === "needs_review" && progress.frame != null ? task.frames.findIndex(f=>f.index===progress.frame) :
    firstProposal >= 0 ? firstProposal : Math.max(0,firstMissing));
  void previewRelease();
}

async function previewRelease() {
  const ticket = ++releaseTicket;
  releaseReady = false;
  releaseURLs.forEach(url=>URL.revokeObjectURL(url));
  releaseURLs = [];
  const hint = task.assistance?.release_hint;
  ui["release-assist"].hidden = !hint;
  ui["release-details"].open = !hint;
  if (!hint) return;
  try {
    for (const [i,side] of ["before","after"].entries()) {
      const frame = task.frames.find(f=>f.index===hint[i]);
      ui["release-"+side+"-image"].removeAttribute("src");
      const blob = sources ? sources.get(frame.file) : session ? await session.frame(frame.file) : null;
      if (ticket !== releaseTicket) return;
      const url = blob ? URL.createObjectURL(blob) : "./demo/"+frame.file;
      if (blob) releaseURLs.push(url);
      const image = ui["release-"+side+"-image"];
      image.src = url;
      await image.decode();
      if (ticket !== releaseTicket) return;
      ui["release-"+side+"-label"].textContent = "F"+hint[i]+" · 크게 보기";
    }
    releaseReady = true;
  } catch {
    ui["release-help"].textContent = "후보 사진을 열지 못했습니다. 앞뒤 프레임을 직접 확인하거나 새로고침하세요.";
  }
  refresh();
}

async function seek(target) {
  index = Math.max(0, Math.min(task.frames.length - 1, target));
  const frame = task.frames[index];
  correction = !!drafts[task.id].rejections?.[frame.index] && !drafts[task.id].marks[frame.index];
  const ticket = ++loadTicket;
  frameRequest?.abort();
  const controller = new AbortController();
  frameRequest = controller;
  currentFrameLoaded = false;
  const image = ui["frame-image"];
  image.removeAttribute("src");
  if (frameURL) URL.revokeObjectURL(frameURL);
  frameURL = null;
  ui.picture.hidden = true;
  ui["frame-id"].textContent = "F"+frame.index+" · "+frame.time_ms.toFixed(1)+" ms";
  ui.scrub.value = index;
  ui.prev.disabled = index === 0;
  ui.next.disabled = index === task.frames.length - 1;
  ui.scrub.disabled = false;
  ui.zoom.disabled = false;
  refresh();
  const button = ui.strip.children[index];
  const stripBounds = ui.strip.getBoundingClientRect();
  const buttonBounds = button.getBoundingClientRect();
  if (buttonBounds.right > stripBounds.right) ui.strip.scrollLeft += buttonBounds.right-stripBounds.right+12;
  if (buttonBounds.left < stripBounds.left) ui.strip.scrollLeft += buttonBounds.left-stripBounds.left-12;
  try {
    const blob = sources ? sources.get(frame.file) :
      session ? await session.frame(frame.file,controller.signal) : null;
    if (ticket !== loadTicket) return;
    frameURL = blob ? URL.createObjectURL(blob) : null;
    const url = frameURL ?? "./demo/" + frame.file;
    const pending = new Image();
    pending.src = url;
    await pending.decode();
    if (ticket !== loadTicket) return;
    if (pending.naturalWidth !== task.width || pending.naturalHeight !== task.height)
      throw new Error("프레임 해상도가 작업 목록과 다릅니다.");
    image.src = url;
    currentFrameLoaded = true;
    ui.empty.hidden = true;
    ui.picture.hidden = false;
    refresh();
  } catch {
    if (ticket === loadTicket)
      ui.notice.textContent = "프레임을 열 수 없습니다. 다른 프레임으로 이동한 뒤 돌아오거나 페이지를 다시 여세요.";
  }
}

function refresh() {
  const draft = drafts[task.id];
  const frame = task.frames[index].index;
  const mark = draft.marks[frame];
  const proposal = task.assistance?.points[frame];
  ui.controls.disabled = !currentFrameLoaded || submittingScene;
  ui.question.textContent = proposal ? "표시한 공 위치가 맞나요?" : "공이 손에서 떨어져 보이나요?";
  ui["proposal-status"].textContent = mark ? "사람이 확인한 기록입니다. 앞뒤 프레임으로 다시 확인할 수 있습니다." :
    correction ? "자동 후보를 거절했습니다. 위치를 고치거나 안 보임을 선택하세요." :
    proposal ? "자동 후보 · 아직 정답으로 확정하지 않았습니다." : "자동으로 찾지 못했습니다. 보이면 공 중심을 눌러주세요.";
  ui.accept.disabled = !proposal || correction;
  ui.reject.disabled = !proposal;
  ui["correction-hint"].hidden = !correction;
  ui["proposal-point"].hidden = !currentFrameLoaded || !proposal || !!mark || correction || !!draft.rejections?.[frame];
  if (proposal) {
    ui["proposal-point"].style.left = 100*proposal.x/task.width+"%";
    ui["proposal-point"].style.top = 100*proposal.y/task.height+"%";
  }
  ui["release-yes"].disabled = !releaseReady;
  ui["release-state"].textContent = draft.uncertain ? "안 보임으로 저장했습니다." :
    draft.contact != null && draft.free != null ? "확인한 범위: F"+draft.contact+" → F"+draft.free : "분리 후보 · 확인 전";
  ui.current.textContent = mark ? states[mark.kind] : "미판독";
  ui.point.hidden = !currentFrameLoaded || mark?.kind !== "ball";
  if (!ui.point.hidden) {
    ui.point.style.left = 100*mark.x/task.width + "%";
    ui.point.style.top = 100*mark.y/task.height + "%";
  }
  ui["contact-value"].textContent = draft.contact == null ? "미지정" : "F"+draft.contact;
  ui["free-value"].textContent = draft.free == null ? "미지정" : "F"+draft.free;
  ui.contact.disabled = draft.uncertain;
  ui.free.disabled = draft.uncertain;
  ui.uncertain.checked = draft.uncertain;
  if (ui.notes.value !== draft.notes) ui.notes.value = draft.notes;
  task.frames.forEach(({index:frame},i)=>{
    const button = ui.strip.children[i];
    const state = draft.marks[frame]?.kind;
    button.dataset.state = state ?? "";
    button.setAttribute("aria-label","프레임 "+frame+" · "+(states[state] ?? "미판독"));
    button.setAttribute("aria-current",i===index ? "true" : "false");
  });
  ui.progress.textContent = Object.keys(draft.marks).length+" / "+task.frames.length;
  let received = 0;
  const names = {complete:"제출 가능",needs_review:"제출 가능 · 검토 필요",in_progress:"진행 중",not_started:"시작 전"};
  manifest.tasks.forEach((t,i)=>{
    const status = sceneProgress(t,drafts[t.id]);
    const state = submissionState(drafts[t.id],sceneSubmissions[t.id]);
    const card = ui["scene-list"].children[i];
    card.dataset.status = status.status;
    card.setAttribute("aria-current",String(t.id===task.id));
    card.lastElementChild.textContent = status.marked+" / "+status.total+" · "+
      (state === "received" ? (sceneSubmissions[t.id].validation === "needs_review" ? "접수 완료 · 검토 필요" : "접수 완료") :
        state === "changed" ? "수정됨 · 다시 제출" : names[status.status]);
    card.title = status.issue;
    if (state === "received") received++;
  });
  ui["scene-total"].textContent = received+" / "+manifest.tasks.length+"개 씬 접수 · 판독은 자동 저장";
  const scene = sceneProgress(task,draft);
  const receiptState = submissionState(draft,sceneSubmissions[task.id]);
  ui["scene-status"].textContent = receiptState === "received" ?
    (sceneSubmissions[task.id].validation === "needs_review" ? "씬 접수 완료. 확인 사항도 함께 전달했습니다. 다음 씬으로 넘어가세요." : "씬 접수 완료. 다음 씬으로 넘어가세요.") :
    receiptState === "changed" ? "제출 후 수정했습니다. 이 씬을 다시 제출하세요." :
    scene.status === "complete" ? "프레임 판독을 마쳤습니다. 이 씬만 제출할 수 있습니다." :
    scene.status === "needs_review" ? "확인할 내용이 있지만 이 씬을 제출할 수 있습니다. "+scene.issue : scene.marked+" / "+scene.total+"프레임 확인";
  ui["inspect-issue"].hidden = scene.status !== "needs_review";
  ui["inspect-issue"].textContent = scene.frame == null ? "릴리스 범위 확인" : "F"+scene.frame+" 다시 확인";
  ui["next-scene"].hidden = receiptState !== "received" || received === manifest.tasks.length;
  const incomplete = manifest.tasks.find(t=>readiness(t,drafts[t.id]));
  const readerOK = /^[A-Za-z0-9_-]{1,32}$/.test(ui.reader.value.trim());
  ui.download.disabled = Boolean(incomplete) || !readerOK;
  ui["submit-all"].disabled = ui.download.disabled || submittingScene;
  ui.submit.disabled = !session || !currentFrameLoaded || submittingScene || scene.marked !== scene.total || receiptState === "received";
  ui.submit.textContent = submittingScene ? "제출 중…" : receiptState === "received" ? "이 씬 접수 완료" : receiptState === "changed" ? "수정한 씬 다시 제출" : "이 씬 제출";
  ui["scene-submit-help"].textContent = receiptState === "received" ? "접수 기록을 서버에 저장했습니다. 판독을 수정하면 다시 제출할 수 있습니다." :
    scene.marked !== scene.total ? "남은 "+(scene.total-scene.marked)+"프레임을 확인하면 이 씬을 제출할 수 있습니다. 안 보이는 공은 ‘안 보임’을 선택하세요." :
    "다른 씬을 끝내지 않아도 제출됩니다. 입력 불일치는 원래 값과 함께 검토용으로 전달합니다.";
  ui.readiness.textContent = incomplete ? incomplete.id+" · "+readiness(incomplete,drafts[incomplete.id]) :
    readerOK ? "내보낼 수 있습니다." : "영문·숫자로 판독자 코드를 입력하세요.";
}

function update(edit) {
  if (!currentFrameLoaded || submittingScene) return;
  edit(drafts[task.id],task.frames[index].index);
  if (session) session.save(drafts);
  else save();
  refresh();
}

function nextMissing() {
  const positions = task.frames.map((_,i)=>(index+1+i)%task.frames.length);
  const next = positions.find(i=>!drafts[task.id].marks[task.frames[i].index]);
  if (next != null) void seek(next);
  else ui.notice.textContent = "프레임은 모두 저장됐습니다. ‘이 씬 제출’을 눌러 판독을 전달하세요.";
}

function decide(decision,point) {
  if (!currentFrameLoaded || submittingScene) return;
  update((draft,frame)=>reviewFrame(task,draft,frame,decision,point));
  if (decision === "rejected") { correction = true; refresh(); }
  else nextMissing();
}

function mark(kind) { decide(kind); }

ui.folder.addEventListener("change", async event=>{
  const files = new Map(Array.from(event.target.files,file=>[file.webkitRelativePath.split("/").slice(1).join("/"),file]));
  try {
    if (!files.has("assignment.json")) throw new Error("assignment.json이 있는 작업 폴더를 선택하세요.");
    await mount(JSON.parse(await files.get("assignment.json").text()),files);
  } catch (error) { ui.notice.textContent = error.message; }
  finally { event.target.value = ""; }
});
ui.demo.addEventListener("click",()=>void mount(PitchPractice,null));
ui.task.addEventListener("change",()=>select(ui.task.value));
ui.prev.addEventListener("click",()=>seek(index-1));
ui.next.addEventListener("click",()=>seek(index+1));
ui.scrub.addEventListener("input",()=>seek(Number(ui.scrub.value)));
ui.zoom.addEventListener("click",()=>{
  const active = ui.picture.classList.toggle("zoom");
  ui.zoom.setAttribute("aria-pressed",active);
  ui.zoom.textContent = active ? "1× 원래 크기" : "2× 확대";
});
ui["frame-image"].addEventListener("click",event=>{
  const box = event.currentTarget.getBoundingClientRect();
  const x = Math.min(task.width-0.1,Math.max(0,Math.round((event.clientX-box.left)*task.width/box.width*10)/10));
  const y = Math.min(task.height-0.1,Math.max(0,Math.round((event.clientY-box.top)*task.height/box.height*10)/10));
  decide("ball",{x,y});
});
ui.accept.addEventListener("click",()=>decide("accepted"));
ui.reject.addEventListener("click",()=>decide("rejected"));
ui.unseen.addEventListener("click",()=>decide("unknown"));
ui.absent.addEventListener("click",()=>mark("absent"));
ui.unknown.addEventListener("click",()=>mark("unknown"));
ui.clear.addEventListener("click",()=>update((draft,frame)=>{ delete draft.marks[frame]; }));
ui.contact.addEventListener("click",()=>update((draft,frame)=>{ draft.contact = frame; draft.releaseReview = "manual"; }));
ui.free.addEventListener("click",()=>update((draft,frame)=>{ draft.free = frame; draft.releaseReview = "manual"; }));
ui.uncertain.addEventListener("change",()=>update(draft=>{ draft.uncertain = ui.uncertain.checked; draft.releaseReview = draft.uncertain ? "unknown" : "manual"; }));
for (const [i,side] of ["before","after"].entries()) {
  ui["release-"+side].addEventListener("click",()=>seek(task.frames.findIndex(f=>f.index===task.assistance.release_hint[i])));
}
ui["release-yes"].addEventListener("click",()=>{
  if (!releaseReady) return;
  update(draft=>{ [draft.contact,draft.free] = task.assistance.release_hint; draft.uncertain = false; draft.releaseReview = "accepted"; });
});
ui["release-no"].addEventListener("click",()=>{
  update(draft=>{ draft.contact = null; draft.free = null; draft.uncertain = false; draft.releaseReview = "manual"; });
  ui["release-details"].open = true;
});
ui["release-unknown"].addEventListener("click",()=>update(draft=>{ draft.uncertain = true; draft.releaseReview = "unknown"; }));
ui.notes.addEventListener("input",()=>update(draft=>{ draft.notes = ui.notes.value; }));
ui.reader.addEventListener("input",()=>{ if (task) { save(); refresh(); } });
ui.retry.addEventListener("click",()=>session?.flush());
ui["inspect-issue"].addEventListener("click",()=>{
  const progress = sceneProgress(task,drafts[task.id]);
  if (progress.frame != null) void seek(task.frames.findIndex(f=>f.index===progress.frame));
  else ui["release-details"].open = true;
});
ui["next-scene"].addEventListener("click",()=>{
  const next = manifest.tasks.find(t=>submissionState(drafts[t.id],sceneSubmissions[t.id]) !== "received");
  if (next) select(next.id);
});
ui.submit.addEventListener("click",async()=>{
  if (!session || ui.submit.disabled) return;
  const id = task.id;
  submittingScene = true;
  refresh();
  try {
    const ok = await session.save(drafts,false,id);
    ui.notice.textContent = ok ? "씬 접수 완료. 원래 판독과 확인 사항을 서버에 저장했습니다." : "제출을 완료하지 못했습니다. 아래 저장 상태를 확인하고 다시 저장을 눌러주세요.";
  } finally { submittingScene = false; refresh(); }
});
ui["submit-all"].addEventListener("click",()=>session?.save(drafts,true));
ui.download.addEventListener("click",()=>{
  try {
    const result = submission(manifest,drafts,ui.reader.value.trim());
    const url = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:"application/json"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = (manifest.practice ? "practice-" : "")+manifest.batch+"-"+result.reader+".json";
    link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  } catch (error) { ui.notice.textContent = error.message; }
});
document.addEventListener("keydown",event=>{
  if (!task || event.altKey || event.ctrlKey || event.metaKey ||
      ["INPUT","SELECT","TEXTAREA"].includes(event.target.tagName)) return;
  if (event.key === "ArrowLeft") { event.preventDefault(); seek(index-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); seek(index+1); }
  if (event.key === "0") mark("absent");
  if (event.key === "?") mark("unknown");
  if (event.repeat) return;
  if (event.key === "1" && !ui.accept.disabled) { event.preventDefault(); decide("accepted"); }
  if (event.key === "2" && !ui.reject.disabled) { event.preventDefault(); decide("rejected"); }
  if (event.key === "3") { event.preventDefault(); decide("unknown"); }
});

async function openAssignment(accessToken) {
  document.querySelector(".load").hidden = true;
  ui.notice.textContent = "서버에서 배정된 투구를 여는 중…";
  try {
    const online = PitchReviewSession(accessToken,showSaveStatus,record=>{
      sceneSubmissions = record.sceneSubmissions ?? {};
      if (task) refresh();
    });
    const remote = await online.open();
    if (!remote.assignment) throw new Error("이 링크에 배정된 서버 작업이 없습니다.");
    await mount(remote.assignment,null,online,remote);
    token = accessToken;
    try { localStorage.setItem(accountStorageKey,accessToken); } catch { /* The current session stays open. */ }
    history.replaceState(null,"",location.pathname+location.search);
    ui["access-code"].value = "";
    return true;
  } catch (error) {
    ui.notice.textContent = error.message;
    ui.portal.hidden = false;
    return false;
  }
}

byId("guide-link").addEventListener("click",event=>{
  event.preventDefault();
  byId("guide").scrollIntoView();
});
ui["login-form"].addEventListener("submit",async event=>{
  event.preventDefault();
  ui["login-submit"].disabled = true;
  ui["login-message"].textContent = "작업을 불러오는 중…";
  try {
    const response = await fetch("/api/session",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({reader:ui["login-reader"].value,accessCode:ui["access-code"].value.trim()})});
    if (!response.ok) throw new Error(response.status === 429 ? "접속 시도가 많습니다. 1분 뒤 다시 시도하세요." : "작업자와 접속 코드를 확인하세요.");
    const login = await response.json();
    if (!await openAssignment(login.token)) throw new Error("작업을 열지 못했습니다. 다시 시도하세요.");
  } catch (error) { ui["login-message"].textContent = error.message; }
  finally { ui["login-submit"].disabled = false; }
});
ui["change-reader"].addEventListener("click",()=>{
  if (session && !session.canLeave()) {
    ui.notice.textContent = "현재 기록의 서버 저장을 먼저 마쳐주세요. 저장 실패 시 다시 저장을 눌러주세요.";
    return;
  }
  try { localStorage.removeItem(accountStorageKey); } catch { /* Reload discards the active handle. */ }
  location.assign(location.pathname+location.search);
});

async function start() {
  try {
    const response = await fetch("/api/portal",{cache:"no-store"});
    if (response.ok) {
      const portal = await response.json();
      if (portal.enabled) {
        ui.portal.hidden = false;
        document.querySelector(".load").hidden = true;
        ui["login-reader"].replaceChildren(...portal.readers.map(reader=>new Option(reader,reader)));
        ui.notice.textContent = "본인의 작업자 번호로 접속하면 모든 씬과 진행 상태가 표시됩니다.";
      }
    }
  } catch { /* File-based and offline practice remain available. */ }
  if (token) await openAssignment(token);
}
void start();
