const byId = id => document.getElementById(id);
const ui = Object.fromEntries([
  "folder","demo","notice","task-title","frame-id","viewport","picture","frame-image","point","empty",
  "prev","next","scrub","zoom","strip","progress","task","reader","controls","current","absent","unknown",
  "clear","contact","free","contact-value","free-value","uncertain","notes","download","readiness","save-status","retry","submit",
  "proposal-point","proposal-status","question","accept","reject","unseen","correction-hint",
  "release-assist","release-before","release-after","release-before-image","release-after-image",
  "release-before-label","release-after-label","release-help","release-yes","release-no","release-unknown","release-state","release-details",
].map(id => [id, byId(id)]));
const {checkManifest, readiness, submission, reviewFrame} = PitchLabels;
const states = {ball:"공 중심 표시됨", absent:"자유 비행 없음", unknown:"판독 불가"};
let manifest, task, drafts, sources, index = 0, frameURL, currentFrameLoaded = false, loadTicket = 0;
let storageKey = "";
let storageFailed = false;
let session;
let frameRequest;
let correction = false, releaseReady = false, releaseTicket = 0, releaseURLs = [];
const token = new URLSearchParams(location.hash.slice(1)).get("key");

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
  ui["save-status"].hidden = !online;
  ui.retry.hidden = true;
  ui.submit.hidden = !online;
  ui.reader.readOnly = Boolean(online);
  manifest = data;
  sources = files;
  storageKey = "pitch-labels-v1:" + JSON.stringify([data.batch,data.tasks.map(t=>[t.id,t.sha256,t.frames[0].index,t.frames.at(-1).index])]);
  let previous;
  try { previous = JSON.parse(localStorage.getItem(storageKey) || "null"); } catch { previous = null; }
  drafts = Object.fromEntries(data.tasks.map(t=>[t.id,(online ? remote.drafts[t.id] : previous?.drafts?.[t.id]) ??
    {marks:{},contact:null,free:null,uncertain:false,notes:""}]));
  ui.reader.value = remote?.reader ?? previous?.reader ?? "";
  ui.task.replaceChildren(...data.tasks.map(t=>new Option(t.id+" · "+t.group,t.id)));
  ui.task.disabled = false;
  ui.notice.textContent = data.practice ? "연습용 합성 프레임입니다. 연습 결과는 연구 판독에 포함하지 않습니다." :
    online ? "자동 후보를 먼저 확인하세요. 맞음·틀림·안 보임으로 답하면 서버에 자동 저장됩니다." :
      "영상은 브라우저에서만 엽니다. 모든 작업을 판독한 뒤 결과 JSON을 의뢰자에게 보내세요.";
  select(data.tasks[0].id);
  session?.report();
}

function select(id) {
  task = manifest.tasks.find(entry=>entry.id===id);
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
  seek(firstProposal >= 0 ? firstProposal : Math.max(0,firstMissing));
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
  ui.controls.disabled = !currentFrameLoaded;
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
  const incomplete = manifest.tasks.find(t=>readiness(t,drafts[t.id]));
  const readerOK = /^[A-Za-z0-9_-]{1,32}$/.test(ui.reader.value.trim());
  ui.download.disabled = Boolean(incomplete) || !readerOK;
  ui.submit.disabled = ui.download.disabled;
  ui.readiness.textContent = incomplete ? incomplete.id+" · "+readiness(incomplete,drafts[incomplete.id]) :
    readerOK ? "내보낼 수 있습니다." : "영문·숫자로 판독자 코드를 입력하세요.";
}

function update(edit) {
  if (!currentFrameLoaded) return;
  edit(drafts[task.id],task.frames[index].index);
  if (session) session.save(drafts);
  else save();
  refresh();
}

function nextMissing() {
  const positions = task.frames.map((_,i)=>(index+1+i)%task.frames.length);
  const next = positions.find(i=>!drafts[task.id].marks[task.frames[i].index]);
  if (next != null) void seek(next);
  else ui.notice.textContent = "이 투구의 프레임 확인을 마쳤습니다. 릴리스 범위를 확인한 뒤 작업 번호에서 다음 투구를 여세요.";
}

function decide(decision,point) {
  if (!currentFrameLoaded) return;
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
ui.submit.addEventListener("click",()=>session?.save(drafts,true));
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

async function openAssignment() {
  if (!token) return;
  document.querySelector(".load").hidden = true;
  byId("guide-link").addEventListener("click",event=>{
    event.preventDefault();
    byId("guide").scrollIntoView();
  });
  ui.notice.textContent = "서버에서 배정된 투구를 여는 중…";
  try {
    const online = PitchReviewSession(token,showSaveStatus);
    const remote = await online.open();
    if (!remote.assignment) throw new Error("이 링크에 배정된 서버 작업이 없습니다.");
    await mount(remote.assignment,null,online,remote);
  } catch (error) { ui.notice.textContent = error.message; }
}

void openAssignment();
