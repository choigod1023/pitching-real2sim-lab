const byId = id => document.getElementById(id);
const ui = Object.fromEntries([
  "folder","demo","notice","task-title","frame-id","viewport","picture","frame-image","point","empty",
  "prev","next","scrub","zoom","strip","progress","task","reader","controls","current","absent","unknown",
  "clear","contact","free","contact-value","free-value","uncertain","notes","download","readiness",
].map(id => [id, byId(id)]));
const {checkManifest, readiness, submission} = PitchLabels;
const states = {ball:"공 중심 표시됨", absent:"자유 비행 없음", unknown:"판독 불가"};
let manifest, task, drafts, sources, index = 0, frameURL, currentFrameLoaded = false, loadTicket = 0;
let storageKey = "";
let storageFailed = false;

function save() {
  try {
    localStorage.setItem(storageKey, JSON.stringify({reader:ui.reader.value,drafts}));
  } catch {
    if (!storageFailed) ui.notice.textContent += " 브라우저 임시 저장을 쓸 수 없습니다. 이 탭에서 작업을 마치세요.";
    storageFailed = true;
  }
}

function mount(data, files) {
  checkManifest(data);
  if (files) {
    for (const task of data.tasks) {
      for (const frame of task.frames) {
        if (!files.has(frame.file)) throw new Error("작업 폴더에 필요한 프레임 파일이 없습니다.");
      }
    }
  }
  manifest = data;
  sources = files;
  storageKey = "pitch-labels-v1:" + JSON.stringify([data.batch,data.tasks.map(t=>[t.id,t.sha256,t.frames[0].index,t.frames.at(-1).index])]);
  let previous;
  try { previous = JSON.parse(localStorage.getItem(storageKey) || "null"); } catch { previous = null; }
  drafts = Object.fromEntries(data.tasks.map(t=>[t.id,previous?.drafts?.[t.id] ??
    {marks:{},contact:null,free:null,uncertain:false,notes:""}]));
  ui.reader.value = previous?.reader ?? "";
  ui.task.replaceChildren(...data.tasks.map(t=>new Option(t.id+" · "+t.group,t.id)));
  ui.task.disabled = false;
  ui.notice.textContent = data.practice ? "연습용 합성 프레임입니다. 연습 결과는 연구 판독에 포함하지 않습니다." :
    "영상은 브라우저에서만 엽니다. 모든 작업을 판독한 뒤 결과 JSON을 의뢰자에게 보내세요.";
  select(data.tasks[0].id);
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
  seek(0);
}

function seek(target) {
  index = Math.max(0, Math.min(task.frames.length - 1, target));
  const frame = task.frames[index];
  currentFrameLoaded = false;
  const image = ui["frame-image"];
  image.removeAttribute("src");
  if (frameURL) URL.revokeObjectURL(frameURL);
  frameURL = sources ? URL.createObjectURL(sources.get(frame.file)) : null;
  const url = frameURL ?? "./demo/" + frame.file;
  const ticket = ++loadTicket;
  const pending = new Image();
  pending.onload = ()=>{
    if (ticket !== loadTicket) return;
    if (pending.naturalWidth !== task.width || pending.naturalHeight !== task.height) {
      ui.notice.textContent = "프레임 해상도가 작업 목록과 다릅니다. 원래 작업 ZIP을 다시 여세요.";
      return;
    }
    image.src = url;
    currentFrameLoaded = true;
    ui.empty.hidden = true;
    ui.picture.hidden = false;
    refresh();
  };
  pending.onerror = ()=>{
    if (ticket !== loadTicket) return;
    ui.notice.textContent = "프레임을 열 수 없습니다. 작업 폴더를 다시 확인하세요.";
  };
  ui.picture.hidden = true;
  pending.src = url;
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
}

function refresh() {
  const draft = drafts[task.id];
  const frame = task.frames[index].index;
  const mark = draft.marks[frame];
  ui.controls.disabled = !currentFrameLoaded;
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
  ui.readiness.textContent = incomplete ? incomplete.id+" · "+readiness(incomplete,drafts[incomplete.id]) :
    readerOK ? "내보낼 수 있습니다." : "영문·숫자로 판독자 코드를 입력하세요.";
}

function update(edit) {
  if (!currentFrameLoaded) return;
  edit(drafts[task.id],task.frames[index].index);
  save();
  refresh();
}

function mark(kind) { update((draft,frame)=>{ draft.marks[frame] = {kind}; }); }

ui.folder.addEventListener("change", async event=>{
  const files = new Map(Array.from(event.target.files,file=>[file.webkitRelativePath.split("/").slice(1).join("/"),file]));
  try {
    if (!files.has("assignment.json")) throw new Error("assignment.json이 있는 작업 폴더를 선택하세요.");
    mount(JSON.parse(await files.get("assignment.json").text()),files);
  } catch (error) { ui.notice.textContent = error.message; }
  finally { event.target.value = ""; }
});
ui.demo.addEventListener("click",()=>mount(PitchPractice,null));
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
  update((draft,frame)=>{ draft.marks[frame] = {kind:"ball",x,y}; });
});
ui.absent.addEventListener("click",()=>mark("absent"));
ui.unknown.addEventListener("click",()=>mark("unknown"));
ui.clear.addEventListener("click",()=>update((draft,frame)=>{ delete draft.marks[frame]; }));
ui.contact.addEventListener("click",()=>update((draft,frame)=>{ draft.contact = frame; }));
ui.free.addEventListener("click",()=>update((draft,frame)=>{ draft.free = frame; }));
ui.uncertain.addEventListener("change",()=>update(draft=>{ draft.uncertain = ui.uncertain.checked; }));
ui.notes.addEventListener("input",()=>update(draft=>{ draft.notes = ui.notes.value; }));
ui.reader.addEventListener("input",()=>{ if (task) { save(); refresh(); } });
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
});
