(function (root) {
  function reviewSession(token, status, onRecord = ()=>{}) {
    const headers = {"Authorization":"Bearer "+token, "Content-Type":"application/json"};
    let record, running = false, pending, conflict = false, generation = 0;
    let activePromise;
    const {stableJSON,submissionState} = root.PitchLabels;
    const key = "pitch-review-outbox:"+token;
    function backup() {
      try {
        if (pending) localStorage.setItem(key,JSON.stringify(pending));
        else localStorage.removeItem(key);
      } catch { /* The server remains the primary store. */ }
    }
    async function open() {
      const response = await fetch("/api/review",{headers,cache:"no-store"});
      if (!response.ok) throw new Error("작업자 링크를 확인하거나 서버에 다시 연결하세요.");
      record = await response.json();
      try { pending = JSON.parse(localStorage.getItem(key)); } catch { pending = null; }
      if (pending && pending.revision !== record.revision) {
        if (stableJSON(pending.drafts) === stableJSON(record.drafts)) {
          const submitted = pending.submitScene ? submissionState(pending.drafts[pending.submitScene],record.sceneSubmissions?.[pending.submitScene]) === "received" :
            !pending.final || Boolean(record.submittedAt);
          if (submitted) pending = null;
          else pending.revision = record.revision;
        } else conflict = true;
      }
      backup();
      return {...record,drafts:pending?.drafts ?? record.drafts};
    }
    function flush() {
      if (activePromise) return activePromise;
      if (conflict || !pending) return Promise.resolve(!conflict);
      activePromise = sendAll().finally(()=>{ activePromise = null; });
      return activePromise;
    }
    async function sendAll() {
      running = true;
      try {
        while (pending) {
          const request = pending;
          const sent = generation;
          status("서버에 저장 중…");
          const response = await fetch("/api/review",{
            method:"PUT",headers,body:JSON.stringify({batch:record.batch,...request}),
          });
          if (response.status === 409) {
            conflict = true;
            status("다른 탭에서 기록이 바뀌었습니다. 이 탭의 기록을 덮어쓰지 않았습니다.");
            return false;
          }
          if (!response.ok) throw new Error("Save failed");
          record = {...record,...await response.json()};
          onRecord(record);
          if (sent === generation) pending = null;
          else pending.revision = record.revision;
          backup();
          if (!pending) report();
        }
        return true;
      } catch {
        status("서버 저장 실패. 연결을 확인하고 ‘다시 저장’을 누르세요.");
        return false;
      } finally { running = false; }
    }
    function report() {
      if (conflict) status("서버 기록과 이 탭의 임시 기록이 다릅니다. 관리자에게 알려 주세요.");
      else if (pending) void flush();
      else if (record?.submittedAt) status("서버 제출 완료");
      else if (record?.savedAt) status("서버 자동 저장 완료");
      else status("서버 연결됨 · 판독할 때마다 저장");
    }
    return {open,report,flush,canLeave:()=>!pending && !running && !conflict,async frame(file,signal) {
      const response = await fetch("/api/"+file,{headers,signal,cache:"no-store"});
      if (!response.ok) throw new Error("Frame unavailable");
      return response.blob();
    },save(drafts,final=false,submitScene) {
      if (!record) return Promise.resolve(false);
      generation++;
      pending = {revision:record.revision,drafts:JSON.parse(JSON.stringify(drafts)),final};
      if (submitScene != null) pending.submitScene = submitScene;
      backup();
      if (conflict) { report(); return Promise.resolve(false); }
      return flush();
    }};
  }
  root.PitchReviewSession = reviewSession;
})(globalThis);
