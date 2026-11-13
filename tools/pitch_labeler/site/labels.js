(function (root) {
  function stableJSON(value) {
    if (Array.isArray(value)) return "["+value.map(stableJSON).join(",")+"]";
    if (value && typeof value === "object") return "{"+Object.keys(value).sort().map(key=>JSON.stringify(key)+":"+stableJSON(value[key])).join(",")+"}";
    return JSON.stringify(value);
  }

  function submissionState(draft, receipt) {
    if (!receipt?.submittedAt || !receipt.draft) return "none";
    return stableJSON(draft) === stableJSON(receipt.draft) ? "received" : "changed";
  }
  function releasePair(task, draft) {
    const frames = new Set(task.frames.map(f=>f.index));
    if (frames.has(draft.contact) && frames.has(draft.free) && draft.contact < draft.free)
      return {frames:[draft.contact,draft.free],source:"reader"};
    return task.assistance?.release_hint ? {frames:task.assistance.release_hint,source:"model"} : null;
  }
  function checkManifest(manifest) {
    if (!manifest || manifest.version !== 1 || !/^[\w-]{1,64}$/.test(manifest.batch ?? "") || !Array.isArray(manifest.tasks) || !manifest.tasks.length)
      throw new Error("작업 목록 형식을 확인하세요.");
    const ids = new Set();
    for (const task of manifest.tasks) {
      if (!task || !/^[\w-]{1,64}$/.test(task.id ?? "") || ids.has(task.id) || typeof task.group !== "string" ||
          !Number.isInteger(task.width) || task.width < 1 ||
          !Number.isInteger(task.height) || task.height < 1 ||
          !/^[0-9a-f]{64}$/.test(task.sha256) || !Array.isArray(task.frames) || task.frames.length < 2 ||
          task.frames.some((frame, i, all) => !frame || !Number.isInteger(frame.index) || frame.index < 0 ||
            !Number.isFinite(frame.time_ms) || (i > 0 && (frame.index !== all[i - 1].index + 1 ||
              frame.time_ms <= all[i - 1].time_ms)) || frame.file !== "frames/" + task.id + "/" + frame.index + ".png"))
        throw new Error("작업 번호, 프레임 또는 원본 정보에 오류가 있습니다.");
      ids.add(task.id);
      if (task.assistance) {
        const a = task.assistance;
        const frames = new Set(task.frames.map(frame=>String(frame.index)));
        if (typeof a.method !== "string" || !a.points || Array.isArray(a.points) ||
            Object.entries(a.points).some(([frame,p])=>!frames.has(frame) || p?.kind !== "ball" ||
              !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.x >= task.width || p.y < 0 || p.y >= task.height) ||
            (a.release_hint != null && (!Array.isArray(a.release_hint) || a.release_hint.length !== 2 ||
              a.release_hint.some(frame=>!Number.isInteger(frame) || !frames.has(String(frame))) ||
              a.release_hint[0] >= a.release_hint[1]))) throw new Error("자동 후보 정보를 확인하세요.");
      }
    }
    return manifest;
  }

  function reviewFrame(task, draft, frame, decision, point) {
    if (!task.frames.some(f=>f.index===frame)) throw new Error("Unknown frame");
    const proposal = task.assistance?.points[frame];
    if (decision === "accepted") {
      if (!proposal) throw new Error("No candidate to accept");
      draft.marks[frame] = {...proposal,review:"accepted"};
      if (draft.rejections) delete draft.rejections[frame];
    } else if (decision === "rejected") {
      if (!proposal) throw new Error("No candidate to reject");
      (draft.rejections ??= {})[frame] = true;
      // A rejection alone is not a confirmed negative or a replacement coordinate.
      delete draft.marks[frame];
    } else if (decision === "ball") {
      if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) ||
          point.x < 0 || point.x >= task.width || point.y < 0 || point.y >= task.height) throw new Error("Invalid point");
      draft.marks[frame] = {kind:"ball",x:point.x,y:point.y,review:proposal ? "corrected" : "manual"};
    } else if (["absent","held","caught","unknown"].includes(decision)) {
      draft.marks[frame] = {kind:decision,review:({unknown:"unseen",absent:"no_free",held:"in_hand",caught:"caught"})[decision]};
    } else throw new Error("Unknown decision");
  }

  function validReview(task, frame, mark) {
    if (mark.review == null) return true; // Preserve older manual records.
    if (mark.review === "accepted") {
      const p = task.assistance?.points[frame];
      return !!p && mark.kind === "ball" && p.x === mark.x && p.y === mark.y;
    }
    return (mark.kind === "ball" && ["corrected","manual"].includes(mark.review)) ||
      (mark.kind === "unknown" && mark.review === "unseen") || (mark.kind === "absent" && mark.review === "no_free") ||
      (mark.kind === "held" && mark.review === "in_hand") || (mark.kind === "caught" && mark.review === "caught");
  }

  function readiness(task, draft) {
    for (const frame of task.frames) {
      const mark = draft.marks[frame.index];
      if (!mark) return "지정된 프레임을 모두 판독하세요.";
      if (!validReview(task,frame.index,mark) || !["ball", "absent", "held", "caught", "unknown"].includes(mark.kind) ||
          (mark.kind === "ball" && !(mark.x >= 0 && mark.x < task.width && mark.y >= 0 && mark.y < task.height)))
        return "공 좌표 또는 판독 상태를 확인하세요.";
    }
    if (draft.uncertain) return "";
    if (draft.contact == null || draft.free == null || draft.contact >= draft.free)
      return "마지막 접촉과 첫 분리 프레임을 순서대로 지정하세요.";
    if (!["held","absent"].includes(draft.marks[draft.contact]?.kind) || draft.marks[draft.free]?.kind !== "ball")
      return "마지막 접촉은 ‘손에 있음’, 첫 분리는 ‘날아가는 중’으로 확인하세요. 흐려서 구분되지 않으면 릴리스 범위를 판독 불가로 남기세요.";
    const early = task.frames.find(({index}) => index < draft.free && draft.marks[index].kind === "ball");
    if (early) return "F"+early.index+"에 공 위치가 있지만 첫 분리는 F"+draft.free+"입니다. F"+early.index+" 프레임을 다시 확인하세요.";
    const phaseConflict = task.frames.find(({index}) =>
      (index >= draft.free && draft.marks[index].kind === "held") || (index <= draft.contact && draft.marks[index].kind === "caught"));
    if (phaseConflict) return "F"+phaseConflict.index+"의 손·글러브 상태와 릴리스 순서가 다릅니다. 원영상을 다시 확인하세요.";
    return "";
  }

  function sceneProgress(task, draft) {
    if (!draft) return {marked:0,total:task.frames.length,status:"not_started",issue:"아직 시작하지 않았습니다.",frame:task.frames[0].index,reviewFrames:[]};
    const marked = task.frames.filter(f=>draft.marks[f.index]).length;
    const issue = readiness(task,draft);
    const missing = task.frames.find(f=>!draft.marks[f.index]);
    const early = !draft.uncertain && draft.free != null && task.frames.find(f=>f.index < draft.free && draft.marks[f.index]?.kind === "ball");
    const inconsistentContact = draft.contact != null && !["held","absent"].includes(draft.marks[draft.contact]?.kind);
    const inconsistentFree = draft.free != null && draft.marks[draft.free]?.kind !== "ball";
    const reviewFrames = draft.uncertain ? [] : [...new Set([
      ...task.frames.filter(f=>draft.free != null && f.index < draft.free && draft.marks[f.index]?.kind === "ball").map(f=>f.index),
      ...task.frames.filter(f=>(draft.free != null && f.index >= draft.free && draft.marks[f.index]?.kind === "held") ||
        (draft.contact != null && f.index <= draft.contact && draft.marks[f.index]?.kind === "caught")).map(f=>f.index),
      ...(inconsistentContact ? [draft.contact] : []), ...(inconsistentFree ? [draft.free] : []),
    ])].sort((a,b)=>a-b);
    return {marked,total:task.frames.length,status:!issue ? "complete" : !marked ? "not_started" : marked < task.frames.length ? "in_progress" : "needs_review",
      issue,reviewFrames,frame:missing?.index ?? (inconsistentContact ? draft.contact : inconsistentFree ? draft.free : early?.index ?? reviewFrames[0] ?? null)};
  }

  function submission(manifest, drafts, reader) {
    if (!/^[A-Za-z0-9_-]{1,32}$/.test(reader)) throw new Error("영문·숫자로 판독자 코드를 입력하세요.");
    const clips = manifest.tasks.map(task => {
      const draft = drafts[task.id];
      const problem = readiness(task, draft);
      if (problem) throw new Error(task.id + " · " + problem);
      const positive = [], negative = [], unknown = [];
      for (const {index} of task.frames) {
        const mark = draft.marks[index];
        if (mark.kind === "ball") positive.push([index, mark.x, mark.y]);
        if (["absent","held","caught"].includes(mark.kind)) negative.push(index);
        if (mark.kind === "unknown") unknown.push(index);
      }
      // The research tracker scans starting at release[1] + 1. A clearly separated frame is already free.
      return {
        id:task.id, group:task.group, sha256:task.sha256, width:task.width, height:task.height,
        window:[task.frames[0].index, task.frames.at(-1).index],
        release:draft.uncertain ? null : [draft.contact, draft.free - 1],
        last_confirmed_contact:draft.uncertain ? null : draft.contact,
        first_confirmed_free:draft.uncertain ? null : draft.free,
        positive, negative, unknown, notes:draft.notes,
        review_mode:task.assistance ? "machine_assisted" : "unassisted",
        machine_proposals:task.assistance ?? null,
        human_review:JSON.parse(JSON.stringify(draft)),
      };
    });
    return {schema:"pitch-video-annotation/1", batch:manifest.batch, practice:Boolean(manifest.practice), reader, clips};
  }

  root.PitchLabels = {checkManifest, readiness, submission, reviewFrame, validReview, sceneProgress, stableJSON, submissionState, releasePair};
  if (typeof module === "object") module.exports = root.PitchLabels;
})(globalThis);
