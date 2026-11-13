(function (root) {
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
    }
    return manifest;
  }

  function readiness(task, draft) {
    for (const frame of task.frames) {
      const mark = draft.marks[frame.index];
      if (!mark) return "지정된 프레임을 모두 판독하세요.";
      if (!["ball", "absent", "unknown"].includes(mark.kind) ||
          (mark.kind === "ball" && !(mark.x >= 0 && mark.x < task.width && mark.y >= 0 && mark.y < task.height)))
        return "공 좌표 또는 판독 상태를 확인하세요.";
    }
    if (draft.uncertain) return "";
    if (draft.contact == null || draft.free == null || draft.contact >= draft.free)
      return "마지막 접촉과 첫 분리 프레임을 순서대로 지정하세요.";
    if (draft.marks[draft.contact]?.kind !== "absent" || draft.marks[draft.free]?.kind !== "ball")
      return "접촉에는 ‘자유 비행 없음’, 첫 분리에는 공 중심을 표시하세요.";
    if (task.frames.some(({index}) => index < draft.free && draft.marks[index].kind === "ball"))
      return "첫 분리 이전에 찍은 공 중심을 확인하세요.";
    return "";
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
        if (mark.kind === "absent") negative.push(index);
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
      };
    });
    return {schema:"pitch-video-annotation/1", batch:manifest.batch, practice:Boolean(manifest.practice), reader, clips};
  }

  root.PitchLabels = {checkManifest, readiness, submission};
  if (typeof module === "object") module.exports = root.PitchLabels;
})(globalThis);
