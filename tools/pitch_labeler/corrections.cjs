const {createHash} = require("node:crypto");
const {stableJSON,validReview} = require("./site/labels.js");
const {correctionState} = require("./site/corrections.js");

function prepareCorrections(config,assignment,readers) {
  if (config == null) return [];
  if (config.schema !== "pitch-correction-candidates/1" || config.batch !== assignment.batch || !Array.isArray(config.candidates))
    throw new Error("Invalid correction registry");
  const ids = new Set(), slots = new Set();
  return config.candidates.map(input=>{
    const c = JSON.parse(JSON.stringify(input));
    const task = assignment.tasks.find(t=>t.id===c.scene);
    const frames = new Set(task?.frames.map(f=>f.index));
    const slot = [c.reader,c.scene,c.frame].join("/");
    if (!/^[a-zA-Z0-9_-]{1,100}$/.test(c.id) || ["__proto__","constructor","prototype"].includes(c.id) || ids.has(c.id) || slots.has(slot) || !Object.hasOwn(readers,c.reader) ||
        !task || c.source_sha256 !== task.sha256 || !frames.has(c.frame) || typeof c.receipt_id !== "string" ||
        !c.receipt_id || c.original_mark?.kind !== "ball" || !validReview(task,c.frame,c.original_mark) ||
        !Number.isFinite(c.original_mark.x) || !Number.isFinite(c.original_mark.y) ||
        c.original_mark.x<0 || c.original_mark.x>=task.width || c.original_mark.y<0 || c.original_mark.y>=task.height ||
        !c.original_release || ![c.original_release.contact,c.original_release.free].every(f=>f===null || frames.has(f)) ||
        typeof c.original_release.uncertain !== "boolean" || !["held","caught","unknown"].includes(c.proposed_mark?.kind) ||
        !validReview(task,c.frame,c.proposed_mark) || c.proposed_mark.x != null || c.proposed_mark.y != null ||
        typeof c.reason !== "string" || !c.reason.trim() || c.reason.length>600 ||
        !Array.isArray(c.evidence_frames) || !c.evidence_frames.includes(c.frame) || c.evidence_frames.some(f=>!frames.has(f)))
      throw new Error("Invalid correction candidate");
    ids.add(c.id);slots.add(slot);
    delete c.candidate_hash;
    c.candidate_hash=createHash("sha256").update(stableJSON(c)).digest("hex");
    return c;
  });
}

function reviewCorrection(candidates,reader,record,action,at) {
  const candidate = candidates.find(c=>c.id===action?.id && c.reader===reader);
  if (!candidate || action.candidate_hash !== candidate.candidate_hash || !["accept","reject","defer"].includes(action.decision))
    return {error:"invalid_correction",status:400};
  if (correctionState(candidate,record)==="stale") return {error:"stale_correction",status:422};
  return {review:{candidate_id:candidate.id,candidate_hash:candidate.candidate_hash,decision:action.decision,at,
    candidate:JSON.parse(JSON.stringify(candidate))}};
}

module.exports={prepareCorrections,reviewCorrection};
