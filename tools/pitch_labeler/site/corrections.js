(function(root) {
  const labels = typeof module === "object" ? require("./labels.js") : root.PitchLabels;
  function correctionState(candidate, record) {
    const draft = record.drafts?.[candidate.scene];
    const receipt = record.sceneSubmissions?.[candidate.scene];
    const release = draft && {contact:draft.contact,free:draft.free,uncertain:draft.uncertain};
    if (!draft || receipt?.id !== candidate.receipt_id || receipt.source_sha256 !== candidate.source_sha256 ||
        labels.stableJSON(draft.marks[candidate.frame]) !== labels.stableJSON(candidate.original_mark) ||
        labels.stableJSON(release) !== labels.stableJSON(candidate.original_release)) return "stale";
    const review = record.correctionReviews?.[candidate.id];
    return review?.candidate_hash === candidate.candidate_hash ? review.decision : "pending";
  }
  function acceptedCorrections(candidates,record) {
    return candidates.filter(c=>correctionState(c,record)==="accept").map(c=>({
      id:c.id,scene:c.scene,frame:c.frame,original_mark:c.original_mark,proposed_mark:c.proposed_mark,
      source_sha256:c.source_sha256,receipt_id:c.receipt_id,candidate_hash:c.candidate_hash,
    }));
  }
  const api = {correctionState,acceptedCorrections};
  root.PitchCorrections = api;
  if (typeof module === "object") module.exports = api;
})(globalThis);
