const test=require("node:test"),assert=require("node:assert/strict");
const {mkdtempSync,writeFileSync,rmSync}=require("node:fs");
const {join,resolve,sep}=require("node:path"),{tmpdir}=require("node:os");
const {prepareCorrections,reviewCorrection}=require("./corrections.cjs");
const {correctionState,acceptedCorrections}=require("./site/corrections.js");
const {makeServer}=require("./serve.cjs");
const task={id:"pitch",group:"QA",width:100,height:100,sha256:"a".repeat(64),frames:[1,2].map(index=>({index,time_ms:index*16,file:"frames/pitch/"+index+".png"}))};
const assignment={version:1,batch:"qa",tasks:[task]}, readers={R01:"a".repeat(48),R02:"b".repeat(48)};
const draft={marks:{1:{kind:"ball",x:20,y:30},2:{kind:"ball",x:40,y:50}},contact:null,free:null,uncertain:true,notes:"original"};
const candidate={id:"check-1",reader:"R01",scene:"pitch",frame:1,source_sha256:task.sha256,receipt_id:"receipt-1",
  original_mark:draft.marks[1],original_release:{contact:null,free:null,uncertain:true},
  proposed_mark:{kind:"held",review:"in_hand"},reason:"Original source shows the marked point moving with the hand; review required.",evidence_frames:[1,2]};
const config={schema:"pitch-correction-candidates/1",batch:"qa",candidates:[candidate]};
function original() {
  const receipt={id:"receipt-1",source_sha256:task.sha256,submittedAt:"before",draft:structuredClone(draft)};
  return {revision:1,drafts:{pitch:structuredClone(draft)},submittedAt:"before",result:{preserve:"original export"},
    sceneSubmissions:{pitch:receipt},sceneSubmissionHistory:[receipt]};
}

test("correction registry binds source, frame, reader and immutable content",()=>{
  const first=prepareCorrections(config,assignment,readers);
  assert.equal(first.length,1);assert.match(first[0].candidate_hash,/^[a-f0-9]{64}$/);
  assert.equal(first[0].candidate_hash,prepareCorrections(structuredClone(config),assignment,readers)[0].candidate_hash);
  const changed=structuredClone(config);changed.candidates[0].reason+=" revised";
  assert.notEqual(first[0].candidate_hash,prepareCorrections(changed,assignment,readers)[0].candidate_hash);
  for(const edit of [c=>c.source_sha256="b".repeat(64),c=>c.frame=3,c=>c.reader="other",c=>c.evidence_frames=[99],c=>c.proposed_mark.x=20,c=>c.id="__proto__"]) {
    const bad=structuredClone(config);edit(bad.candidates[0]);assert.throws(()=>prepareCorrections(bad,assignment,readers));
  }
  assert.throws(()=>prepareCorrections({...config,candidates:[candidate,candidate]},assignment,readers));
  assert.deepEqual(prepareCorrections(null,assignment,readers),[]);
});

test("accepted overlay preserves original observations and invalidates stale contexts",()=>{
  const [c]=prepareCorrections(config,assignment,readers),r=original(),before=structuredClone(r);
  assert.equal(correctionState(c,r),"pending");
  r.correctionReviews={[c.id]:reviewCorrection([c],"R01",r,{id:c.id,candidate_hash:c.candidate_hash,decision:"accept"},"now").review};
  assert.equal(correctionState(c,r),"accept");assert.equal(acceptedCorrections([c],r)[0].proposed_mark.kind,"held");
  assert.deepEqual(r.drafts,before.drafts);assert.deepEqual(r.sceneSubmissionHistory,before.sceneSubmissionHistory);
  for(const edit of [r=>r.drafts.pitch.marks[1].x++,r=>r.drafts.pitch.uncertain=false,r=>r.sceneSubmissions.pitch.id="new"]) {
    const stale=structuredClone(r);edit(stale);assert.equal(correctionState(c,stale),"stale");assert.deepEqual(acceptedCorrections([c],stale),[]);
  }
  assert.equal(correctionState({...c,candidate_hash:"changed"},r),"pending");
  assert.equal(reviewCorrection([c],"R02",r,{id:c.id,candidate_hash:c.candidate_hash,decision:"accept"},"now").status,400);
  assert.equal(reviewCorrection([c],"R01",r,{id:c.id,candidate_hash:"wrong",decision:"accept"},"now").status,400);
});

test("yes/no decisions append history without replacing drafts, receipts or submitted results",async()=>{
  const directory=mkdtempSync(join(tmpdir(),"pitch-corrections-")),saved=original();
  writeFileSync(join(directory,"qa-R01.json"),JSON.stringify(saved));
  const server=makeServer({assignment,readers,directory,corrections:config});
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const url="http://127.0.0.1:"+server.address().port+"/api/review";
  const headers=reader=>({Authorization:"Bearer "+readers[reader],"Content-Type":"application/json"});
  const get=async(reader="R01")=>(await fetch(url,{headers:headers(reader)})).json();
  const put=(revision,extra={},reader="R01")=>fetch(url,{method:"PUT",headers:headers(reader),body:JSON.stringify({batch:"qa",revision,final:false,drafts:saved.drafts,...extra})});
  try {
    let r=await get();const c=r.correctionCandidates[0],action={id:c.id,candidate_hash:c.candidate_hash,decision:"accept"};
    assert.equal((await get("R02")).correctionCandidates.length,0);
    assert.equal((await put(1,{correctionDecision:action})).status,200);
    r=await get();assert.deepEqual(r.drafts,saved.drafts);assert.deepEqual(r.sceneSubmissions,saved.sceneSubmissions);
    assert.deepEqual(r.sceneSubmissionHistory,saved.sceneSubmissionHistory);assert.deepEqual(r.result,saved.result);assert.equal(r.submittedAt,"before");
    assert.equal(r.correctionReviewHistory.length,1);assert.equal(r.correctionReviews[c.id].decision,"accept");
    assert.equal((await put(1,{correctionDecision:action})).status,409);
    const edited=structuredClone(saved.drafts);edited.pitch.marks[1].x++;
    assert.equal((await put(2,{drafts:edited,correctionDecision:action})).status,422);
    assert.equal((await put(2,{correctionDecision:{...action,decision:"reject"}})).status,200);
    r=await get();assert.equal(r.correctionReviewHistory.length,2);assert.deepEqual(acceptedCorrections(r.correctionCandidates,r),[]);
    assert.equal((await put(3,{correctionDecision:{...action,decision:"defer"}})).status,200);
    assert.equal((await get()).correctionReviews[c.id].decision,"defer");
    assert.equal((await put(4,{drafts:edited,correctionReviews:{}})).status,200);
    r=await get();assert.equal(r.correctionReviewHistory.length,3);assert.equal(r.correctionReviews[c.id].decision,"defer");
    assert.equal(correctionState(c,r),"stale");
    assert.equal((await put(5,{drafts:edited,correctionDecision:action})).status,422);
    assert.deepEqual((await get()).sceneSubmissions,saved.sceneSubmissions);
  } finally {
    await new Promise(resolve=>server.close(resolve));assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));rmSync(directory,{recursive:true,force:true});
  }
});
