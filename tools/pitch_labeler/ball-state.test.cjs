const test = require("node:test");
const assert = require("node:assert/strict");
const {mkdtempSync,rmSync} = require("node:fs");
const {tmpdir} = require("node:os");
const {join,resolve,sep} = require("node:path");
const {reviewFrame,readiness,sceneProgress,submission,validReview,releasePair} = require("./site/labels.js");
const {makeServer} = require("./serve.cjs");
const task = {id:"pitch",group:"QA",width:100,height:100,sha256:"a".repeat(64),
  frames:[1,2,3,4].map(index=>({index,time_ms:index*16,file:"frames/pitch/"+index+".png"})),
  assistance:{method:"fixture",points:{3:{kind:"ball",x:40,y:30}},release_hint:[1,3]}};
const manifest = {version:1,batch:"state-qa",tasks:[task]};
function draft() {
  const d = {marks:{},contact:1,free:3,uncertain:false,notes:""};
  reviewFrame(task,d,1,"held"); reviewFrame(task,d,2,"unknown");
  reviewFrame(task,d,3,"accepted"); reviewFrame(task,d,4,"caught");
  return d;
}

test("release preview uses the selected frame pair instead of a stale automatic hint",()=>{
  const d = draft(); d.contact = 2;
  assert.deepEqual(releasePair(task,d),{frames:[2,3],source:"reader"});
  assert.deepEqual(task.assistance.release_hint,[1,3]);
  d.uncertain = true;
  assert.deepEqual(releasePair(task,d).frames,[2,3]); // Displaying it does not clear uncertainty.
  d.contact = null;
  assert.deepEqual(releasePair(task,d),{frames:[1,3],source:"model"});
  assert.equal(releasePair({...task,assistance:null},d),null);
  d.contact=500;
  assert.equal(releasePair(task,d).source,"model");
});

test("held, unresolved and caught states never export as flight coordinates",()=>{
  const d = draft(), out = submission(manifest,{pitch:d},"QA").clips[0];
  assert.equal(readiness(task,d),"");
  assert.deepEqual(out.positive,[[3,40,30]]);
  assert.deepEqual(out.negative,[1,4]);
  assert.deepEqual(out.unknown,[2]);
  assert.equal(out.human_review.marks[1].kind,"held");
  assert.equal(out.human_review.marks[4].kind,"caught");
  assert.equal(out.human_review.marks[1].x,undefined);
  assert.deepEqual(out.release,[1,2]);
  d.marks[1] = {kind:"absent",review:"no_free"};
  assert.equal(readiness(task,d),""); // Keep legacy records without inventing a phase.
});

test("wrong contact identity and impossible phase order remain review issues",()=>{
  const d = draft();
  reviewFrame(task,d,1,"caught");
  assert.notEqual(readiness(task,d),"");
  assert.ok(sceneProgress(task,d).reviewFrames.includes(1));
  reviewFrame(task,d,1,"held"); reviewFrame(task,d,4,"held");
  assert.notEqual(readiness(task,d),"");
  assert.ok(sceneProgress(task,d).reviewFrames.includes(4));
  assert.equal(validReview(task,1,{kind:"held",review:"accepted"}),false);
  assert.equal(validReview(task,1,{kind:"caught",review:"in_hand"}),false);
});

test("unclear hand-ball boundary stays unknown and is not promoted to contact",()=>{
  const d = draft(); reviewFrame(task,d,1,"unknown");
  assert.notEqual(readiness(task,d),"");
  d.uncertain = true;
  const clip = submission(manifest,{pitch:d},"QA").clips[0];
  assert.equal(clip.release,null);
  assert.equal(clip.last_confirmed_contact,null);
  assert.deepEqual(clip.unknown,[1,2]);
});

test("server persists separate states and rejects flight coordinates attached to held marks",async()=>{
  const directory = mkdtempSync(join(tmpdir(),"pitch-ball-state-"));
  const server = makeServer({assignment:manifest,readers:{QA:"a".repeat(48)},directory});
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const url = "http://127.0.0.1:"+server.address().port+"/api/review";
  const headers = {Authorization:"Bearer "+"a".repeat(48),"Content-Type":"application/json"};
  const d = draft();
  try {
    const put = revision=>fetch(url,{method:"PUT",headers,body:JSON.stringify({batch:manifest.batch,revision,final:false,drafts:{pitch:d},submitScene:"pitch"})});
    assert.equal((await put(0)).status,200);
    const saved = await (await fetch(url,{headers})).json();
    assert.equal(saved.drafts.pitch.marks[1].kind,"held");
    assert.equal(saved.drafts.pitch.marks[4].kind,"caught");
    assert.equal(saved.sceneSubmissions.pitch.validation,"ready");
    d.marks[1].x=20;d.marks[1].y=20;
    assert.equal((await put(1)).status,400);
    assert.equal((await (await fetch(url,{headers})).json()).revision,1);
  } finally {
    await new Promise(resolve=>server.close(resolve));
    assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));
    rmSync(directory,{recursive:true,force:true});
  }
});
