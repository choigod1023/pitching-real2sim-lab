const test = require("node:test");
const assert = require("node:assert/strict");
const {checkManifest,readiness,submission,reviewFrame,validReview} = require("./site/labels.js");

const task = {id:"pitch",group:"QA",width:100,height:80,sha256:"a".repeat(64),
  frames:[1,2,3].map(index=>({index,time_ms:index*16,file:"frames/pitch/"+index+".png"})),
  assistance:{method:"proposal_union",points:{2:{kind:"ball",x:40,y:20}},release_hint:[1,2]}};
const manifest = {version:1,batch:"qa",tasks:[task]};
const draft = ()=>({marks:{},contact:null,free:null,uncertain:true,notes:""});

test("model points are not completed labels and require a human acceptance",()=>{
  const d = draft();
  checkManifest(manifest);
  assert.notEqual(readiness(task,d),"");
  assert.deepEqual(d.marks,{});
  assert.throws(()=>reviewFrame(task,d,1,"accepted"));
  reviewFrame(task,d,2,"accepted");
  assert.deepEqual(d.marks[2],{kind:"ball",x:40,y:20,review:"accepted"});
  assert.notEqual(readiness(task,d),"");
  assert.equal(validReview(task,2,{kind:"ball",x:41,y:20,review:"accepted"}),false);
});

test("rejection stays pending; correction and unknown are distinct and retained",()=>{
  const d = draft();
  reviewFrame(task,d,2,"accepted");
  reviewFrame(task,d,2,"rejected");
  assert.equal(d.rejections[2],true);
  assert.equal(d.marks[2],undefined);
  reviewFrame(task,d,2,"ball",{x:45,y:23});
  assert.equal(d.marks[2].review,"corrected");
  assert.equal(task.assistance.points[2].x,40);
  reviewFrame(task,d,1,"absent");
  reviewFrame(task,d,3,"unknown");
  const clip = submission(manifest,{pitch:d},"QA").clips[0];
  assert.equal(clip.review_mode,"machine_assisted");
  assert.deepEqual(clip.positive,[[2,45,23]]);
  assert.deepEqual(clip.unknown,[3]);
  assert.equal(clip.machine_proposals.points[2].x,40);
  assert.equal(clip.human_review.rejections[2],true);
});

test("assistance rejects out of window and invalid coordinates but preserves old manual records",()=>{
  const changed = structuredClone(manifest);
  changed.tasks[0].assistance.points[4] = {kind:"ball",x:1,y:1};
  assert.throws(()=>checkManifest(changed));
  const d = draft();
  d.marks[2] = {kind:"ball",x:12,y:18};
  reviewFrame(task,d,1,"absent");
  reviewFrame(task,d,3,"unknown");
  assert.equal(readiness(task,d),"");
  assert.equal(d.marks[2].x,12);
  assert.throws(()=>reviewFrame(task,d,3,"ball",{x:NaN,y:1}));
});
