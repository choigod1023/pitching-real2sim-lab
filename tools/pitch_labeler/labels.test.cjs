const assert = require("node:assert/strict");
const test = require("node:test");
const {checkManifest,readiness,submission} = require("./site/labels.js");

const task = {
  id:"pitch-1",group:"MLB",width:1280,height:720,sha256:"a".repeat(64),
  frames:[10,11,12].map((index,i)=>({index,time_ms:16.7*i,file:"frames/pitch-1/"+index+".png"})),
};
const manifest = {version:1,batch:"batch-1",tasks:[task]};
const draft = () => ({
  marks:{10:{kind:"absent"},11:{kind:"unknown"},12:{kind:"ball",x:50.2,y:20}},
  contact:10,free:12,uncertain:false,notes:"",
});

test("checks source metadata and requires an ordered, unmodified frame sequence", ()=>{
  assert.equal(checkManifest(manifest),manifest);
  assert.throws(()=>checkManifest({...manifest,tasks:[task,task]}));
  assert.throws(()=>checkManifest({...manifest,tasks:[{...task,frames:[task.frames[0],task.frames[2]]}]}));
  assert.throws(()=>checkManifest({...manifest,tasks:[{...task,frames:[task.frames[0],
    {...task.frames[1],file:"frames/pitch-1/10.png"},task.frames[2]]}]}));
});

test("keeps coordinates, unknowns, and release bounds in research frame indices", ()=>{
  const clip = submission(manifest,{"pitch-1":draft()},"R01").clips[0];
  assert.deepEqual(clip.release,[10,11]);
  assert.equal(clip.first_confirmed_free,12);
  assert.deepEqual(clip.positive,[[12,50.2,20]]);
  assert.deepEqual(clip.negative,[10]);
  assert.deepEqual(clip.unknown,[11]);
});

test("allows explicit uncertainty while refusing incomplete or contradictory records", ()=>{
  const result = draft();
  result.contact = 12;
  assert.notEqual(readiness(task,result),"");
  result.uncertain = true;
  assert.equal(submission(manifest,{"pitch-1":result},"R01").clips[0].release,null);
  result.uncertain = false;
  result.contact = 10;
  result.marks[11] = {kind:"ball",x:50,y:20};
  assert.notEqual(readiness(task,result),"");
  delete result.marks[11];
  assert.notEqual(readiness(task,result),"");
  assert.throws(()=>submission(manifest,{"pitch-1":draft()},"이름"));
});
