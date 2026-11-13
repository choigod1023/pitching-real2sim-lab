const test = require("node:test");
const assert = require("node:assert/strict");
const {mkdtempSync,rmSync,readdirSync} = require("node:fs");
const {tmpdir} = require("node:os");
const {join} = require("node:path");
const {sceneProgress} = require("./site/labels.js");
const {makeServer} = require("./serve.cjs");

const task = {id:"pitch",group:"QA",width:100,height:100,sha256:"a".repeat(64),
  frames:[164,165,166].map(index=>({index,time_ms:index*16,file:"frames/pitch/"+index+".png"}))};
const empty = ()=>({marks:{},contact:null,free:null,uncertain:false,notes:""});

test("a fully marked scene identifies the old pre-release point rather than declaring completion",()=>{
  const d = empty();
  assert.equal(sceneProgress(task,d).status,"not_started");
  d.marks[164] = {kind:"ball",x:30,y:20};
  assert.equal(sceneProgress(task,d).status,"in_progress");
  d.marks[165] = {kind:"absent"};
  d.marks[166] = {kind:"ball",x:40,y:30};
  d.contact = 165;
  d.free = 166;
  const progress = sceneProgress(task,d);
  assert.equal(progress.marked,3);
  assert.equal(progress.status,"needs_review");
  assert.equal(progress.frame,164);
  assert.deepEqual(progress.reviewFrames,[164]);
  assert.match(progress.issue,/F164/);
  assert.equal(d.marks[164].kind,"ball"); // The status calculation must not edit human labels.
  d.marks[164] = {kind:"absent"};
  assert.equal(sceneProgress(task,d).status,"complete");
  d.free = null;
  assert.equal(sceneProgress(task,d).status,"needs_review");
  assert.equal(sceneProgress(task,d).frame,null);
});

test("common entrance requires a code, selects the correct reader, and retains API authorization",async()=>{
  const directory = mkdtempSync(join(tmpdir(),"pitch-hub-test-"));
  const readers = {R01:"a".repeat(48),R02:"b".repeat(48)};
  const portal = {accessCode:"test-access-2026"};
  const assignment = {version:1,batch:"qa",tasks:[task]};
  const server = makeServer({assignment,readers,portal,directory});
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const url = "http://127.0.0.1:"+server.address().port;
  const login = data=>fetch(url+"/api/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
  try {
    const info = await (await fetch(url+"/api/portal")).json();
    assert.deepEqual(info,{enabled:true,readers:["R01","R02"]});
    assert.equal((await fetch(url+"/api/review")).status,401);
    assert.equal((await login({reader:"R01",accessCode:"wrong"})).status,401);
    assert.equal((await login({reader:"__proto__",accessCode:portal.accessCode})).status,401);
    for (const reader of Object.keys(readers)) {
      const response = await login({reader,accessCode:portal.accessCode});
      assert.equal(response.status,200);
      const data = await response.json();
      assert.equal(data.token,readers[reader]);
      const record = await (await fetch(url+"/api/review",{headers:{Authorization:"Bearer "+data.token}})).json();
      assert.equal(record.reader,reader);
      assert.equal(record.scenes.pitch.status,"not_started");
    }
    assert.deepEqual(readdirSync(directory),[]); // Login/read-only inspection never writes labels.
    assert.equal((await login({reader:"R01",accessCode:"x".repeat(5000)})).status,413);
    for (let i=0;i<16;i++) await login({reader:"R01",accessCode:"wrong"});
    assert.equal((await login({reader:"R01",accessCode:portal.accessCode})).status,429);
  } finally {
    await new Promise(resolve=>server.close(resolve));
    rmSync(directory,{recursive:true,force:true});
  }
});
