const test = require("node:test");
const assert = require("node:assert/strict");
const {mkdtempSync,mkdirSync,readFileSync,rmSync,writeFileSync} = require("node:fs");
const {tmpdir} = require("node:os");
const {join} = require("node:path");
const {makeServer} = require("./serve.cjs");

test("isolates readers, saves and reloads drafts, and validates submissions and revisions", async ()=>{
  const directory = mkdtempSync(join(tmpdir(),"pitch-labels-"));
  const assignment = {version:1,batch:"test-batch",tasks:[{
    id:"pitch",group:"QA",width:100,height:100,sha256:"a".repeat(64),
    frames:[1,2].map(index=>({index,time_ms:index*16,file:"frames/pitch/"+index+".png"})),
    assistance:{method:"proposal_union",points:{2:{kind:"ball",x:50,y:25}},release_hint:[1,2]},
  }]};
  const readers = {R01:"a".repeat(48),R02:"b".repeat(48)};
  const frameDirectory = join(directory,"assignment");
  const framePath = join(frameDirectory,"frames","pitch");
  mkdirSync(framePath,{recursive:true});
  const frameBytes = Buffer.from([137,80,78,71]);
  writeFileSync(join(framePath,"1.png"),frameBytes);
  writeFileSync(join(framePath,"3.png"),frameBytes);
  const config = {assignment,readers,directory,frameDirectory};
  let server;
  let url;
  async function start() {
    server = makeServer(config);
    await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
    url = "http://127.0.0.1:"+server.address().port;
  }
  const stop = ()=>new Promise(resolve=>server.close(resolve));
  const request = (method,reader,data)=>fetch(url+"/api/review",{
    method,headers:{"Authorization":"Bearer "+readers[reader],"Content-Type":"application/json"},
    body:data && JSON.stringify({batch:assignment.batch,...data}),
  });
  await start();
  try {
    assert.equal((await request("GET","invalid")).status,401);
    const route = url+"/api/frames/pitch/";
    const auth = {headers:{"Authorization":"Bearer "+readers.R01}};
    assert.equal((await fetch(route+"1.png")).status,401);
    const frame = await fetch(route+"1.png",auth);
    assert.equal(frame.headers.get("content-type"),"image/png");
    assert.deepEqual(Buffer.from(await frame.arrayBuffer()),frameBytes);
    assert.equal((await fetch(route+"3.png",auth)).status,404);
    assert.deepEqual((await (await request("GET","R01")).json()).assignment,assignment);
    const drafts = {pitch:{marks:{1:{kind:"absent"}},contact:1,free:null,notes:"",uncertain:false}};
    assert.equal((await request("PUT","R01",{drafts,revision:0,final:true})).status,422);
    assert.equal((await request("PUT","R01",{drafts,revision:0,final:false})).status,200);
    assert.equal((await request("PUT","R01",{drafts,revision:0,final:false})).status,409);
    assert.deepEqual((await (await request("GET","R02")).json()).drafts,{});
    assert.equal((await fetch(url+"/test-batch-R01.json")).status,404);
    await stop();
    await start();
    const restored = await (await request("GET","R01")).json();
    assert.deepEqual(restored.drafts,drafts);
    assert.equal(restored.revision,1);
    drafts.pitch.free = 2;
    drafts.pitch.marks[2] = {kind:"ball",x:51,y:25,review:"accepted"};
    assert.equal((await request("PUT","R01",{drafts,revision:1,final:true})).status,400);
    drafts.pitch.marks[2] = {kind:"ball",x:50,y:25,review:"accepted"};
    const final = await request("PUT","R01",{drafts,revision:1,final:true});
    assert.equal(final.status,200);
    const saved = JSON.parse(readFileSync(join(directory,"test-batch-R01.json"),"utf8"));
    assert.ok(saved.submittedAt);
    assert.deepEqual(saved.result.clips[0].release,[1,1]);
    assert.deepEqual(saved.result.clips[0].positive,[[2,50,25]]);
    assert.equal(saved.result.clips[0].review_mode,"machine_assisted");
    assert.deepEqual(saved.assistance.pitch,assignment.tasks[0].assistance);
    assert.equal(saved.drafts.pitch.marks[1].review,undefined); // Old record preserved.
  } finally {
    await stop();
    rmSync(directory,{recursive:true,force:true});
  }
});
