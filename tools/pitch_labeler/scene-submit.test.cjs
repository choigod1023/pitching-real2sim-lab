const test = require("node:test");
const assert = require("node:assert/strict");
const {mkdtempSync,readFileSync,rmSync} = require("node:fs");
const {tmpdir} = require("node:os");
const {join} = require("node:path");
const {makeServer} = require("./serve.cjs");

test("submit one fully marked scene with unresolved labels while other scenes are untouched",async()=>{
  const directory = mkdtempSync(join(tmpdir(),"pitch-scene-submit-"));
  const tasks = ["one","two"].map(id=>({id,group:"QA",width:100,height:100,sha256:"a".repeat(64),
    frames:[164,165,166].map(index=>({index,time_ms:index*16,file:"frames/"+id+"/"+index+".png"}))}));
  const assignment = {version:1,batch:"qa",tasks};
  const server = makeServer({assignment,readers:{R01:"a".repeat(48)},directory});
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const url = "http://127.0.0.1:"+server.address().port+"/api/review";
  const headers = {Authorization:"Bearer "+"a".repeat(48),"Content-Type":"application/json"};
  const drafts = {
    one:{marks:{164:{kind:"ball",x:10,y:20},165:{kind:"absent"},166:{kind:"ball",x:20,y:30}},contact:165,free:166,notes:"",uncertain:false},
    two:{marks:{},contact:null,free:null,notes:"",uncertain:false},
  };
  const put = (revision,extra={})=>fetch(url,{method:"PUT",headers,body:JSON.stringify({batch:"qa",revision,final:false,drafts,...extra})});
  try {
    const response = await put(0,{submitScene:"one"});
    assert.equal(response.status,200);
    const first = await response.json();
    assert.ok(first.sceneSubmissions?.one?.submittedAt,"scene submission must return a receipt");
    assert.equal(first.sceneSubmissions.one.validation,"needs_review");
    assert.match(first.sceneSubmissions.one.issue,/F164/);
    assert.equal(first.sceneSubmissions.one.result,null);
    assert.deepEqual(first.sceneSubmissions.one.draft,drafts.one);
    assert.equal(first.submittedAt,null); // Never promote this to an all-scene validated result.
    assert.equal((await put(1,{submitScene:"two"})).status,422);
    assert.equal((await put(1,{submitScene:"missing"})).status,400);
    assert.equal((await put(1,{submitScene:"one",final:true})).status,400);
    assert.equal((await put(0,{submitScene:"one"})).status,409);
    drafts.one.marks[164] = {kind:"absent"};
    assert.equal((await put(1)).status,200);
    const edited = await (await fetch(url,{headers})).json();
    assert.equal(edited.sceneSubmissions.one.draft.marks[164].kind,"ball"); // Submitted snapshot survives edits.
    const second = await (await put(2,{submitScene:"one"})).json();
    assert.equal(second.sceneSubmissions.one.validation,"ready");
    assert.ok(second.sceneSubmissions.one.result);
    const saved = JSON.parse(readFileSync(join(directory,"qa-R01.json"),"utf8"));
    assert.equal(saved.sceneSubmissionHistory.length,2);
    assert.deepEqual(saved.drafts.two,drafts.two);
  } finally {
    await new Promise(resolve=>server.close(resolve));
    rmSync(directory,{recursive:true,force:true});
  }
});
