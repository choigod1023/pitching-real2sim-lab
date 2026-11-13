const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const labels = require("./site/labels.js");
const source = readFileSync(join(__dirname,"site/sync.js"),"utf8");
const drafts = {one:{marks:{1:{kind:"unknown"}},contact:null,free:null,uncertain:true,notes:""}};
const response = data=>({ok:true,status:200,json:async()=>data});

function client(fetch,initial={}) {
  const storage = new Map(Object.entries(initial));
  const messages = [], receipts = [];
  const context = {fetch,PitchLabels:labels,localStorage:{
    getItem:key=>storage.get(key) ?? null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key),
  }};
  vm.runInNewContext(source,context);
  return {session:context.PitchReviewSession("qa",s=>messages.push(s),r=>receipts.push(r)),storage,messages,receipts};
}

test("scene submission waits for the in-flight autosave and receives the final receipt",async()=>{
  let release, entered;
  const started = new Promise(resolve=>{entered=resolve;});
  const requests = [];
  const receipt = {submittedAt:"now",draft:drafts.one,validation:"needs_review"};
  const c = client(async (_url,options)=>{
    if (!options.method) return response({batch:"qa",revision:0,drafts:{},submittedAt:null});
    const request = JSON.parse(options.body);
    requests.push(request);
    if (requests.length === 1) {
      entered();
      await new Promise(resolve=>{release=resolve;});
      return response({revision:1,submittedAt:null,sceneSubmissions:{}});
    }
    return response({revision:2,submittedAt:null,sceneSubmissions:{one:receipt}});
  });
  await c.session.open();
  const saving = c.session.save(drafts);
  await started;
  const submitting = c.session.save(drafts,false,"one");
  assert.equal(c.session.canLeave(),false);
  release();
  assert.deepEqual(await Promise.all([saving,submitting]),[true,true]);
  assert.equal(requests.length,2);
  assert.equal(requests[1].revision,1);
  assert.equal(requests[1].submitScene,"one");
  assert.equal(c.receipts.at(-1).sceneSubmissions.one.submittedAt,"now");
  assert.equal(c.session.canLeave(),true);
  assert.equal(c.storage.size,0);
});

test("failed scene submission stays in the outbox and retry preserves the submission action",async()=>{
  let attempts = 0;
  const requests = [];
  const c = client(async (_url,options)=>{
    if (!options.method) return response({batch:"qa",revision:0,drafts:{},submittedAt:null});
    requests.push(JSON.parse(options.body));
    if (++attempts === 1) throw new Error("offline");
    return response({revision:1,submittedAt:null,sceneSubmissions:{one:{submittedAt:"now",draft:drafts.one}}});
  });
  await c.session.open();
  assert.equal(await c.session.save(drafts,false,"one"),false);
  assert.equal(c.session.canLeave(),false);
  assert.equal(JSON.parse(c.storage.get("pitch-review-outbox:qa")).submitScene,"one");
  assert.equal(await c.session.flush(),true);
  assert.equal(requests[1].submitScene,"one");
  assert.equal(c.session.canLeave(),true);
});

test("reload recovers a lost acknowledgement only when the submitted snapshot exists",async()=>{
  const pending = {revision:0,drafts,final:false,submitScene:"one"};
  const c = client(async()=>response({batch:"qa",revision:1,drafts,submittedAt:null,
    sceneSubmissions:{one:{submittedAt:"now",draft:drafts.one}}}),{"pitch-review-outbox:qa":JSON.stringify(pending)});
  await c.session.open();
  assert.equal(c.session.canLeave(),true);
  assert.equal(c.storage.size,0);
  const c2 = client(async()=>response({batch:"qa",revision:1,drafts,submittedAt:null}),
    {"pitch-review-outbox:qa":JSON.stringify(pending)});
  await c2.session.open();
  assert.equal(c2.session.canLeave(),false);
  const restored = JSON.parse(c2.storage.get("pitch-review-outbox:qa"));
  assert.equal(restored.revision,1);
  assert.equal(restored.submitScene,"one");
});

test("receipt matching ignores object key order but detects changed annotations",()=>{
  const receipt = {submittedAt:"now",draft:{b:2,a:1}};
  assert.equal(labels.submissionState({a:1,b:2},receipt),"received");
  assert.equal(labels.submissionState({a:1,b:3},receipt),"changed");
  assert.equal(labels.submissionState({},undefined),"none");
});
