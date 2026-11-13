const {createServer} = require("node:http");
const {randomUUID} = require("node:crypto");
const {readFileSync, writeFileSync, mkdirSync, renameSync} = require("node:fs");
const {dirname, join} = require("node:path");
const {checkManifest, submission, validReview} = require("./site/labels.js");

const site = join(__dirname, "site");
const assets = {
  "/": ["index.html", "text/html"], "/app.js": ["app.js", "text/javascript"],
  "/sync.js": ["sync.js", "text/javascript"], "/labels.js": ["labels.js", "text/javascript"],
  "/style.css": ["style.css", "text/css"], "/demo/practice.js": ["demo/practice.js", "text/javascript"],
};
for (let frame=0; frame<8; frame++) {
  const file = "demo/frames/practice/"+frame+".png";
  assets["/"+file] = [file,"image/png"];
}
const plain = value => value != null && typeof value === "object" && !Array.isArray(value);
const empty = () => ({revision:0, drafts:{}, savedAt:null, submittedAt:null, result:null});

function loadRecord(file) {
  try { return JSON.parse(readFileSync(file,"utf8")); }
  catch (error) {
    if (error.code !== "ENOENT") throw error;
    return empty();
  }
}

function checkDrafts(assignment, drafts) {
  if (!plain(drafts) || Object.keys(drafts).length !== assignment.tasks.length) return false;
  return assignment.tasks.every(task => {
    const draft = drafts[task.id];
    if (!plain(draft) || !plain(draft.marks) || typeof draft.uncertain !== "boolean" ||
        typeof draft.notes !== "string" || draft.notes.length > 500) return false;
    const frames = new Set(task.frames.map(frame => String(frame.index)));
    if (![draft.contact,draft.free].every(frame => frame === null ||
        (Number.isInteger(frame) && frames.has(String(frame))))) return false;
    if (draft.rejections != null && (!plain(draft.rejections) ||
        Object.entries(draft.rejections).some(([frame,value])=>value !== true || !task.assistance?.points[frame]))) return false;
    if (draft.releaseReview != null && !["accepted","manual","unknown"].includes(draft.releaseReview)) return false;
    if (draft.releaseReview === "accepted" && (!task.assistance?.release_hint || draft.uncertain ||
        draft.contact !== task.assistance.release_hint[0] || draft.free !== task.assistance.release_hint[1])) return false;
    return Object.entries(draft.marks).every(([frame, mark]) => {
      if (!frames.has(frame) || !plain(mark) || !validReview(task,frame,mark)) return false;
      if (mark.kind === "unknown" || mark.kind === "absent") return true;
      return mark.kind === "ball" && Number.isFinite(mark.x) && Number.isFinite(mark.y) &&
        mark.x >= 0 && mark.x < task.width && mark.y >= 0 && mark.y < task.height;
    });
  });
}

function send(response, status, payload) {
  response.writeHead(status, {"Content-Type":"application/json; charset=utf-8","Cache-Control":"no-store"});
  response.end(JSON.stringify(payload));
}

function makeServer({assignment, readers, directory, frameDirectory}) {
  checkManifest(assignment);
  if (assignment.practice || !plain(readers)) throw new Error("A research assignment and readers are required");
  const byToken = new Map();
  for (const [reader, token] of Object.entries(readers)) {
    if (!/^[A-Za-z0-9_-]{1,32}$/.test(reader) || !/^[a-f0-9]{48}$/.test(token) || byToken.has(token))
      throw new Error("Invalid reader credential");
    byToken.set(token, reader);
  }
  const assignedFrames = new Set(assignment.tasks.flatMap(task => task.frames.map(frame => "/"+frame.file)));
  mkdirSync(directory,{recursive:true});
  return createServer(async (request, response) => {
    response.setHeader("X-Content-Type-Options","nosniff");
    response.setHeader("Referrer-Policy","no-referrer");
    response.setHeader("Content-Security-Policy",
      "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'");
    const path = request.url.split("?")[0];
    const frameRequest = path.startsWith("/api/frames/");
    if (path !== "/api/review" && !frameRequest) {
      const asset = Object.hasOwn(assets,path) ? assets[path] : null;
      if (!asset || !["GET","HEAD"].includes(request.method)) return send(response,404,{});
      const [file,type] = asset;
      const bytes = readFileSync(join(site,file));
      response.writeHead(200,{"Content-Type":type+(type.startsWith("text/") ? "; charset=utf-8" : ""),
        "Content-Length":bytes.length,"Cache-Control":"no-cache"});
      return response.end(request.method === "HEAD" ? undefined : bytes);
    }
    const reader = byToken.get(request.headers.authorization?.replace(/^Bearer /,""));
    if (!reader) return send(response,401,{error:"invalid_access"});
    const file = join(directory,assignment.batch+"-"+reader+".json");
    try {
      if (frameRequest) {
        const frame = path.slice("/api".length);
        if (!frameDirectory || !assignedFrames.has(frame) || !["GET","HEAD"].includes(request.method))
          return send(response,404,{});
        const bytes = readFileSync(join(frameDirectory,frame.slice(1)));
        response.writeHead(200,{"Content-Type":"image/png","Content-Length":bytes.length,"Cache-Control":"no-store"});
        return response.end(request.method === "HEAD" ? undefined : bytes);
      }
      if (request.method === "GET") {
        return send(response,200,{batch:assignment.batch,reader,assignment:frameDirectory ? assignment : null,...loadRecord(file)});
      }
      if (request.method !== "PUT") return send(response,405,{});
      if (request.headers["content-type"] !== "application/json") return send(response,415,{});
      let size = 0;
      const chunks = [];
      for await (const chunk of request) {
        size += chunk.length;
        if (size > 128*1024) return send(response,413,{});
        chunks.push(chunk);
      }
      let update;
      try { update = JSON.parse(Buffer.concat(chunks).toString("utf8")); }
      catch { return send(response,400,{error:"invalid_json"}); }
      if (!plain(update) || update.batch !== assignment.batch || !Number.isSafeInteger(update.revision) ||
          typeof update.final !== "boolean" || !checkDrafts(assignment,update.drafts))
        return send(response,400,{error:"invalid_draft"});
      const old = loadRecord(file);
      if (old.revision !== update.revision) return send(response,409,{error:"revision_conflict"});
      const savedAt = new Date().toISOString();
      const next = {revision:old.revision+1,drafts:update.drafts,savedAt,submittedAt:null,result:null,
        assistance:Object.fromEntries(assignment.tasks.map(task=>[task.id,task.assistance ?? null]))};
      if (update.final) {
        try { next.result = submission(assignment,update.drafts,reader); }
        catch { return send(response,422,{error:"incomplete_review"}); }
        next.submittedAt = savedAt;
      }
      const temporary = file+"."+randomUUID()+".tmp";
      writeFileSync(temporary,JSON.stringify(next,null,2),{encoding:"utf8",mode:0o600,flush:true});
      renameSync(temporary,file);
      return send(response,200,{revision:next.revision,savedAt,submittedAt:next.submittedAt});
    } catch {
      return send(response,500,{error:"save_failed"});
    }
  });
}

if (require.main === module) {
  const [assignmentFile, readersFile, directory, port = "8767"] = process.argv.slice(2);
  const read = file => JSON.parse(readFileSync(file,"utf8").replace(/^\uFEFF/,""));
  makeServer({assignment:read(assignmentFile),readers:read(readersFile),directory,frameDirectory:dirname(assignmentFile)})
    .listen(Number(port),"127.0.0.1",()=>console.log("Pitch review listening on 127.0.0.1:"+port));
}

module.exports = {makeServer};
