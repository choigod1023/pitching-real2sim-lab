import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const destination = resolve(root, 'web/public/models/pose_landmarker_full.task');
const url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const expected = '5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1';
const valid = bytes => bytes && createHash('sha256').update(bytes).digest('hex') === expected;

let bytes = await readFile(destination).catch(error => {
  if (error.code !== 'ENOENT') throw error;
  return undefined;
});
if (!valid(bytes)) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`MediaPipe download failed (${response.status})`);
  bytes = Buffer.from(await response.arrayBuffer());
  if (!valid(bytes)) throw new Error('MediaPipe model checksum does not match.');
}
await mkdir(dirname(destination),{recursive:true});
await writeFile(destination,bytes);
console.log('MediaPipe Pose Landmarker Full: verified for web.');
