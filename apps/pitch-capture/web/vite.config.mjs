import {defineConfig} from 'vite';
import {cpSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const require = createRequire(import.meta.url);
const root = dirname(fileURLToPath(import.meta.url));
cpSync(join(dirname(require.resolve('@mediapipe/tasks-vision')), 'wasm'),
  join(root, 'public', 'wasm'), {recursive:true});
export default defineConfig({base:'./', worker:{format:'iife'}});
