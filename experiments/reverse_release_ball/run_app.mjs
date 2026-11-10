import {readFileSync} from 'node:fs';
import {collectReview} from '../preserve_ball_path/collect.mjs';
import {preservePath} from '../preserve_ball_path/preserve.mjs';
import {recoverRelease} from './recover.mjs';

const {app,evidence,proposalEvidence,release,width,length} = collectReview(process.argv[2],readFileSync(0));
const reverse = recoverRelease(app.points,evidence,release,width);
process.stdout.write(JSON.stringify({proposalEvidence,models:{
  app,
  anchored:preservePath(app.points,evidence,release,width,length,{anchorGuard:true}),
  reverse:{...app,...reverse},
}}));
