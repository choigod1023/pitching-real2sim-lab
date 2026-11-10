import {readFileSync} from 'node:fs';
import {releaseTrack} from '../release_conditioned_ball/track.mjs';
import {collectReview} from './collect.mjs';
import {preservePath} from './preserve.mjs';

const {app,evidence,proposalEvidence,release,width,length} = collectReview(process.argv[2],readFileSync(0));
process.stdout.write(JSON.stringify({proposalEvidence,models:{
  app, release_beam:releaseTrack(evidence,release,length,width),
  preserved:preservePath(app.points,evidence,release,width,length),
  anchored:preservePath(app.points,evidence,release,width,length,{anchorGuard:true}),
}}));
