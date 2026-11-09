import {OfflineBallReview} from '@pitch/core/offline-ball.mjs';

export async function reviewRecordedBall(blob, {start,end,anchors,gates,metres}, onProgress) {
  const {ALL_FORMATS,BlobSource,Input,VideoSampleSink} = await import('mediabunny');
  const input = new Input({source:new BlobSource(blob),formats:ALL_FORMATS});
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track || !await track.canDecode()) {
      throw new Error('이 브라우저가 녹화 원본의 코덱을 디코딩하지 못합니다.');
    }
    const canvas = document.createElement('canvas');
    const pixels = canvas.getContext('2d',{willReadFrequently:true});
    let review = null, decoded = 0;
    for await (const sample of new VideoSampleSink(track).samples(start,end)) {
      try {
        if (sample.timestamp < start || sample.timestamp >= end) continue;
        if (!review) {
          const scale = Math.min(1,960/Math.max(sample.displayWidth,sample.displayHeight));
          const width = canvas.width = Math.round(sample.displayWidth*scale);
          const height = canvas.height = Math.round(sample.displayHeight*scale);
          const pixelPoint = p => ({x:p.x*width,y:p.y*height});
          const [a,b] = anchors.map(pixelPoint);
          review = new OfflineBallReview(width,height,a,b,metres,{gates:gates?.map(pixelPoint)});
        }
        sample.draw(pixels,0,0,canvas.width,canvas.height);
        review.push(pixels.getImageData(0,0,canvas.width,canvas.height).data,sample.timestamp);
        if (++decoded % 15 === 0) onProgress(decoded);
      } finally { sample.close(); }
    }
    if (!review) throw new Error('선택한 구간에 디코딩된 프레임이 없습니다.');
    return review.finish();
  } finally { input.dispose(); }
}
