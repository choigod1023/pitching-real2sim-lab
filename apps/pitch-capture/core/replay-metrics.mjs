const MPH_TO_KMH = 1.609344;
const IN_TO_CM = 2.54;

function csvRows(text) {
  const rows = [];
  let row = [], cell = '', quoted = false, closed = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char !== '"') cell += char;
      else if (text[i+1] === '"') { cell += '"'; i++; }
      else { quoted = false; closed = true; }
    } else if (char === ',' || char === '\r' || char === '\n') {
      row.push(cell.trim());
      cell = '';
      closed = false;
      if (char !== ',') {
        if (char === '\r' && text[i+1] === '\n') i++;
        rows.push(row);
        row = [];
      }
    } else if (char === '"' && !cell && !closed) {
      quoted = true;
    } else if (char === '"' || (closed && char !== ' ' && char !== '\t')) {
      throw new Error('CSV의 따옴표 형식을 확인하세요.');
    } else if (!closed) {
      cell += char;
    }
  }
  if (quoted) throw new Error('CSV에서 닫히지 않은 따옴표를 확인하세요.');
  if (row.length || cell || closed) rows.push([...row,cell.trim()]);
  return rows;
}

const COLUMNS = {
  releaseKmh:[['relspeed (mph)',MPH_TO_KMH],['relspeed (km/h)',1]],
  rpm:[['spinrate',1]],
  ivbCm:[['inducedvertbreak (inches)',IN_TO_CM],['inducedvertbreak (cm)',1]],
  plateHeightCm:[['platelocheight (inches)',IN_TO_CM],['platelocheight (cm)',1]],
};

export function parseRapsodoCsv(text) {
  const [header, ...rows] = csvRows(text.replace(/^\uFEFF/,''));
  const columns = new Map(header?.map((label,index) => [label.toLowerCase(),index]));
  if (!columns.has('game pitch') ||
    !Object.values(COLUMNS).flat().some(([name]) => columns.has(name))) {
    throw new Error('Game Pitch와 단위가 표시된 Rapsodo 기록 열이 필요합니다.');
  }
  const records = [];
  for (const [index,row] of rows.entries()) {
    if (!row.some(Boolean)) continue;
    const field = name => row[columns.get(name)] ?? '';
    const pitchNumber = field('game pitch');
    if (!pitchNumber) throw new Error('CSV의 투구 번호(Game Pitch)가 비어 있습니다.');
    const measurements = {};
    for (const [key,choices] of Object.entries(COLUMNS)) {
      const source = choices.find(([name]) => field(name));
      if (!source) { measurements[key] = null; continue; }
      const [name,scale] = source;
      const raw = field(name);
      const value = Number(raw);
      if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(raw) || !Number.isFinite(value) ||
          (key === 'releaseKmh' && value <= 0) || (key !== 'ivbCm' && value < 0)) {
        throw new Error('CSV에서 투구 ' + pitchNumber + '의 ' + name + ' 값이 올바르지 않습니다.');
      }
      measurements[key] = value * scale;
    }
    if (Object.values(measurements).some(value => value !== null)) {
      records.push({row:index+2,pitchNumber,recordedAt:field('pitch created at') || null,...measurements});
    }
  }
  if (!records.length) throw new Error('측정값이 있는 투구를 찾지 못했습니다.');
  return records;
}

export function linkRapsodoRecord(references, pitchId, record, source) {
  for (const [linkedPitchId,reference] of Object.entries(references ?? {})) {
    if (linkedPitchId !== pitchId && reference.source.sha256 === source.sha256 && reference.record.row === record.row) {
      throw new Error('이 CSV 행은 같은 영상의 다른 투구에 이미 연결되어 있습니다.');
    }
  }
  return {...references,[pitchId]:{kind:'rapsodo_record',match:'user_confirmed',record,source}};
}

export function videoSpeedKmh(review) {
  return review?.captureTimingConfirmed === true && review.result?.speedStatus === 'available' ?
    review.result.kmh : null;
}
