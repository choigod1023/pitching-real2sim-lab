import assert from 'node:assert/strict';
import test from 'node:test';
import {linkRapsodoRecord, parseRapsodoCsv, videoSpeedKmh} from './replay-metrics.mjs';

test('Rapsodo records convert explicit units and retain missing outcomes', () => {
  const [first,second] = parseRapsodoCsv('\uFEFFGame Pitch,Pitch Created At,RelSpeed (MPH),SpinRate,InducedVertBreak (Inches),PlateLocHeight (Inches)\r\n' +
    '4,"Nov 10, 2026",79.4,1893,13.9,26\r\n8,,77.9,1893,-1.2,\r\n');
  assert.equal(first.pitchNumber,'4');
  assert.equal(first.recordedAt,'Nov 10, 2026');
  assert.equal(first.releaseKmh,79.4*1.609344);
  assert.equal(first.rpm,1893);
  assert.equal(first.ivbCm,13.9*2.54);
  assert.equal(first.plateHeightCm,66.04);
  assert.equal(second.plateHeightCm,null);
  assert.equal(second.ivbCm,-1.2*2.54);
  assert.equal(second.row,3);
});

test('explicit metric columns, quoted text and zero readings are accepted', () => {
  const [record] = parseRapsodoCsv('Game Pitch,Pitch Created At,RelSpeed (km/h),SpinRate,InducedVertBreak (cm)\n' +
    '1,"entry ""A""",125.4,0,0\n');
  assert.equal(record.releaseKmh,125.4);
  assert.equal(record.recordedAt,'entry "A"');
  assert.equal(record.rpm,0);
  assert.equal(record.ivbCm,0);
});

test('rejects unknown units, malformed rows and invalid measurements', () => {
  assert.throws(() => parseRapsodoCsv('Game Pitch,RelSpeed\n4,79.4'), /Rapsodo/);
  assert.throws(() => parseRapsodoCsv('Game Pitch,SpinRate\n4,"1893'), /따옴표/);
  assert.throws(() => parseRapsodoCsv('Game Pitch,SpinRate\n,1893'), /투구 번호/);
  assert.throws(() => parseRapsodoCsv('Game Pitch,SpinRate\n4,-1'), /올바르지/);
  assert.throws(() => parseRapsodoCsv('Game Pitch,RelSpeed (MPH)\n4,0'), /올바르지/);
});

test('same CSV record cannot silently become the reference for two pitches in one session', () => {
  const source = {name:'report.csv',sha256:'sha'};
  const rows = parseRapsodoCsv('Game Pitch,SpinRate\n4,1900\n8,2000');
  const before = linkRapsodoRecord({},'a',rows[0],source);
  assert.equal(before.a.match,'user_confirmed');
  assert.throws(() => linkRapsodoRecord(before,'b',rows[0],source), /이미/);
  assert.equal(linkRapsodoRecord(before,'b',rows[1],source).b.record.pitchNumber,'8');
  assert.equal(before.b,undefined);
  assert.equal(linkRapsodoRecord(before,'a',rows[1],source).a.record.pitchNumber,'8');
});

test('only a bracketed speed with confirmed capture timing is displayed', () => {
  const result = {speedStatus:'available',kmh:104.5};
  assert.equal(videoSpeedKmh({result}),null);
  assert.equal(videoSpeedKmh({result,captureTimingConfirmed:false}),null);
  assert.equal(videoSpeedKmh({result,captureTimingConfirmed:true}),104.5);
  assert.equal(videoSpeedKmh({result:{...result,speedStatus:'unbracketed_gates'},captureTimingConfirmed:true}),null);
});
