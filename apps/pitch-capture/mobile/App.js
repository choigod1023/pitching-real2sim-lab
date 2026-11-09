import React, {useCallback, useEffect, useRef, useState} from 'react';
import {AppState, Pressable, ScrollView, StyleSheet, Text, View} from 'react-native';
import {SafeAreaProvider, SafeAreaView} from 'react-native-safe-area-context';
import {Camera, useCameraDevice, useCameraFormat, useCameraPermission} from 'react-native-vision-camera';
import {Delegate, RunningMode, usePoseDetection} from 'react-native-mediapipe-posedetection';
import {useVideoPlayer, VideoView} from 'expo-video';
import Svg, {Circle, Line} from 'react-native-svg';
import {BONES, JOINTS, PitchDetector, comparisonError, confirmed, markEvent, nearestPose,
  normalizePose, phaseTime} from '@pitch/core';
import {keepVideo, loadSessions, saveSession} from './archive';

const A = '#75e4d3', B = '#ffd17e';
const id = () => Date.now().toString(36) + '-' + Math.random().toString(36).slice(2,10);
const formatTime = time => time == null ? '미확인' : `${time.toFixed(3)}초`;

function Button({title, onPress, active = false, disabled = false}) {
  return <Pressable accessibilityRole="button" accessibilityState={{disabled}}
    disabled={disabled} onPress={onPress} style={[styles.button,active && styles.active,disabled && styles.disabled]}>
    <Text style={styles.buttonText}>{title}</Text>
  </Pressable>;
}

function Skeleton({layers, aspect}) {
  const height = 1000/aspect;
  return <Svg pointerEvents="none" style={StyleSheet.absoluteFill}
    width="100%" height="100%" viewBox={`0 0 1000 ${height}`} preserveAspectRatio="xMidYMid meet">
    {layers.flatMap(([points,color],layer) => points ? [
      ...BONES.map(([a,b]) => points[a] && points[b] &&
        <Line key={`${layer}-bone-${a}-${b}`} x1={points[a][0]*1000} y1={points[a][1]*height}
          x2={points[b][0]*1000} y2={points[b][1]*height} stroke={color} strokeWidth="5"/>),
      ...JOINTS.map(joint => points[joint] &&
        <Circle key={`${layer}-joint-${joint}`} cx={points[joint][0]*1000} cy={points[joint][1]*height}
          r="5" fill={color}/>),
    ] : [])}
  </Svg>;
}

function LiveCamera({side, onSession, onStatus, onBusy}) {
  const camera = useRef(null), recording = useRef(null), timeout = useRef(null);
  const device = useCameraDevice('back');
  const format = useCameraFormat(device,[{videoResolution:{width:1280,height:720}},{fps:30}]);
  const permission = useCameraPermission();
  const [ready,setReady] = useState(false), [inProgress,setInProgress] = useState(false);
  const [foreground,setForeground] = useState(AppState.currentState === 'active');
  const [pose,setPose] = useState(null), [aspect,setAspect] = useState(16/9), [count,setCount] = useState(0);

  useEffect(() => {
    const listener = AppState.addEventListener('change', state => setForeground(state === 'active'));
    return () => { listener.remove(); clearTimeout(timeout.current); };
  }, []);

  const onResults = useCallback(result => {
    const points = normalizePose(result.results[0]?.landmarks[0]);
    const ratio = result.inputImageWidth/result.inputImageHeight;
    const frameAspect = Number.isFinite(ratio) && ratio > 0 ? ratio : 16/9;
    setPose(points);
    setAspect(frameAspect);
    const active = recording.current;
    if (!active) return;
    // The native detector reports inference duration. The video start offset
    // is approximate and is never displayed as a synchronized sensor clock.
    const capture = performance.now() - (Number.isFinite(result.inferenceTime) ? result.inferenceTime : 0);
    const time = (capture-active.started)/1000;
    if (time < 0 || time <= (active.samples.at(-1)?.time ?? -1)) return;
    active.detector ??= new PitchDetector(active.side,frameAspect);
    active.samples.push({time,points});
    active.detector.add(time,points);
    active.aspect = frameAspect;
    setCount(active.detector.peaks.length);
  }, []);
  const onError = useCallback(error => onStatus(error.message), [onStatus]);
  const detector = usePoseDetection({onResults,onError}, RunningMode.LIVE_STREAM,
    'pose_landmarker_full.task', {delegate:Delegate.CPU, fpsMode:20});
  useEffect(() => {
    if (device) detector.cameraDeviceChangeHandler(device);
    detector.resizeModeChangeHandler('contain');
  }, [device,detector.cameraDeviceChangeHandler,detector.resizeModeChangeHandler]);

  const stop = async () => {
    clearTimeout(timeout.current);
    try { await camera.current?.stopRecording(); }
    catch (error) { onStatus(error.message); }
  };
  const start = () => {
    if (!camera.current || !ready || inProgress) return;
    const active = {started:performance.now(), side, samples:[], detector:null, aspect};
    recording.current = active;
    setInProgress(true);
    onBusy(true);
    setCount(0);
    function finish() {
      clearTimeout(timeout.current);
      recording.current = null;
      setInProgress(false);
      onBusy(false);
    }
    try {
      camera.current.startRecording({fileType:'mp4',videoCodec:'h264',
        onRecordingError:error => { finish(); onStatus(error.message); },
        onRecordingFinished:async clip => {
          finish();
          try {
            const sessionId = id();
            const uri = await keepVideo(sessionId,clip.path);
            const duration = clip.duration;
            const pitches = (active.detector?.finish(duration) ?? []).map(pitch => ({...pitch,id:id()}));
            const session = {id:sessionId, name:new Date().toLocaleTimeString('ko-KR'),
              fingerprint:sessionId, uri, created:Date.now(), duration, aspect:active.aspect,
              side:active.side, samples:active.samples.filter(frame => frame.time <= duration), pitches};
            await saveSession(session);
            onSession(session);
            onStatus(`자동 저장 ${pitches.length}구 · 영상에서 착지와 릴리스를 확인하세요.`);
          } catch (error) { onStatus(error.message); }
        },
      });
      timeout.current = setTimeout(stop,90000);
      onStatus('녹화 중 · 최대 90초');
    } catch (error) { finish(); onStatus(error.message); }
  };

  if (!permission.hasPermission) return <Button title="카메라 권한 허용" onPress={permission.requestPermission}/>;
  if (!device) return <Text style={styles.hint}>후면 카메라를 찾고 있습니다.</Text>;
  return <>
    <View style={styles.stage}>
      <Camera ref={camera} style={StyleSheet.absoluteFill} device={device} format={format}
        fps={format ? Math.max(format.minFps,Math.min(30,format.maxFps)) : undefined} pixelFormat="rgb"
        isActive={foreground} resizeMode="contain" video audio={false}
        frameProcessor={detector.frameProcessor} onLayout={detector.cameraViewLayoutChangeHandler}
        onOutputOrientationChanged={detector.cameraOrientationChangedHandler}
        onInitialized={() => setReady(true)} onError={error => onStatus(error.message)}/>
      <Skeleton layers={[[pose,A]]} aspect={aspect}/>
    </View>
    <View style={styles.row}>
      <Button title={inProgress ? '녹화 끝내기' : '녹화 시작'} active disabled={!ready}
        onPress={inProgress ? stop : start}/>
      <Text style={styles.hint}>{inProgress ? `투구 후보 ${count}구` : '투구 손과 발끝이 보이게 배치하세요.'}</Text>
    </View>
  </>;
}

function ReplayScreen() {
  const [sessions,setSessions] = useState([]), [mode,setMode] = useState('live');
  const [side,setSide] = useState('right'), [status,setStatus] = useState('카메라를 열어 투구를 기록하세요.');
  const [recording,setRecording] = useState(false);
  const [selection,select] = useState(null), [pair,setPair] = useState([null,null]);
  const [checked,setChecked] = useState(false), [phase,setPhase] = useState(.5), [time,setTime] = useState(0);
  const [manualStart,setManualStart] = useState(null);
  const player = useVideoPlayer(null, instance => { instance.timeUpdateEventInterval = .08; });

  useEffect(() => {
    loadSessions().then(setSessions).catch(error => setStatus(error.message));
    const subscription = player.addListener('timeUpdate',event => setTime(event.currentTime));
    return () => subscription.remove();
  }, [player]);
  useEffect(() => {
    if (mode === 'review' && selection?.pitch && time > selection.pitch.end) player.pause();
  }, [mode,selection,time,player]);

  const items = sessions.flatMap(session => session.pitches.map((pitch,index) =>
    ({session,pitch,label:`${session.name} · ${index+1}구`})));
  const find = pitchId => items.find(item => item.pitch.id === pitchId);
  const a = find(pair[0]), b = find(pair[1]);
  const aspect = (mode === 'compare' ? a?.session.aspect : selection?.session.aspect) || 16/9;
  const addSession = useCallback(session => setSessions(rows => [session,...rows]), []);

  async function open(item) {
    player.pause();
    await player.replaceAsync(item.session.uri);
    select(item);
    setManualStart(null);
    player.currentTime = item.pitch?.landing ?? item.pitch?.start ?? 0;
    setTime(player.currentTime);
    setMode('review');
  }

  function move(delta) {
    player.pause();
    const start = selection?.pitch?.start ?? 0;
    const end = selection?.pitch?.end ?? selection?.session.duration ?? 0;
    const next = Math.max(start,Math.min(end,player.currentTime + delta));
    player.currentTime = next;
    setTime(next);
  }

  async function persist(updated, next) {
    await saveSession(updated);
    setSessions(rows => rows.map(row => row.id === updated.id ? updated : row));
    select({session:updated,pitch:next});
  }

  async function mark(event) {
    if (!selection?.pitch) return;
    player.pause();
    const next = markEvent(selection.pitch,event,player.currentTime);
    const updated = {...selection.session,pitches:selection.session.pitches.map(pitch => pitch.id === next.id ? next : pitch)};
    await persist(updated,next);
    setStatus(confirmed(next) ? '착지와 릴리스를 저장했습니다.' : '사건을 저장했습니다.');
  }

  async function addManual() {
    if (!selection || manualStart == null) throw new Error('시작 시점을 먼저 표시하세요.');
    const end = player.currentTime;
    if (end-manualStart < .25) throw new Error('끝은 시작보다 0.25초 이상 뒤여야 합니다.');
    if (selection.session.pitches.some(p => Math.max(p.start,manualStart) < Math.min(p.end,end)))
      throw new Error('기존 투구 구간과 겹칩니다.');
    const pitch = {id:id(),start:manualStart,end,landing:null,release:null,kind:'manual'};
    await persist({...selection.session,pitches:[...selection.session.pitches,pitch]},pitch);
    setManualStart(null);
    setStatus('투구 구간을 저장했습니다.');
  }

  async function compare() {
    const error = comparisonError(a,b,checked);
    if (error) throw new Error(error);
    player.pause();
    await player.replaceAsync(a.session.uri);
    player.currentTime = phaseTime(a.pitch,.5);
    setPhase(.5);
    setMode('compare');
    setStatus('기준 영상 위에 A와 B의 관절선을 겹쳤습니다.');
  }

  function advancePhase(delta) {
    const next = Math.max(0,Math.min(1,phase+delta));
    player.currentTime = phaseTime(a.pitch,next);
    setPhase(next);
  }

  const attempt = action => () => Promise.resolve().then(action).catch(error => setStatus(error.message));
  const live = mode === 'live';
  const layers = mode === 'compare' && a && b
    ? [[nearestPose(a.session.samples,phaseTime(a.pitch,phase)),A],
       [nearestPose(b.session.samples,phaseTime(b.pitch,phase)),B]]
    : [[nearestPose(selection?.session.samples ?? [],time),A]];

  return <SafeAreaView style={styles.screen}>
    <ScrollView contentContainerStyle={styles.content}>
      <View style={styles.row}><Text style={styles.title}>Pitch Replay</Text>
        <Button title="촬영 화면" active={live} onPress={() => { player.pause(); setMode('live'); }}/>
        <Button title={side === 'right' ? '오른손 투구' : '왼손 투구'} disabled={recording}
          onPress={() => setSide(previous => previous === 'right' ? 'left' : 'right')}/>
      </View>
      <Text accessibilityLiveRegion="polite" style={styles.status}>{status}</Text>
      <View style={styles.columns}>
        <View style={styles.viewer}>
          {live ? <LiveCamera side={side} onSession={addSession} onStatus={setStatus} onBusy={setRecording}/> : <>
            <View style={styles.stage}>
              <VideoView player={player} style={StyleSheet.absoluteFill} nativeControls={false}
                contentFit="contain" surfaceType="textureView"/>
              <Skeleton layers={layers} aspect={aspect}/>
            </View>
            {mode === 'compare' ? <View style={styles.row}>
              <Button title="−5%" onPress={() => advancePhase(-.05)}/>
              <Text style={styles.hint}>착지 → 릴리스 {Math.round(phase*100)}% · 민트 A / 주황 B</Text>
              <Button title="+5%" onPress={() => advancePhase(.05)}/>
            </View> : <>
              <Text style={styles.hint}>영상 위치 {formatTime(time)} · 착지 {formatTime(selection?.pitch?.landing)} · 릴리스 {formatTime(selection?.pitch?.release)}</Text>
              <View style={styles.row}>
                <Button title="재생 / 정지" onPress={() => { if (player.playing) player.pause(); else player.play(); }}/>
                <Button title="−0.03초" onPress={() => move(-1/30)}/>
                <Button title="+0.03초" onPress={() => move(1/30)}/>
                <Button title="착지 확인" disabled={!selection?.pitch} onPress={attempt(() => mark('landing'))}/>
                <Button title="릴리스 확인" disabled={!selection?.pitch} onPress={attempt(() => mark('release'))}/>
              </View>
              <View style={styles.row}>
                <Button title="누락 투구 시작" onPress={() => { player.pause(); setManualStart(player.currentTime); }}/>
                <Button title="누락 투구 저장" disabled={manualStart == null} onPress={attempt(addManual)}/>
              </View>
            </>}
            <Text style={styles.hint}>관절 표본의 녹화 시각 대응은 근사입니다. 사건은 원본 영상으로 확인하세요.</Text>
          </>}
        </View>
        <View style={styles.list}>
          <Text style={styles.heading}>투구 기록 · {items.length}구</Text>
          {items.map(item => <View key={item.pitch.id} style={styles.pitch}>
            <Button title={`${item.label} ${confirmed(item.pitch) ? '검수 완료' : '검수 대기'}`} disabled={recording}
              onPress={attempt(() => open(item))}/>
            <View style={styles.row}>
              <Button title="A" active={pair[0] === item.pitch.id} onPress={() => setPair([item.pitch.id,pair[1]])}/>
              <Button title="B" active={pair[1] === item.pitch.id} onPress={() => setPair([pair[0],item.pitch.id])}/>
            </View>
          </View>)}
          {sessions.map(session => <Button key={session.id} title={`${session.name} 원본 검수`} disabled={recording}
            onPress={attempt(() => open({session,pitch:null}))}/>)}
          <Button title={checked ? '✓ 같은 선수와 시야 확인' : '같은 선수와 시야 확인'}
            active={checked} onPress={() => setChecked(previous => !previous)}/>
          <Button title="두 투구 겹치기" disabled={recording} onPress={attempt(compare)}/>
        </View>
      </View>
    </ScrollView>
  </SafeAreaView>;
}

export default function App() {
  return <SafeAreaProvider><ReplayScreen/></SafeAreaProvider>;
}

const styles = StyleSheet.create({
  screen:{flex:1,backgroundColor:'#132427'},
  content:{padding:16,gap:10},
  row:{flexDirection:'row',flexWrap:'wrap',alignItems:'center',gap:8},
  columns:{flexDirection:'row',gap:12,alignItems:'flex-start'},
  viewer:{flex:2,gap:8},
  list:{flex:1,gap:7,backgroundColor:'#1e3335',padding:10,borderRadius:8},
  stage:{aspectRatio:16/9,backgroundColor:'#0b1517',overflow:'hidden',borderRadius:8},
  title:{fontSize:26,fontWeight:'700',color:'#f1f9f6',marginRight:10},
  heading:{fontSize:17,fontWeight:'700',color:'#f1f9f6'},
  status:{fontSize:14,color:'#d1ebe3'},
  hint:{fontSize:13,color:'#c8dbd8'},
  button:{minHeight:42,justifyContent:'center',borderWidth:1,borderColor:'#54736c',
    backgroundColor:'#253d3e',paddingHorizontal:10,paddingVertical:6,borderRadius:6},
  active:{backgroundColor:'#007a64',borderColor:'#5dc8b1'},
  disabled:{opacity:.45},
  buttonText:{fontSize:13,fontWeight:'650',color:'#f1f9f6'},
  pitch:{gap:4,marginBottom:3},
});
