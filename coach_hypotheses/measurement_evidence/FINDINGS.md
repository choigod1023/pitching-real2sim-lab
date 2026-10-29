# 공개 손·공 동시 관측 자료: 원본 확보와 사용 범위

2026-10-29. **같은 던지기의 손과 공 표식을 담은 새 공개 C3D를 실제로 확보했다.** Ozkaya 등(2018)의 자료다. 손·공 상대 표식과 이탈 대리 이벤트의 보조 검증에는 쓸 수 있지만, 공 중심·손가락 접촉·독립 구속 정의가 해결되지 않아 3034_1~5의 물리 모델을 바로 보정하는 정답 자료로 쓸 수는 없다. Stage2의 가정 분포, 기존 프로토콜·결과·과학 gate와 PAUSED 상태를 바꾸지 않았다.

## 새로 내려받아 읽은 자료

[Scientific Data 원문](https://www.nature.com/articles/sdata2018272)은 야구공으로 약 4m 앞 표적을 향해 던지는 초보 여성 7명의 반복 과제를 설명한다. 엘리트 야구 투구와 다른 모집단·과제다. [Figshare collection](https://doi.org/10.6084/m9.figshare.c.4017808.v1)의 API는 15개 article(C3D 7개, CSV 7개, Trial Information)을 반환했다. 이번에는 Subject1의 두 archive와 전체 참가자의 구속·상태표만 받았다. 전 참가자의 3,150개 투구를 검사한 것은 아니다.

- [Subject1 C3D API](https://api.figshare.com/v2/articles/5943103): S1.rar, 372,116,458 bytes, MD5 `7f4357a206fa87d2506e86fd2dbb0c32`.
- [Subject1 CSV API](https://api.figshare.com/v2/articles/6743120): S1_csv.rar, 275,030,892 bytes, MD5 `022f99deca947f51e6afdbb0a56651cb`.
- [Trial Information API](https://api.figshare.com/v2/articles/5943289): Trialinfo.xlsx, 55,186 bytes, MD5 `9b3b3a8fceed15baf56f65d004098085`.

세 파일 모두 API의 크기·MD5와 실제 다운로드가 일치했다. 각 데이터 article의 라이선스는 **CC0**다. 논문의 CC BY 4.0과 구분했다. SHA256·URL·절대경로는 [download_manifest.json](download_manifest.json)에 있다. 원본은 Git에서 제외한 `artifacts/`에 보존했다. 원시 카메라 영상이 아니라 저자가 라벨링·일부 gap filling한 연구 배포 C3D/CSV다.

동적 표본 세 개는 탐색 중 먼저 읽은 S1/1w2d/DP05_O, DP06_O, S1/1w1d/NDP07_O다. 표본을 이용해 실제 계측 오차분포나 참가자 집단의 분포를 추정하지 않았다. [검사 범위](INSPECTION_PROTOCOL.md)와 [실행 집계](sample_audit.json)를 함께 보존한다.

## 실제 헤더·좌표·프레임 확인

세 C3D 모두 200Hz, POINT 단위 mm다. `Ball`과 `R3MC/R5MC/L3MC/L5MC`가 실제 POINT 채널로 있다. 원문의 [marker table](https://www.nature.com/articles/sdata2018272/tables/3)은 3MC/5MC를 각 손의 제3/제5 중수골두 부착 표식으로, Ball을 Baseball로 정의한다. 최고속도나 궤적 모양으로 공을 추정한 것이 아니다. 골중심을 직접 측정한 채널이나 손가락 관절·접촉점 채널로 부르지 않는다.

유효 좌표는 xyz가 모두 유한하고 C3D residual이 비음수인 경우로 정의했다. residual 0을 0mm 정확도라는 뜻으로 읽지 않는다.

- DP05: 647프레임 중 Ball 26개(4.019%), 유효 배열 index 238~263. 제공 이탈 대리 이벤트는 1.29499996s, raw frame259, 배열 index258이다.
- DP06: 931프레임 중 Ball 51개(5.478%), index267~317. 이벤트 1.54499996s, raw frame309, 배열 index308이다.
- NDP07: 1,031프레임 중 Ball 143개(13.870%), index170~312. 이벤트 1.55499995s, raw frame311, 배열 index310이다.

세 투구의 던지는 손 3MC/5MC는 해당 파일 전체에서 유효했다. Ball은 전체 비행을 연속 관측하지 않으며 이탈 주변의 짧은 구간이 주된 사용 범위다. 원문은 이동 시작~끝 사이에 spline/pattern gap filling을 했다고 설명하지만, 배포 파일에서 각 좌표가 직접 광학 관측인지 보간인지 나눌 표본별 표시는 확인하지 못했다.

원문은 이탈을 손–공 표식 거리가 10% 증가하는 순간으로 정의하고 General 이름을 설명한다. 실제 파일에는 LABEL=`Event`, CONTEXT=`General`, GENERIC_FLAG=1인 이벤트가 하나씩 있다. 명칭 차이를 기록하고 원문에 따라 이 이벤트를 이탈 **대리 이벤트**로 연결했다. 어떤 손 표식과 어떤 기준거리에서 10%를 계산했는지까지는 공개 설명으로 재현하지 못했다. 독립적인 마지막 접촉 영상 판독은 없다.

DP05의 원시 C3D 헤더와 TRIAL:ACTUAL_START_FIELD는 첫 프레임1을 기록하지만 ezc3d 헤더는0을 반환한다. `raw_frame=round(event_seconds*200)`, `array_index=raw_frame-raw_header_first`로 대응했다. 실제 CSV ITEM259의 Ball은 C3D 배열258에 대응한다. ezc3d의0만 보고 event index259를 쓰면 한 표본 밀린다. CSV는 확장자와 달리 탭 구분이며 좌표값은 C3D(mm)/1000과 일치한다. 세 표본의 이벤트 ±5프레임 대조에서 최대 차이는 5.09e-7m로, CSV의 소수6자리 반올림과 양립한다. CSV에 좌표 단위 헤더는 없으므로 숫자·C3D와의 대응을 근거로 기록했다.

## 공 중심과 독립 구속은 아직 해결되지 않았다

2018 원문이 이 자료 기반의 선행 연구로 연결한 [2017 KJSB 공식 본문](https://www.e-kjab.org/archive/detail/56)의 Methods2는 “A reflective tape was also affixed on the ball to define the ball release.”라고 명시한다. 따라서 Ball을 공 표면 반사테이프에서 얻은 표식으로 취급할 근거가 있다. 공 중심 변환, 반지름, 테이프 위치와 공 회전 자세는 배포 메타데이터에서 확인하지 못했다. 단일 공 표면점의 미분에는 공의 병진 외에 회전 항도 들어갈 수 있으므로 Ball 속력을 공 중심 구속으로 대입하지 않는다.

2017 본문은 speedgun을 참가자 뒤3m에 놓았다고 설명하지만 기종·단위·측정 구간은 명시하지 않는다. 또한 2017은3세션/모두 오른손잡이, 2018은15세션/S2 왼손 우세로 기술하는 차이가 있어 모든 실험 설정을 배포 전 투구에 복사하지 않았다.

Trialinfo의 실제 같은 투구 대응은 다음과 같다. S1의1w2d DP05는 D5:F5의 DP_05/O/44, DP06은 D6:F6의 DP_06/O/42다. S1의1w1d NDP07은 A26:C26의 NDP_07/M-XIPH/41이다. 단위가 없는 Ball Speed 원값을 그대로 보존했다. 44/42/41을 임의로 km/h 또는 mph로 정하지 않았다. **NDP07 파일명은 `_O`지만 상태표는 `M-XIPH`로 다르다.** 파일명 O만으로 모든 마커가 정상이라고 해석하면 안 된다.

제공 이벤트 다음 +1~+5 표본을 미리 고정해 유효할 때만 선형 위치 적합 기울기를 계산했다. DP05/06의 Ball 표식 속력은 각각11.772339/10.884678m/s였다. NDP07은 +3 이후 결측이어서 계산하지 않았다. 이 계산은 짧은 이탈 대리 이벤트 이후에 좌표가 존재하고 미분 가능한지를 보는 검사다. 물리적 무접촉을 확인한 구간도, 공 중심 속력도 아니며 레이더와 MAE를 계산하지 않았다. 필터·이벤트·배율을 구속에 맞춰 고르지 않았다.

## 정적 파일에서 가능한 보조 기하

같은1w2d의 static_01_D(253프레임), static_02_N(263프레임), 같은1w1d의 실제 이름 static_03_N(194프레임)을 추출했다. 요청한1w1d/static_02_N은 archive에 없어 정해진 이름을 추정하지 않고 멤버 목록에서03_N을 확인했다. 세 파일에서 양쪽 wrist styloid(WRL/WRM)와 forearm cluster(FR1~4)가 모두 전프레임 유효했다. 네 cluster 점은 비공선이어서 정적 표식을 이용한 강체 정렬·가상 손목 추정의 입력 요건을 충족한다.

다만 동적 손에는 3MC/5MC 두 점뿐이다. 두 점만으로 손의 roll을 독립적으로 정할 수 없다. 정적 전완 cluster로 옮긴 styloid 중점은 피부 표식과 강체 가정에 따른 **추정 기준점**이다. 실제 골 손목중심으로 검증한 값이 아니다. 같은 세션 정적 파일을 쓰고 정적 보류 프레임·동적 cluster 잔차를 별도로 검사한 뒤에만 보조 손 프레임으로 검토할 수 있다. 이번 감사에서는 손 프레임이나 그립 모델을 새로 적합하지 않았다.

## 추가 관측의 측정 정의

이미 감사한 [Kusafuka 등 PLOS2023](https://doi.org/10.1371/journal.pone.0290042)은 추가 데이터 발견으로 세지 않았다. 측정 정의를 확인하는 데 사용했다. 저자들은 두960fps 시야에서 영상상 손가락 관절·손톱 중심과 공 중심을 추적하고 LED로 동기화했다. 27개 알려진3D점으로 DLT를 구성했으며, 측면 영상의 이탈 판독을 중지 손톱–공 거리로 확인했다. 이는 뼈 영상으로 관절중심을 직접 측정한 방법과 다르다. 보충자료에서 재실행 가능한 전체 투구 좌표를 확보했다는 종전 결론도 바꾸지 않았다.

새 손–공 관측은 먼저 보이는 손 표면점/손톱점/부착 마커의 정의를 고정하고, 공은 보이는 부분의 밝기 중심·윤곽 중심·실제 구 중심을 구분해야 한다. 가려진 공 일부의 색 중심을 그대로 공 중심으로 넣으면 오프셋과 그 변화량이 섞인다. 카메라별2D관측, 왜곡·스케일을 포함한 보정, 물리적 노출 동기, 동일 투구ID, 마지막 접촉/첫 분리의 판독 구간을 함께 남겨야 한다. 이 문단은 위 측정법과 기존 Prime 감사에서 도출한 후속 측정 요건이며 새 계측 결과는 아니다.

## Prime 및 현재 모델에 대한 판정

기존 [Prime 카메라 감사](C:/Users/user/Documents/Codex/2026-10-20/new-chat-2/outputs/pitching-lab/prime-camera-geometry-audit-20261029/PRIME_CAMERA_FINDINGS.md), [손 정렬 감사](C:/Users/user/Documents/Codex/2026-10-20/new-chat-2/outputs/pitching-lab/prime17-hand-alignment-20261029/PRIME17_HAND_FINDINGS.md), [손 프레임 정의](C:/Users/user/Documents/Codex/2026-10-20/new-chat-2/outputs/pitching-lab/hand-ball-link/HAND_FRAME_SOURCES.md), [손–공 관측](C:/Users/user/Documents/Codex/2026-10-20/new-chat-2/outputs/pitching-lab/hand-ball-link/HAND_BALL_OBSERVATION.md)을 읽었다. 그 결과를 다시 산출하지 않았다. Prime은 같은 영상의 공 후보와 제조사 추정 손목 원점·분절 회전을 연결할 수 있고 상대 위치·벡터 차분의 조건부 진단이 가능하다. 그러나 실제 손가락점이 없고 카메라 기하·동기·이탈 검증이 남는다. 새2018자료는 별도 사람·과제이므로 Prime 및3034_1~5의 정답으로 혼합하지 않는다.

즉시 가능한 후속 작업은 새 C3D의 **손·공 표식에 한정한 보류 구간 예측 검사**다. 동일 세션 정적 정보에서 가상 기준점을 구성할 경우 그 추정 과정을 검증하고, 보정에 쓰지 않은 표식 위치·속도를 예측하는지 볼 수 있다. 이는 손가락 힘·공 중심 역학·OBP 구속을 검증하기 전의 보조 단계다. 새 공개 자료의 존재는 확인했지만, Stage2 거리10~20cm나 상대속도±0.5m/s를 실제 오차확률로 바꿀 근거는 확보하지 못했다.

기존 가중구 Figshare14072936, PLOS2023 부록, Scientific Reports2025 요청자료, npjAI2026 비공개 자료와 SIGGRAPH2004 소개영상은 새 접근 근거가 없어 주요 성과로 다시 세지 않았다. 조사한 범위 밖에도 적합한 공개 자료가 없다는 완전성 주장은 하지 않는다.

## 접근 실패·재현·검증

웹 도구는 Nature table 클릭과 Figshare DOI redirect에서 접근에 실패했다. 공개 Figshare REST API/공식 ndownloader와 publisher table 직접 HTTP 읽기는 성공했다. 로그인·접근제한을 우회하지 않았고 외부 연락·유료 구매·원본 업로드는 하지 않았다. API응답·페이지·archive멤버 목록·원본은 ignored artifacts에 있다.

환경: `outputs/pitching-lab/.venv/Scripts/python.exe`, NumPy와 ezc3d. 스프레드시트는 원본을 수정하지 않고 번들 Python의 표준 zip/XML 파서로 셀을 읽었다. 새5개 단위검사(프레임 대응/격자 밖 이벤트 거부/결측 처리/고정 속도·병진 불변/결측 구간 미연결)가 통과했다. 이는 관측 정확도 인증이 아니다. 변경하지 않은 기존 전체 테스트를 다시 돌리지 않았다.

재실행:

```powershell
../.venv/Scripts/python.exe -m unittest coach_hypotheses.measurement_evidence.test_audit_public_sample -v
../.venv/Scripts/python.exe coach_hypotheses/measurement_evidence/audit_public_sample.py
```

자동 다운로드는 하지 않는다. `download_manifest.json`의 원본을 먼저 받고, 지정한 C3D/CSV와 세 정적 파일을 `artifacts/S1/세션/`에 추출해야 한다. [source_inventory.json](source_inventory.json)은 관측/공개 원본/동기/보정/레이더 대응/이탈검증을 TRUE/FALSE/UNKNOWN으로 분리한다. 공개 파일 확보와 실제 성능 효과 검증은 별개로 유지한다.
