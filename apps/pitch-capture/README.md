# Pitch Replay

한 대의 카메라로 투구를 촬영하고, 사람이 확인한 착지와 릴리스로 두 투구의 2D 관절선을 겹쳐 보는 실험 앱이다. 웹은 브라우저의 IndexedDB에, iOS·Android 앱은 앱 내부 저장소에 영상을 기록한다. 투구마다 시작·끝 시각을 저장하며 같은 세션의 원본 동영상을 공유한다.

## 시작

이 폴더에서 Node.js 24와 npm으로 실행한다.

```powershell
npm ci
npm run assets
npm run web
```

브라우저에서 `http://127.0.0.1:4317/`을 연다. 자산 준비는 기존 모델의 SHA-256을 검사하고, 파일이 없으면 Google의 고정 버전 모델을 받아 웹과 앱에 복사한다. 웹 실행에는 해당 모델 파일을 포함한 빌드가 필요하다. 다른 기기에서 웹 카메라를 쓰려면 그 기기에서 신뢰하는 HTTPS 주소로 빌드를 제공해야 한다. 같은 PC의 localhost는 로컬 카메라 테스트에 쓸 수 있다.

1. 카메라를 켜고 투구 손을 선택한다. 스마트폰 고정 촬영은 `전체 화면`, 투수가 작게 나오는 고정 구도의 중계 영상은 `화면 중앙`부터 시험한다.
2. 녹화를 시작한 뒤 최대 90초 안에 끝낸다. 웹에서 녹화 또는 파일을 받으면 영상 시각을 기준으로 관절을 다시 추적하고, 빠른 팔 동작이 반복된 구간을 투구 후보로 저장한다. 누락된 구간은 `누락된 투구 추가`에서 직접 지정한다.
3. 각 투구의 영상을 재생하거나 탐색해 `착지 확인`과 `릴리스 확인`을 누른다. 시간 표시는 영상 재생 시각이다. 카메라의 실제 셔터 시각과는 별도로 다룬다.
4. A와 B를 선택하고 같은 선수와 고정된 시야인지 확인한다. A 영상 위에서 민트색 A와 주황색 B를 본다. 진행률은 각 투구에서 착지를 0%, 릴리스를 100%로 맞춘다. 관측이 없는 프레임에는 관절선을 이어 그리지 않는다.

## iOS·Android

`mobile/`은 Expo 개발 빌드를 쓰는 React Native 앱이다. VisionCamera의 네이티브 카메라와 프레임 처리에서 MediaPipe를 실행하고, 투구 감지와 정렬 로직은 웹과 공유한다. 후면 카메라의 가로 촬영을 기준으로 한다. iOS·Android 설치 파일은 다음 방식으로 해당 플랫폼의 도구를 사용해 만든다.

```powershell
cd mobile
npx expo prebuild --platform android
npx expo run:android --device
```

macOS에서는 Xcode를 준비하고 `npx expo prebuild --platform ios`, `npx expo run:ios --device`를 실행한다. 최초 빌드는 위의 `npm ci`와 `npm run assets`를 먼저 실행한다. Android에는 SDK와 기기 연결이 필요하다. 카메라 프레임 처리는 커스텀 네이티브 모듈을 사용하므로 개발 빌드로 시험한다. 모바일의 추적 표본 시각은 녹화 시작과 추론 시간을 바탕으로 한 근사값이다. 착지와 릴리스는 원본 재생으로 확인한다. [Expo 개발 빌드](https://docs.expo.dev/get-started/set-up-your-environment/) · [VisionCamera](https://visioncamera4.margelo.com/docs/guides/recording-videos)

## 이번 실행에서 확인한 범위

- 실제 중계 영상의 서로 다른 폴 스킨스 투구 두 개를 가져와 각각 한 구를 자동 저장했다. 착지와 릴리스를 화면에서 표시하고, 21개 진행률에서 A와 B의 픽셀이 모두 그려지는지 확인했다. 새로고침 뒤에도 영상과 표식이 IndexedDB에 남았다. 시연 영상은 실제 브라우저의 겹쳐보기 화면이며 관중석에만 흐림을 넣었다. [첫 번째 영상](https://www.mlb.com/video/paul-skenes-grabs-his-seventh-strikeout) · [두 번째 영상](https://www.mlb.com/video/yasmani-grandal-nabs-christopher-morel-stealing)
- 같은 실영상을 Chromium의 파일 기반 카메라 입력으로 재생했다. 카메라 권한, 실시간 관절선, MediaRecorder 녹화, 분석 후 두 구 자동 저장이 동작했다. 물리 스마트폰과 카메라 지연 측정은 별도 검증이 필요하다.
- 웹 빌드와 공유 로직 테스트 3개가 통과했다. Expo에서 Android 네이티브 프로젝트를 생성했고 iOS·Android용 Metro 번들을 내보냈다. Android SDK와 Xcode가 없는 Windows 환경에서 수행한 검증이다. 양쪽 설치 파일과 실기기 검증은 아직 남아 있다.

관절선은 영상에서 추정한다. 두 영상의 비교에는 같은 카메라 각도와 사람이 확인한 사건이 필요하다. 자동 후보는 이 두 개발 영상에서 확인한 결과다. 구속·회전수·힘은 별도의 장비나 검증된 모델로 연결할 수 있는 항목이다.

화면 설계에는 Damir Sokolovsky의 [Unslop](https://heydamir.design/unslop) 가독성 지침을 참고했다.
