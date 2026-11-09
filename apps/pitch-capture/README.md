# Pitch Replay

한 대의 카메라로 투구를 촬영하고, 사람이 확인한 착지와 릴리스로 두 투구의 2D 관절선을 겹쳐 보는 실험 앱이다. 웹은 브라우저의 IndexedDB에, iOS·Android 앱은 앱 내부 저장소에 영상을 기록한다. 투구마다 시작·끝 시각을 저장하며 같은 세션의 원본 동영상을 공유한다.

## 시작

이 폴더에서 Node.js 24와 npm으로 실행한다.

```powershell
npm ci
npm run assets
npm run web
```

브라우저에서 `http://127.0.0.1:4317/`을 연다. 자산 준비는 기존 모델의 SHA-256을 검사하고, 파일이 없으면 Google의 고정 버전 모델을 웹용으로 받는다. 다른 기기에서 웹 카메라를 쓰려면 그 기기에서 신뢰하는 HTTPS 주소로 빌드를 제공해야 한다. 같은 PC의 localhost는 로컬 카메라 테스트에 쓸 수 있다.

1. 카메라를 켜고 투구 손을 선택한다. 스마트폰 고정 촬영은 `전체 화면`, 투수가 작게 나오는 고정 구도의 중계 영상은 `화면 중앙`부터 시험한다.
2. 녹화를 시작한 뒤 최대 90초 안에 끝낸다. 웹에서 녹화 또는 파일을 받으면 영상 시각을 기준으로 관절을 다시 추적하고, 빠른 팔 동작이 반복된 구간을 투구 후보로 저장한다. 누락된 구간은 `누락된 투구 추가`에서 직접 지정한다.
3. 각 투구의 영상을 재생하거나 탐색해 `착지 확인`과 `릴리스 확인`을 누른다. 시간 표시는 영상 재생 시각이다. 카메라의 실제 셔터 시각과는 별도로 다룬다.
4. A와 B를 선택하고 같은 선수와 고정된 시야인지 확인한다. A 영상 위에서 민트색 A와 주황색 B를 본다. 진행률은 각 투구에서 착지를 0%, 릴리스를 100%로 맞춘다. 관측이 없는 프레임에는 관절선을 이어 그리지 않는다.

## iPhone과 Android

iPhone은 SwiftUI, [AVFoundation](https://developer.apple.com/documentation/avfoundation/capture-setup)과 [Vision](https://developer.apple.com/documentation/vision/detecting-human-body-poses-in-images)을 사용한다. Android는 Kotlin, [CameraX](https://developer.android.com/media/camera/camerax/video-capture)와 [ML Kit Pose Detection](https://developers.google.com/ml-kit/vision/pose-detection/android)을 사용한다. 각 앱은 독립된 프로젝트다. 웹의 Node.js, Expo, 웹용 모델 파일 없이 빌드한다.

- **iPhone(iOS 16 이상):** Mac에서 `ios/PitchReplay.xcodeproj`를 연다. `PitchReplay` 대상의 Signing & Capabilities에서 팀을 고르고 연결한 iPhone을 선택해 Run을 누른다. 카메라는 실제 기기에서 확인한다. 시뮬레이터에서는 기존 영상 가져오기를 시험할 수 있다.
- **Android(Android 8 이상):** Android Studio에서 `android/`을 연다. Android SDK 35와 JDK 17 이상을 준비하고 Gradle 동기화를 마친 뒤, 연결한 기기에서 `app`을 실행한다. IDE의 SDK 설정을 사용하므로 `local.properties`는 커밋하지 않는다.

후면 카메라로 가로 영상을 촬영하거나 길이 90초 이하의 기존 영상을 가져온다. Android의 가져오기 파일은 512MB까지다. 라이브 관절선으로 구도를 확인한다. 녹화가 끝나면 저장 영상의 재생 시각으로 다시 분석하고, 투구 후보 구간을 원본과 연결해 기기 내부에 저장한다. 놓친 투구는 복기 화면에서 추가할 수 있다. 두 투구의 착지와 릴리스를 직접 표시한 뒤 같은 선수·촬영 시야인지 확인하면 기준 원본 위에 A와 B를 겹쳐 볼 수 있다.

관절 검출기는 플랫폼마다 다르므로 같은 영상을 넣어도 검출 점과 자동 후보가 달라질 수 있다. 보이지 않는 관절은 비워 둔다. 라이브 미리보기와 저장 영상의 추적 표본은 서로 다른 처리 경로에서 얻으며, 실시간 지연이나 고속 촬영 정밀도는 기기에서 확인해야 한다.

## 라이브 공 추적

카메라를 측면에 고정한다. 라이브 화면에서 공이 진행할 구간의 시작점과 끝점을 차례로 지정하면 관절선과 공 궤적이 같은 영상 위에 표시된다. 연속 영상에서 밝기가 변한 작은 후보를 찾고, 같은 방향으로 이어지는 중심이 세 개 이상 모이면 궤적을 표시한다. 후보가 끊겼을 때에는 실제로 확인된 점만 남긴다. 첫 점을 릴리스로 자동 확정하지 않는다.

- **평면 구속(km/h)**은 초기 공 중심과 실제 프레임 시각에 직선을 맞춰 계산한다. 공이 날아가는 평면에 있는 실제 거리 표식 두 점을 고르고 그 사이 길이를 입력해야 한다. 카메라가 기울었거나 공이 그 평면에서 크게 벗어나면 한 길이로 전체 영상의 축척을 대신할 수 없다.
- **평면 휨**은 관측한 첫 점과 마지막 점을 잇는 직선에서 공 경로가 가장 멀리 떨어진 정도다. 기준 거리가 없으면 픽셀로 표시한다. 관측 구간이 짧거나 카메라 위치가 다르면 값이 달라진다. MLB는 별도의 추적 자료로 수평·수직 무브먼트를 정의한다. [Statcast CSV 항목](https://baseballsavant.mlb.com/csv-docs)
- **표식 회전수(rpm)**는 파란 점 하나를 붙인 훈련구의 중심과 표식 사이 각도가 고속 프레임에서 어떻게 바뀌는지 계산한다. 회전축이 카메라를 향하고 표식이 계속 보이는 조건에서만 사용한다. 실제 표식 관측 사이의 간격이 약 5ms 이내이고, 회전 한 바퀴 이상에서 각도와 반지름이 일관될 때 숫자를 표시한다. 웹과 iPhone에 표식 모드가 있다. iPhone에서는 카메라를 열기 전에 모드를 켠다. Android의 일반 CameraX 촬영 경로에서는 회전수 측정 불가를 표시한다. [고속 표식 촬영 연구](https://pubmed.ncbi.nlm.nih.gov/16939153/)

iPhone의 Swift 순수 로직은 Mac에서도 Xcode 프로젝트와 별도로 검산할 수 있다.

```sh
swiftc ios/PitchReplay/ReplayCore.swift ios/Tests/ReplayCoreChecks.swift -o /tmp/pitch-replay-check
/tmp/pitch-replay-check
swiftc ios/PitchReplay/ReplayCore.swift ios/PitchReplay/LiveBall.swift ios/Tests/LiveBallChecks.swift -o /tmp/pitch-live-check
/tmp/pitch-live-check
```

Android SDK를 설정한 Windows에서는 `android/` 폴더에서 `./gradlew.bat :app:testDebugUnitTest :app:assembleDebug`를 실행한다.

기기에서 확인할 순서: 기존 영상 두 개를 가져와 각각 투구가 저장되는지 확인한다. 자동 후보가 없다면 구간을 직접 추가한다. 착지·릴리스에 표시하고 A/B를 맞춘 뒤 앱을 다시 실행해 기록이 남아 있는지 확인한다. 같은 절차를 앱의 후면 카메라 녹화로 한 번 더 진행한다. 정렬 화면에서 팔이 가려진 프레임에 선이 새로 생기지 않는지와 영상 회전 방향이 맞는지도 확인한다.

## 실제 영상으로 확인한 범위

- 실제 중계 영상의 서로 다른 폴 스킨스 투구 두 개를 가져와 각각 한 구를 자동 저장했다. 착지와 릴리스를 화면에서 표시하고, 21개 진행률에서 A와 B의 픽셀이 모두 그려지는지 확인했다. 새로고침 뒤에도 영상과 표식이 IndexedDB에 남았다. 시연 영상은 실제 브라우저의 겹쳐보기 화면이며 관중석에만 흐림을 넣었다. [첫 번째 영상](https://www.mlb.com/video/paul-skenes-grabs-his-seventh-strikeout) · [두 번째 영상](https://www.mlb.com/video/yasmani-grandal-nabs-christopher-morel-stealing)
- 초기 웹 검증에서 같은 실영상을 Chromium의 파일 기반 카메라 입력으로 재생했다. 카메라 권한, 실시간 관절선, MediaRecorder 녹화, 분석 후 두 구 자동 저장을 확인했다.
- Prime `OBP_movement_BaseballThrow_001`의 19번 영상에서 브라우저 공 추적기를 원본 360Hz로 실행했다. 기존 밝은 코어 좌표가 있는 8프레임이 모두 추적에 포함됐고, 원본 영상 좌표의 평균 거리는 3.56px이었다. 기존 좌표도 밝기 기반 판독이므로 이 수치는 실제 중심의 오차 한계로 사용할 수 없다. 같은 입력을 60fps로 줄여 Chromium 카메라에 공급한 실행에서 관절선과 공 관측이 한 화면에 표시됐다.
- 웹 테스트의 구속·회전수 수치 검산에는 속도, 거리, 회전율을 아는 합성 프레임을 사용했다. 실제 구속·RPM과 비교할 측면 촬영 및 회전 표식은 이 실영상 시연에 없다.
- 웹 빌드와 테스트 6개, Android 디버그 APK 빌드와 Gradle 단위 테스트 9개를 확인했다. iOS는 Swift 문법 및 Xcode 프로젝트 연결을 검사했다. 스마트폰에서 카메라, 렌즈 보정, 기기별 처리 속도와 표식 회전수는 별도로 확인해야 한다.

관절선은 영상에서 추정한다. 두 영상의 비교에는 같은 카메라 각도와 사람이 확인한 사건이 필요하다. 자동 후보는 이 두 개발 영상에서 확인한 결과다. 구속·회전수·힘은 별도의 장비나 검증된 모델로 연결할 수 있는 항목이다.

화면 설계에는 Damir Sokolovsky의 [Unslop](https://heydamir.design/unslop) 가독성 지침을 참고했다.
