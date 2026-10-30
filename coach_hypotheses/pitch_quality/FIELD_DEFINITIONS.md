# 반복 경기 수집의 필드 정의와 실제 품질

2026-10-30. `CONDITIONAL_PROTOCOL.md`를 커밋한 뒤 2025-06-01~2025-06-14 양끝을 포함한 일별 공식 schedule을 수집했다. 모든 완료 정규시즌 경기를 `officialDate`로 포함하고 `gamePk` 중복만 제거했다. 선수·구속·결과로 표본을 고르지 않았다. 이 문서는 수집과 스키마 검사이며 결과 관계를 계산한 보고서가 아니다.

## 실제 수집

- 일별 schedule 14/14개, 중복 제거 뒤 186/186경기의 공식 JSON을 확보했다. schedule 중복 기록은 2개였다. HTTP·파싱 실패는 0건이다.
- 실제 53,950투구, 14,037 AB와 유효 연결된 투수 463명을 보존했다. 모든 186경기의 두 팀 boxscore `pitchesThrown` 및 `numberOfPitches` 합이 파싱한 투구수와 일치했다.
- 투구 키와 `playId` 중복, 미확인 call, `detailsCode`와 `callCode` 불일치, 누락·범위 오류가 있는 사전 카운트는 모두 0건이었다.
- 첫 투구 뒤 신원 교체가 있는 6개 AB의 36투구는 신원 연결을 `uncertain_quarantined`로 남겼다. 연결용 `pitcherId`, `batterId`, `batSide`, `pitchHand`는 `None`이며 감사용 원 matchup을 별도 보존한다.
- `startSpeed`/`endSpeed`는 각각 2개, `extension`은 1개, `spinRate`/`spinDirection`은 각각 22개가 빠져 있었다. 나머지 수치 필드는 53,950개가 모두 존재하고 유한수였다. 비유한 수치나 잘못된 수치 자료형은 없었다. 결측을 0으로 채우지 않았다.

기존 파서의 엄격한 스키마 기준에서는 10경기에 결측 또는 신원 격리가 있다. 수집 완료를 모든 게임의 기존 data gate 통과로 바꾸지 않는다. 수집기는 모든 파싱 행을 보존하며, 조건부 분석의 적격 행·complete-case 제외는 사전 분석 프로토콜에 따라 별도로 수행해야 한다.

원문·투구별 자료는 전용 worktree의 Git 제외 `artifacts/mlb_quality_games/`에만 저장했다. 행별 파일은 `parsed_pitches.json`이며, 82,322,352 bytes, SHA256 `806206da474392b62f447bf21c79f5ad95abcd2d18ea50743fa3033d32b47229`이다. 단일 개발 경기 777677도 포함된 관측 표본이며 최종 외부 검증 자료로 부르지 않는다.

## 출처와 보존 방식

[공식 schedule 예시](https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2025-06-01)의 실제 feed link만 따라간다. [공식 pitchCodes](https://statsapi.mlb.com/api/v1/pitchCodes)로 원문 bool을 연결한다. 이전 단일 경기 캐시는 원문 bytes·해시·URL·응답 상태를 검사한 뒤 자기 전용 캐시에 복사한다. 다른 worktree의 캐시를 고치거나 덮어쓰지 않는다.

네트워크 동시는 최대 3개다. 네트워크 실패는 최대 2회 추가 시도하며, 영구적인 4xx(429 제외)는 반복하지 않는다. 요청별 URL·UTC 접근 시각·응답 bytes·SHA256·캐시 사용·시도별 성공/오류와 실패를 manifest에 보존한다. raw JSON과 공식 lookup 원문은 재배포하지 않으며 공개 접근을 재배포 라이선스로 간주하지 않는다. 기존 Mendeley/Statcast CSV의 403 실패는 별개의 접근 결과로 남아 있다.

`results/collection_manifest.json`은 schedule·lookup·게임별 실제 source provenance와 행 파일 해시를 담는다. `collection_games.json`은 개별 게임의 전체 스키마 감사, `collection_audit.json`은 결측·격리·카운트·중복·boxscore의 전체 집계를 담는다. 원문이나 투구별 레코드는 이 집계 파일에 넣지 않았다.

## 원필드 정의와 불확실성

원 경로는 `mlb_feed.NUMERIC_PATHS` 및 개별 게임 감사에 기록했다. `pitchData.startSpeed/endSpeed/extension/plateTime`, `pitchData.breaks`의 회전·움직임 값, `pitchData.coordinates`의 pfx·plate·9개 운동학 값을 원필드명과 원값으로 보존한다.

[MLB 공개 gamefeed 배포 JS](https://builds.mlbstatic.com/baseballsavant.mlb.com/v1/sections/gamefeed/builds/4a2930fb3794eed37b9aabce1f4ff3d0966f246b/scripts/build/gamefeed-all.js)는 `startSpeed`를 소수점 한 자리와 `mph`로 표시한다. 2026-10-30 직접 GET도 HTTP 200이었고 2,710,604 bytes, SHA256 `91fdc3a0d6b726c26363330faf1c3c812d96ee48cfbfbf15ceaae4bac5942e14`를 확인했다. 따라서 **startSpeed의 표시 단위 mph만 별도 근거로 기록**한다. 정확한 평가 위치·시점까지 확인했다는 뜻은 아니다. 기존 파서의 field metadata와 과거 gate는 그대로 두었다.

`spinRate`, `spinDirection`, `breakVerticalInduced`, `breakHorizontal`, `pfxX/pfxZ`, `pX/pZ`, 위치·속도·가속도의 StatsAPI 경로별 단위·좌표 방향·시간 기준은 `UNKNOWN`이다. [공식 metrics](https://statsapi.mlb.com/api/v1/metrics)의 유사한 이름만으로 별칭을 확정하지 않는다. 값 범위로 단위를 정하거나 Savant CSV와 치환하지 않는다. `x0/y0/z0`를 손 릴리스 좌표로 이름 바꾸지 않으며, `spinDirection`을 3D 회전축·회전 효율로 바꾸지 않는다.

각 행의 `gameDate`는 실제 schedule의 `officialDate`다. feed의 `gameData.datetime.officialDate`가 제공되면 서로 같은지 검사한다. 사전 balls/strikes는 AB 시작 0/0에서 비투구 이벤트의 사후 count도 반영해 복원한다. 공식 `callCode`, `detailsCode`, `official_pitch_code`는 그대로 남긴다. T의 공식 `swingMissStatus=true`를 바꾸지 않으며, 수집기는 연구용 결과 변수를 계산하지 않는다.

팔·손·그립·심 방향·의도 목표·전신 운동학을 같은 투구에 연결하지 않았다. 미관측 상태와 기존 `body_to_spin_ready=false`를 유지한다. 물리 모델, 회전 효율, 관절 처방, 인과 효과나 실제 구위 향상을 이 수집으로 주장하지 않는다.

## 재현

```powershell
..\.venv\Scripts\python.exe -B -m unittest coach_hypotheses.pitch_quality.test_collect_games -v
..\.venv\Scripts\python.exe -B -m coach_hypotheses.pitch_quality.collect_games
```

수집기 타깃 테스트 10개는 날짜·실제 link 선택·중복/충돌, 캐시 검증·손상 보존, 재시도 상한·영구 실패, 최대 3 worker, 날짜 추가, 파싱 실패, 집계·결측·신원 격리, 부분 schedule 및 lookup 실패를 확인한다. synthetic fixture는 연구 표본 수에 넣지 않는다. 추가로 실제 186경기 자료를 캐시에서 다시 파싱하고 같은 53,950행 해시를 확인했다. 행 파일은 완성한 임시 파일을 교체해 내보내므로 읽는 작업에 부분 JSON을 노출하지 않는다.
