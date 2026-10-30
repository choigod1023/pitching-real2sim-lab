# 시간순 구위 실험의 자료 수집·분할 검증

2026-10-30. 사전등록 `59ae41da8c5f605689a87f4e81a017240539ba38`의 `PROTOCOL.md`에 따라 2025-06-01~2025-08-31 공식 MLB JSON을 확보했다. **수집·분할 검증 결과이며, 모델의 예측 개선·구위 효과를 계산한 결과가 아니다.** 과거 PR128 자료와 판정은 변경하지 않았다.

## 실제 표본과 수집 완전성

- 양끝을 포함한 92일의 일별 schedule을 모두 확보했다. `officialDate`가 범위 안인 R/Final 경기를 `gamePk`로 중복 제거했고 실제 schedule의 feed link만 사용했다.
- 선택·파싱한 경기는 각각 1,187개다. 실제 346,191투구와 유효 신원으로 연결된 투수 726명을 보존했다. 수집·파싱 실패는 0건이다.
- 개발 기간 6/1~7/31은 61일·766경기·222,252투구다. 이 중 기존 6/1~14의 186경기는 원문 bytes·SHA256·URL·응답 상태를 검증한 캐시로 재사용했고 신규 경기 GET는 580개였다.
- 최종 검사 기간 8/1~31은 31일·421경기·123,939투구다. 개발 두 파일을 먼저 완성·해시 검증해 부모 작업에 전달한 다음 8월을 수집했다. 원자료 수집·가용성 감사와 모델의 최종 검사 실행을 구분한다.
- 모든 1,187경기에서 두 팀 boxscore `pitchesThrown`과 `numberOfPitches`의 합이 파싱 투구수와 일치한다. 전체 투구 키·`playId` 중복, 미확인 call, call/details 코드 불일치, 카운트 오류와 파싱 issue는 모두 0건이다.

신원 교체로 불확실한 161투구는 연결 ID를 `None`으로 두고 원 matchup을 보존했다. 구종이 확인되지 않은 170투구도 남겼다. 수치 결측은 startSpeed/endSpeed 각각 173개, extension 199개, spinRate/spinDirection 각각 1,447개다. plateTime·움직임·위치·속도벡터·가속도 등 나머지 수치 원필드는 각각 170개가 빠져 있다. 잘못된 수치 자료형·비유한 값은 없었다. 결측이나 신원 불확실성을 0으로 채우지 않았다.

`collection_complete=true`는 92일·1,187경기의 원자료 확보와 파싱이 끝났다는 뜻이다. 원필드 결측·신원 격리 및 기존 엄격한 스키마 게이트와는 구분한다. 적격 FF·비번트 스윙·complete-case 제외는 부모 모델이 사전 기준에 따라 별도로 수행한다.

## 세 파일과 원문 연결

모든 투구 행의 원래 키와 값을 유지하고 `(gamePk, atBatIndex, pitchIndex)` 순서를 각 파일 안에서 유지했다. 분할에는 schedule에서 온 `gameDate`만 사용하며 결과 값·성적은 사용하지 않았다.

- `dev_train.json`: 6/1~7/15, 167,532행·577경기, 255,604,472 bytes. SHA256 `a0281af982076904b8d518b0836a4c02a61f18349317a0b0f9e99647b050f617`.
- `dev_validation.json`: 7/16~31, 54,720행·189경기, 83,497,429 bytes. SHA256 `b7d72da9a168d50fafe1a667500ec8df2ae3d40d4db5278d1d88c209622fcd15`.
- `final_test.json`: 8/1~31, 123,939행·421경기, 189,067,433 bytes. SHA256 `8e3b31d8fd1b9e725f0b3be752e19933b46f2a197f6d692c3d79b509614c1322`.

세 파일은 전용 worktree의 Git 제외 `artifacts/temporal_quality_games/`에 있다. `final_train.json`은 만들지 않았으며 앞의 두 배열을 연결하면 된다. 개발 전체 행 파일의 SHA256은 `df08b39aeedbb88aa7f9726a4762dd001616fb031b6204dfd7ae735f33908599`다. 각 분할 metadata에 이 상위 행 파일 해시를 기록하고, manifest의 `source_feed_hashes`에 모든 게임의 원문 feed SHA256을 연결했다. 단순 재파싱으로 과거 해시를 붙이지 않고 실제 저장 bytes를 검증했다.

대형 행 JSON은 객체 배열을 조금씩 해석하는 decoder로 분할했다. 완성한 임시 파일을 교체해 부분 JSON이 읽히지 않도록 했다. 부모에 개발 파일 완료를 알린 뒤 그 두 파일은 변경하지 않았다. 최종 검증은 세 파일을 다시 해시로 읽었고 행 파일을 다시 쓰거나 네트워크로 재수집하지 않았다.

## 최종 metadata 계약과 보존 범위

`results/collection_audit.json`과 `results/collection_manifest.json` 최상위에 다음 값을 동일하게 제공한다: `interval=[2025-06-01,2025-08-31]`, `collection_complete=true`, `schedule_days_requested=92`, `schedule_days_received=92`, `selected_game_count=1187`, `parsed_game_count=1187`, `failures=[]`.

두 파일의 `split_exports`는 `dev_train`, `dev_validation`, `final_test`를 키로 둔 dict다. 각 항목에는 interval, rows, game_count, gamePks, bytes, sha256, source_row_file_sha256, 정렬 키, 상대 경로와 절대 경로를 넣었다. 파일의 bytes·해시와 기간이 다르거나, 완료라고 주장하면서 날짜·경기 수·실패 상태가 맞지 않으면 검증을 거부한다. 이 계약은 부모의 최종 technical 판정에 사용할 수 있다.

새로운 개발·최종 검사 단계의 metadata는 `results/collection_development/`와 `results/collection_final_test/`에 있다. 게임별 감사는 **게임 한 개를 한 줄에 담는 JSON 배열**로만 압축했다. 값·실패·원문 해시를 모두 보존하고 해당 파일의 새 bytes·SHA256을 상위 manifest에 반영했다. 기존 PR128의 집계 파일은 그대로 두었다.

네트워크 동시는 최대 3개, 실패 시 추가 시도는 최대 2회다. 각 요청의 URL·원래 UTC 접근 시각·응답 bytes·SHA256·캐시 사용·시도와 실패를 보존한다. 다른 worktree의 캐시는 읽기 검증 후 자기 캐시에 복사했으며 수정하지 않았다. 원문 feed·일별 schedule·lookup·행별 자료는 모두 ignored artifacts에 있고 Git에는 코드·테스트·집계·manifest·이 문서만 포함했다. 공개 접근을 원문 재배포 라이선스로 해석하지 않는다.

원필드의 단위·방향·평가 시점 미확인을 유지한다. startSpeed의 mph 표시 근거는 기존 프로토콜대로 별도로 기록하지만 실제 평가 위치·시점을 확정하지 않는다. 스윙/미스 관계, knot·계수·모델 선택, 신체→구위 연결, 3D 회전 복원과 개입 효과는 이 수집 작업에서 계산하지 않았다.

## 재현·검증

```powershell
..\.venv\Scripts\python.exe -B -m unittest coach_hypotheses.temporal_quality.test_collect coach_hypotheses.pitch_quality.test_collect_games coach_hypotheses.pitch_quality.test_mlb_feed
..\.venv\Scripts\python.exe -B -m coach_hypotheses.temporal_quality.collect
```

타깃 테스트 37개를 통과했다. 날짜 구간 일반화 후 기본 6/1~14의 동작과 테스트를 유지하고, 다른 기간은 별도 cache/results를 요구한다. Unicode·작은 chunk·잘못된 JSON·경계 날짜·원값 보존·정렬·중복·원본/분할 해시·원자적 파일 공개·개발 우선 공개·92일 metadata 계약·불완전 수집과 변조된 분할 거부를 검사했다. synthetic fixture는 실제 표본으로 집계하지 않았다. 실제 수집 결과와 세 분할의 bytes·해시도 별도로 재검증했다.
