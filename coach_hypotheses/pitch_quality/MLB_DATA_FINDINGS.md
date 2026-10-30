# MLB 공식 JSON 투구 자료 확보·스키마 검증

2026-10-30. `DATA_PROTOCOL.md`를 고정한 뒤 공식 JSON 한 경기를 확보하고 파서를 실행했다. **실제 관측 자료 확보와 스키마 검증은 통과했지만, 구위 관련성·개입 효과·자세 처방은 검증하지 않았다.** Mendeley 원문 파일과 Statcast CSV의 기존 HTTP 403 실패는 그대로 남긴다. 이번에 받은 것은 대체 경로의 MLB 공식 JSON이다.

## 고정 표본과 출처

[공식 schedule](https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2025-06-01)의 2025-06-01 완료 정규시즌 15경기 중 숫자 `gamePk`가 가장 작은 777677(DET@KC)을 선택했다. 성적이나 공의 특성으로 고르지 않았다. 응답 안의 실제 `link=/api/v1.1/game/777677/feed/live`를 따라 [공식 feed](https://statsapi.mlb.com/api/v1.1/game/777677/feed/live)를 받았다. HTTP 200, 725,682 bytes이며 최초 저장 시각은 2026-10-30 00:41:10 UTC다.

Feed SHA256: `7f40e379d47c491aa1318190a030fe6ba5ff9f609dd9f42ebc2c852a692d0fc7`. 사전 접근 감사의 해시와 같다. 동적 feed이므로 재접근 시 현재 해시를 다시 기록하며 과거 해시를 강제로 붙이지 않는다. schedule·feed·공식 코드표·metrics의 각 URL, bytes, 원문 해시, 접근 시각은 `results/mlb_manifest.json`에 기록했다.

원문 JSON과 투구별 파싱 자료는 전용 worktree의 Git에서 제외한 `artifacts/mlb_feed_audit/`에만 저장했다. Git에는 코드·테스트·집계 감사·메타데이터만 포함한다. 공개 GET 성공을 원문 재배포 허용이나 CC BY 라이선스로 해석하지 않는다.

## 실제 필드와 검사 결과

`liveData.plays.allPlays` 69개에서 `isPitch=true`인 이벤트 256개를 추출했다. 8명 투수의 반복 투구가 있으며, boxscore의 두 팀 `pitchesThrown`과 `numberOfPitches` 합은 각각 122+134=256으로 일치한다. `(gamePk,atBatIndex,pitchIndex)`와 `playId` 중복은 각각 0건이다.

다음 25개 수치 필드는 각각 256/256개가 존재하고 모두 유한수였다. 누락·잘못된 자료형·비유한 값은 0건이다. 원값과 원필드명을 보존하며 변환하지 않았다.

- `pitchData`: `startSpeed`, `endSpeed`, `extension`, `plateTime`.
- `pitchData.breaks`: `spinRate`, `spinDirection`, `breakAngle`, `breakLength`, `breakY`, `breakHorizontal`, `breakVertical`, `breakVerticalInduced`.
- `pitchData.coordinates`: `pfxX`, `pfxZ`, `pX`, `pZ`, `x0`, `y0`, `z0`, `vX0`, `vY0`, `vZ0`, `aX`, `aY`, `aZ`.

`matchup.pitcher.id`, `matchup.batter.id`, `matchup.batSide.code`, `matchup.pitchHand.code`, `details.type.code`, `details.call.code`도 전 투구에서 유효했다. `details.code`와 `details.call.code` 불일치는 0건이다. 전체 개별 값이나 결과별 비율은 이 보고서에 내보내지 않았다.

## 신원·카운트·결과 코드의 의미

AB 안의 투수 교체 6건은 모두 첫 투구 이전이었다. 첫 투구 뒤 투수·타자 교체로 신원 연결을 격리한 투구는 0개다. 파서는 첫 투구 뒤 투수 교체, PH 또는 종류가 불명확한 공격 교체를 발견하면 해당 AB 전체의 신원 연결을 `uncertain_quarantined`로 기록하고 연결용 ID를 `None`으로 둔다. 명시적 PR은 주자 교체로 구별한다. 원래 matchup 값은 감사용으로 별도 보존한다.

`event.count`는 사후 값이다. AB 시작 balls/strikes=0/0에서 출발해 비투구 이벤트도 순서대로 반영하고, 각 투구 직전 상태를 별도로 저장했다. 첫 투구 뒤 비투구 이벤트가 실제로 20건 있으므로 투구끼리만 한 칸 이동하면 안 된다. 256개 사전 카운트를 복원했고 잘못되거나 빠진 이벤트 카운트, AB 최종 카운트 불일치는 모두 0건이었다. 테스트는 자동 볼·스트라이크가 비투구로 들어오는 경우, 첫 투구 전 자동 판정, AB 전환과 잘못된 카운트 뒤 불확실성 전파도 확인한다.

[공식 pitchCodes](https://statsapi.mlb.com/api/v1/pitchCodes) 39개 항목을 실제로 받아 각 투구의 코드와 대조했다. 전 투구가 목록에 있었다. 원문 `swingStatus`, `swingMissStatus`, `swingContactStatus`, `contactStatus`, `buntAttemptStatus`, `pitchStatus` bool을 그대로 보존한다. 공식 목록은 `T`(foul tip)에도 `swingMissStatus=true`를 지정하므로 이를 연구용 `whiff`로 바꾸지 않았다. 별도 헛스윙 정의나 결과 연관 모델은 만들지 않았다.

## 단위와 관측 한계

Feed에는 단위 메타데이터가 없다. [공식 metrics](https://statsapi.mlb.com/api/v1/metrics)는 `releaseSpinRate=RPM`, `releaseExtension=FT`, `releaseSpeed=MPH`를 명시하지만 `spinRate`, `extension`, `startSpeed`와의 별칭 관계까지 확인해 주지는 않는다. 모든 feed 수치 필드의 단위·좌표 관례·시점은 이번 구현에서 `UNKNOWN`으로 유지한다. 값의 범위를 보고 단위를 확정하지 않는다.

특히 `pfxX/pfxZ`와 Savant CSV의 `pfx_x/pfx_z`를 같은 단위로 복사하지 않으며, `x0/y0/z0`를 손 릴리스 위치로 이름 바꾸지 않는다. 이 표본의 `y0`는 50.0000058~50.0068467이다. `spinDirection`을 3D 회전벡터나 회전 효율로 해석하지 않는다. [Savant 공식 CSV 문서](https://baseballsavant.mlb.com/csv-docs)의 정의는 참고 자료이며 이 JSON의 다른 필드명을 자동으로 확정하는 근거로 쓰지 않았다. 현재 [StatsAPI 문서](https://docs.statsapi.mlb.com/)는 로그인 보호를 확인해 공개 shell 이후 보호된 내용 추출은 진행하지 않았다.

arm angle, 손가락·그립·심 방향, 3D 회전벡터·회전 효율, 의도 목표점, 전신 운동학을 같은 투구에 연결하지 못했다. 미관측 값은 `None`으로 남긴다. ball-flight 계산, 3D 보간, 자세·근력·일·개입 효과 추정은 하지 않았다.

## 재현과 상태

저장소 루트에서 공용 LAB Python으로 실행한다. `-B`는 bytecode 생성을 막는다.

```powershell
..\.venv\Scripts\python.exe -B -m unittest coach_hypotheses.pitch_quality.test_mlb_feed -v
..\.venv\Scripts\python.exe -B -m coach_hypotheses.pitch_quality.mlb_feed
```

처음 실행하면 고정된 schedule→실제 feed 링크→공식 코드표·metrics만 GET하고 정확한 응답을 캐시한다. 다음 실행은 해시를 검사한 캐시를 읽는다. `--refresh`는 같은 고정 표본을 다시 받아 현재 출처를 기록한다. 캐시의 bytes·해시·URL이 틀리면 조용히 대체하지 않고 오류를 낸다.

타깃 unittest 18개가 통과했다. 실자료 검증 결과는 `results/mlb_audit.json`과 `results/mlb_verification.json`에 있다. `technical_access_ready=true`, `conditional_study_ready=false`, `body_to_spin_ready=false`, `spin_3d_ready=false`다. 한 경기 확보는 구위 연구의 최종 표본·단위·평가 설계를 통과했다는 뜻이 아니다. synthetic fixture는 파서의 오류 대응 검사에만 사용했고 연구 관측 수에 포함하지 않았다.
