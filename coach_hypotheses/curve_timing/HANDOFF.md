# 재개 기록

2026-10-30. 곡선중심시각검증과공식MLB JSON파서/스키마감사를완료했다. 사전기준은curve_timing/PROTOCOL.md와pitch_quality/DATA_PROTOCOL.md(원commit b784cac)다. 부모 REPORT.md/FINDINGS.md/results와pitch_quality/MLB_DATA_FINDINGS.md/results를먼저읽는다.

동일399투구/98명/666쌍의중심시각범위중앙값8.200→1.389ms,p95 22.300→2.580ms. 주20Hz구속변화MAE1.024556→1.018900이지만0기준1.015200보다나빠개발예측조건미통과. 이미실행한10773적분/3996예측검산은다시반복하지않는다. 안정성개선을선수효과나기존절대구속오차수정으로부르지않는다.

공식schedule의사전날짜2025-06-01/최소완료Rgame777677→feed실제link로접근성공.256투구/8투수/25수치필드전부finite,boxscore/count/신원일치. agent와부모feedhash=7f40e379d47c491aa1318190a030fe6ba5ff9f609dd9f42ebc2c852a692d0fc7. 원JSON과파싱행은자기root/artifacts/mlb_feed_audit캐시에만있다. CSV403이해결된것은아니며대체공식JSON을확보한것이다.

다음은feed필드의단위/좌표/시점확인과연구용결과정의다. official swingMissStatus의foul tip처리를명시하고,이후반복경기/구종/투수별조건부분석의표본/분할/기준을사전에고정한다. 단위를모두확인못하면가정/미확인과단위불변기술분석을구분하고물리시뮬레이션을잠근다. 직접관측없는3D회전/심/그립/의도목표를0으로만들지않는다.

현재technical_access_ready만true이며conditional_study_ready/body_to_spin/개입효과는false다. 기존구속gate/실패/자동화상태를바꾸지않았다. 새코드는최신main의별도codex worktree에서격리하고미병합선행만명시적으로가져온다. 원본/투구별MLB자료/대량CSV를커밋하지않는다. 새선수모집·외부연락·유료·배포·PR병합은하지않는다. 전체355중343통과/기존10오류/skip2이며새27테스트통과. 자기model.xml테스트부작용만복원했다.
