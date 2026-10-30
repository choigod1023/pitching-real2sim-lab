# 반복 경기 구위 관측 분석 실행

먼저 `CONDITIONAL_FINDINGS.md`를 읽는다. 회전·움직임 추가 모델은 포심 사전 기준 미통과다. 연구용 관측 분석이며 훈련·의료 조언이 아니다. 실제 신체 조정 효과·안전·편안함은 검증하지 않았다.

`CONDITIONAL_PROTOCOL.md`와 결과 계산 전 보충인 `CONDITIONAL_EXECUTION_NOTES.md`가 실행 기준이다. 기존 결과 파일과 기존 Real2Sim gate는 수정하지 않았다.

저장소 root에서 사용한 Python은 `../.venv/Scripts/python.exe`다. 새 의존성 설치는 필요하지 않았다.

```powershell
& '../.venv/Scripts/python.exe' -m coach_hypotheses.pitch_quality.collect_games
& '../.venv/Scripts/python.exe' -m coach_hypotheses.pitch_quality.conditional --input 'artifacts/mlb_quality_games/parsed_pitches.json'
& '../.venv/Scripts/python.exe' -m unittest coach_hypotheses.pitch_quality.test_collect_games coach_hypotheses.pitch_quality.test_conditional coach_hypotheses.pitch_quality.test_mlb_feed -v
& '../.venv/Scripts/python.exe' -m unittest discover -v
```

실제 첫 수집은 격리 agent worktree `../mlb-quality-collection-agent-20261030`에서 수행했다. 부모의 실제 모델 명령은 다음과 같았다.

```powershell
& '../.venv/Scripts/python.exe' -m coach_hypotheses.pitch_quality.conditional --input '../mlb-quality-collection-agent-20261030/artifacts/mlb_quality_games/parsed_pitches.json'
& '../.venv/Scripts/python.exe' artifacts/conditional_pitch_quality/independent_check.py
```

두 번째 검산 스크립트와 행별 예측은 로컬 ignored artifact다. `results/conditional_verification.json`에 검산 수·차이를 보존했다. 전체 테스트는 unittest discovery를 Python runner로 실행하고 결과의 test.id를 stage0의 표시 문자열 속 완전한 ID와 대조했다. 알려진 10개 오류의 원인도 같았다. `results/conditional_test_verification.json`과 `conditional_full_suite.txt`에 남겼다.

변경 파일:

- `collect_games.py`, `test_collect_games.py`: 날짜 고정 수집·해시/캐시·행별 원문·게임별 감사.
- `conditional.py`, `test_conditional.py`: 훈련 내 전처리, 조건부 로지스틱, 경기 검사, 투수별 손실, bootstrap·판정.
- `FIELD_DEFINITIONS.md`: 공식 판정, 원필드 정의·단위 미확인과 실제 결측.
- `results/collection_{audit,games,manifest}.json`: 행별 원자료를 제외한 수집/감사 기록.
- `results/conditional_{summary,manifest,verification,test_verification}.json` 및 전체 테스트 로그: 수치·분할·계수·판정·검산.

원자료·행별 export는 Git 제외 artifacts에만 둔다. MLB 공개 API 접근을 원자료의 CC 라이선스로 해석하지 않는다. 기존 OBP 파생자료의 CC BY-NC-SA 4.0 인용·라이선스는 유지하며 다른 자료에 전용하지 않는다.
