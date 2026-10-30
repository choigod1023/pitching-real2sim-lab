# 재개 기록

2026-10-30. 사용자 승인 범위의 다른 투구 적용 검증을 마쳤다. 자동 루틴은 재개하지 않는다. 원래 자료나 기존 결과 파일은 바꾸지 않았다.

- 선행: PR135, e2f7131da61a63757aad4baf112e08cd81373dae. 최신 원격 main 63fc06c2c37b9af55c48d341a5c5e39d8a6deb2e에서 별도 worktree/branch를 만들고 필요한 선행 체인을 fast-forward했다.
- 브랜치: codex/pitch-transfer-validation-20261030.
- 사전 계획 커밋: 01e0d06. PROTOCOL.md를 새 결과 계산 전에 고정했다.
- 로컬 계산: 이 worktree의 artifacts/pitch-transfer/<pitch>/{input,baseline,normalized,absolute,refit} NPZ와 report/direct/refit-partial JSON. 완료 계산은 재실행하지 않는다.
- 보존 결과: results/summary.json, 투구별 json.gz, suite_verification.json. 데이터 원본, C3D, 영상, reference NPZ는 업로드하지 않는다.
- 판정: 같은 선수 고정 전이 0/4, 다른 선수 고정 전이 0/4. 절대 입력 차이 보조 전이도 각각 0/4. 같은 선수의 개별 재탐색 총 960쌍은 예산 종료, 전부 baseline fallback. 가능성의 부정 증명이나 실제 선수 성능 검증이 아니다.
- 새 영상: 합격 후보가 없어 미생성. 실패 후보를 향상 영상으로 출력하지 않는다.
- 테스트: 새 15/15. 전체 479 = 467 통과 + 기존 오류 10 + skip 2. 기존 대비 새 오류/실패 0. tests가 생성한 이 worktree의 model.xml만 복원했다.
- 다음: README의 여러 투구 공동 제약/미사용 평가 설계와 전체 순일·root힘 한도 점검을 별도 사전 등록한 뒤 실행한다. 지금 결과를 구속 예측 오차 감소나 훈련 처방으로 해석하지 않는다.
- PR 생성 후 첨부를 시도하되 기존 100개 첨부 한도를 만나면 URL만 보존한다. 다른 작업의 첨부를 제거하지 않는다. 병합은 하지 않는다.
