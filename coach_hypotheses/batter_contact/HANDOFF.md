# 투수–타자 연결 인계

**최신 사용자 정정이 우선한다.** 사용자가 원한 것은 투수 동작 수정으로 구위가 달라지는 비교 영상이며, 현재의 타자 데모는 대체물이 아니다. 기본 실행의 데모 생성·표시를 중단했다. `PITCHER_PERFORMANCE_VIDEO_POLICY.md`를 먼저 읽는다. `--show-demo`는 명시적으로 요청받은 진단에만 사용한다. 현재 검증된 투수 구위 향상 후보 영상은 없다. 아래는 기존 PR130 구현 이력이다.

전용 branch/worktree `codex/pitcher-batter-contact-20261030` / 동명. 최신 원격main63fc06c에서 만든 뒤 필요한 선행 PR129의8999ab2를 명시해 fast-forward했다. 프로토콜3daae1d, 기하 계산 전 타자 ID 정정ad9fec6이다. 이전 연구 결과·Real2Sim gate·자동화 상태를 바꾸지 않았다.

이번 구현은 두 부분이다. 기존 MLB 스윙·미스 확률을 복원/결합한 `response.py`와, 실측 투수/별도 타자 궤적을 배치한 조건부 기하 및 영상이다. 서로 같은 실제 대결이 아니다. 실제 신체→공 지표→타구 결과의 검증된 연결로 혼동하지 않는다. `FINDINGS.md`에 수치·가정·실패·한계를 구분했다.

사용자는 **성공한 결과가 있으면 항상 투수 시뮬레이션 영상을 띄워 달라**고 요청했다. 신규 성공 경로는 `success_bundle`로 검증한 MP4/투수 장면을 필수로 검사하고 `emit_video`로 절대 경로 Markdown 태그를 내보낸다. 최종 채팅에도 태그를 넣는다. 그래프만 보여주지 않는다. 현재 athlete_candidate 성공은 차단한다. engineering/prediction 성공 영상을 실제 선수 향상처럼 표시하지 않는다.

원 ZIP/C3D와 새 raw 선택표는 자기 ignored `artifacts/batter_contact`에 있다. OBP 기존 pitching 자료는 `../openbiomechanics`, MLB frozen fit/예측은 `../temporal-pitch-quality-20261030`, 행 자료는 `../temporal-collection-agent-20261030`이다. 원자료 보존 없이 정리하지 않는다. 생성 공유 영상은 `coach_hypotheses/batter_contact/media/pitcher_batter_overlay.mp4`,382프레임/30fps다. CC BY-NC-SA4.0 인용을 유지한다.

0/+12ms 모두 스윗스폿 근접이었다. 유리한 오프셋을 선택하거나 투수 개선으로 바꾸지 않았다. 힘/일·실제 컨택·타구 속도·발사각·구위 향상은 미검증이다. 확률의 other는 물리적 contact가 아니며 API의 physical_contact_probability는 None이다. 기존 확률130,472개는 정확히 재현했고 이는 계산 재현이지 실제 오차0이 아니다. 새 테스트15개 통과, 전체419개 중407통과·기존 오류10·skip2·새 회귀0이다.

향후 동일 투구의 직접 관측/힘·일 검증 근거가 생기면 body_mapping 검증을 추가한다. 현재 raw 필드나 이질적 기록을 맞춰 관절 최적화/실제 성능 향상으로 진행하지 않는다. 원래 목표는 같은 힘 능력·일 예산 아래 작은 동작 수정의 효과 검증이며, 타자에게 불리한 결과를 만드는지까지 확인하는 것이다.
