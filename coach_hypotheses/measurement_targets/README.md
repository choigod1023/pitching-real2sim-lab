# 손–공 입력 측정 우선순위의 조건부 계산

거리를 ±1mm로만 좁혀도 기존 시점·필터 가정을 남기면 구속 시나리오 폭이
11.32~17.19km/h 남습니다. 거리와 함께 시점·처리 선택을 제한해야 폭이 작아집니다.
실제 측정을 추가한 실험이 아니라 가정 범위를 좁힌 계산입니다.

- [계산 전 고정한 프로토콜](PROTOCOL.md), [결과와 한계](FINDINGS.md)
- [24조건×5구×2seed CSV](results/summaries.csv), [전체 파생 요약](results/scenario_results.json)
- [최종 별도 기록 비교](results/recorded_comparison.json), [원본 보존](results/manifest.json)
- [테스트와 부작용 복원](results/test_verification.json)

```powershell
& ../.venv/Scripts/python.exe -m coach_hypotheses.measurement_targets.run --data ../openbiomechanics/baseball_pitching/data
& ../.venv/Scripts/python.exe -m unittest coach_hypotheses.measurement_targets.test_targets coach_hypotheses.stage2.test_uncertainty -v
& ../.venv/Scripts/python.exe -m unittest discover -v
```

마지막 전체 테스트는 기존 artifacts/coupled-body/model.xml을 쓰므로 격리 worktree에서
실행하고 자신의 실행 직전 바이트와 생성 후 해시를 확인해 복원해야 합니다.
기존 단계1/2 결과·기준·FAIL을 보존하고 단계3은 구현하지 않았습니다.

자료: [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics),
Driveline Baseball Research and Development, [openbiomechanics.org](https://openbiomechanics.org).
기존 마커/관절중심의 기구학을 재사용해 조건부 MC 요약을 추가했습니다.
results/ 파생자료는 [CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 및
[원자료 조건](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md),
상위 NOTICE.md를 따릅니다. 코드의 MIT와 파생자료 라이선스를 구분합니다.
원본 C3D·영상·대형 배열은 포함하지 않습니다. Driveline의 보증을 받지 않았습니다.
