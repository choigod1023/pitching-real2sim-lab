# 투수–공–타자 연결 모듈

실측 투구 자세와 별도 타격 궤적을 한 장면에서 재생하고, 배트 스윗스폿 구간과 참조 공의 근접을 계산한다. 별도 API는 기존 MLB 모델의 스윙·통계적 미스 확률을 복원해 결합한다. 두 자료를 같은 실제 대결로 취급하지 않는다. **현재는 실제 선수의 성능 향상·같은 힘/일에서의 관절 수정 효과를 검증한 프로그램이 아니다. 훈련·의료 조언을 제공하지 않는다.**

## 실행

저장소 root에서 실행한다. Python은 기존 lab의 전용 환경을 사용한다.

```powershell
New-Item -ItemType Directory -Force artifacts/batter_contact/external
gh release download dataset-v1 --repo drivelineresearch/openbiomechanics --pattern hitting_landmarks.zip --dir artifacts/batter_contact/external
gh release download dataset-v1 --repo drivelineresearch/openbiomechanics --pattern hitting_c3d.zip --dir artifacts/batter_contact/external
& '../.venv/Scripts/python.exe' -m coach_hypotheses.batter_contact.run
& '../.venv/Scripts/python.exe' -m unittest coach_hypotheses.batter_contact.test_contact -v
```

기본 입력:

- `../openbiomechanics`: 기존 실측 pitching 자료와 hitting metadata.
- `artifacts/batter_contact/external`: 공식 release `dataset-v1`의 `hitting_landmarks.zip`과 `hitting_c3d.zip`. GitHub asset SHA256과 비교한다. 두 원본 ZIP은 Git 제외다.
- `../temporal-pitch-quality-20261030`: PR129 모델 적합 감사 JSON·기존 예측. summary의 해시와 저장 계수를 대조하며 검사 자료로 다시 학습하지 않는다.
- `../temporal-collection-agent-20261030/artifacts/temporal_quality_games`: 기존 MLB 세 분할이다. 새 기간을 수집하거나 새로운 미사용 검사라고 부르지 않는다.

모든 경로는 CLI 인자로 바꿀 수 있다. raw·행별자료는 ignored artifacts에만 둔다. 결과는 `results/integration.json`, `results/response_integration.json`, 생성한 투구 장면은 `media/pitcher_batter_overlay.mp4`다. 성공 출력에는 바로 사용할 수 있는 절대 경로 Markdown 영상 태그를 내보낸다. **성공을 보고할 때 이 투구 영상을 채팅에 함께 표시한다.**

## 세 부분을 구분한다

1. `response.py`: 같은 MLB 공 원필드의 P(swing), P(strict S/W | swing), P(official statistical miss | swing)를 계산한다. 노스윙·스윙/해당 미스·스윙/그외의 세 확률은 합계1이다. 그외를 물리적 contact로 바꾸지 않는다. 공식 미스에는 파울팁이 포함되며 strict에는 피치아웃 미스가 제외되는 등 정의 차이가 있다. physical_contact_probability는 None이다. OBP 신체 좌표를 MLB 입력처럼 보내면 거부한다.
2. `geometry.py`/`scene.py`: 처리된 투수 관절과 실제 손/스윗스폿 궤적을 같은 장면에 배치한다. 공은 기록 구속을 초기값으로 쓴 중력 포물선이고 기준 스윗스폿을 지나도록 구성한다. 따라서 기준 근접은 설계상 정렬이다. 실제 공 좌표 재현, 타격 반작용, 회전·타구 속도/발사각은 검증하지 않았다. 표면 사이 signed clearance≤0이면 정한 구간에 겹친다는 기하 판정이며 실제 배트 전체의 컨택/헛스윙이 아니다.
3. `render.py`: 원 투수 재생, 기준/12ms 지연 타자의 전신·배트 겹침, 공 참조 궤적, 근접 구간 확대를 보여준다. 전체 투구 뒤 접촉 구간을 다시 재생한다. 투수 기록이 끝나면 마지막 프레임을 유지하고 화면에 표시한다. 원/비교의 배트 축 각도·점 위치 차이를 보여주되 투수 수정은0이다. 헤드 원과 완전한 배트 외곽선은 도식이며 측정된 해부학적 점/배트 끝점이 아니다.

`success_bundle`은 성공한 결과에 검증한 MP4와 투수 장면 표시를 요구한다. engineering/prediction 성공과 athlete_candidate를 구분한다. **현재 athlete_candidate 성공은 차단**하며 단순 bool 체크리스트로 우회할 수 없다. 영상 없는 성공이나 실제 구위 향상처럼 보이는 출력으로 대체하지 않는다.

설계는 `PROTOCOL.md`, 결과 전 식별자 정정은 `AMENDMENT_01_IDENTITY.md`, 실제 수치·실패·검산은 `FINDINGS.md`를 읽는다. 기존 자료·PR129 수치·Real2Sim gate는 수정하지 않았다.

## 자료 인용과 라이선스

관절·타격 궤적은 Driveline Baseball의 [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics), [hitting 문서](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_hitting), [자료 라이선스](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md)를 따른다. 파생 동작·수치·생성 영상은 **CC BY-NC-SA 4.0** 및 상류 인용 조건을 유지한다. 코드의 MIT 라이선스가 자료 조건을 대체하지 않는다. MLB 공개 API 자료는 별도의 CC 라이선스로 간주하지 않으며 원문·투구별 export·모델 적합 원자료는 업로드하지 않는다.
