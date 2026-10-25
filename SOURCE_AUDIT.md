# 공개 투구 자료의 이탈·구속·손 마커 정의 감사

2026-10-25. OpenBiomechanics 소스 커밋 `44b98dae05cceb016f080ab39d105c85b8639084`와 현재 내려받은 C3D를 확인했다. 이 문서는 구속 향상 실험 결과가 아니라 측정 대상과 모델 가정을 구분하는 기록이다.

## 직접 확인한 내용

- 메타데이터의 411개 투구와 100개 정적 모델 파일을 모두 읽었다. 실패 0개. 파일명에 기록된 0.1mph 단위 구속은 411개 모두 메타데이터와 일치했다. 동일한 값의 복제 여부를 확인한 것이며 독립적인 구속 측정 검증은 아니다.
- 동적 C3D 368개, 정적 C3D 85개에는 EVENT 그룹이 있지만 사용 이벤트 수는 모두 0이었다. 511개 파일 모두 헤더의 이름 있는 이벤트와 0이 아닌 이벤트 시간도 없었다. 비어 있는 헤더 슬롯 18개를 실제 이벤트 18개로 세지 않았다.
- 511개 POINT 라벨에 `ball`이라는 문자열이 포함된 이름은 없었다. `CustomMarker10` 같은 미확인 이름이 있으므로 공을 절대 측정하지 않았다는 주장으로 확대하지 않는다. 공개 라벨만으로 식별된 공 궤적은 확보하지 못했다.
- 정적 파일 100개 중 99개에는 정확한 RFIN/RWRA/RWRB 및 LFIN/LWRA/LWRB 이름과 유효한 비공선 프레임이 있다. 나머지 1개에는 `Skeleton_001_` 접두사가 붙은 손 마커가 있다. 현재 감사 코드는 이름을 자동 치환하지 않으므로 이 1개를 실제 마커 누락으로 해석하면 안 된다. 동적 파일은 정확한 이름 기준 양손 각각 408개가 유효하다.

실행: `../.venv/Scripts/python.exe source_audit.py`. `artifacts/source-audit/report.json`에 집계, `files.json`에 파일별 SHA256과 검사 결과를 남긴다. 유효 프레임은 좌표 유한성, 비음수 잔차, 삼각형 외적 크기 >1e-8 m²로 판정했다. 이는 손 해부학이나 부착 정확도 검증이 아니다.

## BR 및 구속의 출처

공개 추출 노트북은 구속을 `bp_poi_metrics.pitch_speed_mph`에서 가져오고, BR은 `bp_events.BR_time`을 각 시계열에 조인한다. 데이터베이스 필드를 어디서 가져오는지는 확인했지만 그 값을 만드는 검출 알고리즘은 이 코드에 없다. `baseball_pitching/code` 아래 모든 파일을 확장자 제한 없이 읽고 노트북은 셀 소스를 파싱했다. CMO.v3s, 모델 .mdh, 네 .vik 및 두 노트북에서 BR 생성과 구속 장비·위치·시점을 확정하지 못했다. 검출 방식이 틀렸다고 판단한 것은 아니다.

출처: [공개 추출 노트북](https://github.com/drivelineresearch/openbiomechanics/blob/44b98dae05cceb016f080ab39d105c85b8639084/baseball_pitching/code/py/copy_anonymize_c3ds_pull_db_csvs.ipynb), [데이터 정의](https://github.com/drivelineresearch/openbiomechanics/blob/44b98dae05cceb016f080ab39d105c85b8639084/baseball_pitching/README.md).

Driveline의 2021년 연구실 소개에는 당시 기존 장비로 Stalker Pro II 레이더가 나온다. 그러나 이를 공개 411개 투구 각각의 측정 장비·측정 위치·릴리스 구속 정의로 연결할 근거는 확보하지 못했다. [연구실 소개](https://drivelinebaseball.com/blogs/blog/not-just-a-pitching-lab-the-brand-new-driveline-sports-science-laboratory).

Qualisys의 현재 야구 워크플로는 수동 Release 입력이 없으면 던지는 손의 최대 각속도를 사용한다고 설명한다. 이는 비교 가능한 업체 정의일 뿐 OpenBiomechanics가 해당 규칙을 썼다는 증거가 아니다. 따라서 현재 BR을 이 규칙으로 덮어쓰지 않는다. [Qualisys 이벤트 정의](https://docs.qualisys.com/baseball/Content/PAF%20Baseball/Appendix%20A%20-%20variable%20definitions.htm).

## 손 기하와 공의 구분

공개 모델의 RHA 추적 마커는 RFIN/RWRA/RWRB다. RHJC는 RFIN/RWRA/RWJC를 참조하고 MCS_ML=-.015, MCS_AP=-.015를 지정한 파생 랜드마크다. RHJC를 RFIN 또는 공 중심으로 동일시하지 않는다. 축 변환의 완전 재현 없이 이 두 숫자를 세계 좌표 오프셋으로 적용하지 않는다. [모델 정의](https://github.com/drivelineresearch/openbiomechanics/blob/44b98dae05cceb016f080ab39d105c85b8639084/baseball_pitching/code/v3d/model/v6_model_hybrid_lm.mdh).

공개 마커 그림에서 RFIN은 손 부위에 있다. 그림만으로 특정 중수골의 정확한 부착 지점, 개인별 피부 오프셋 또는 공 중심을 결정할 수 없다. 현재 모델의 손목에서 15cm 공 중심은 여전히 가정이다. [마커 그림](https://github.com/drivelineresearch/openbiomechanics/blob/44b98dae05cceb016f080ab39d105c85b8639084/baseball_pitching/imgs/image2.png).

## 다음 실험과 식별 한계

정적 보정 파일을 같은 세션의 동적 파일에 대응시켜 손 클러스터의 크기와 프레임을 구속 목표와 무관하게 추정한다. 우선 접두사 처리의 충돌 여부와 잔차를 확인한 뒤, 기존 투구별 템플릿과 정적 템플릿을 동일한 처리 조건에서 비교한다. 처음 10명·29개 투구는 계속 개발 자료로 표기한다. 성능이 좋아진 필터만 고르지 않는다.

이로써 검사할 수 있는 것은 관찰된 손 기하의 일관성과 조건부 속도 예측이다. 실제 공 중심·손가락 접촉·이탈 순간의 상대속도는 직접 공 추적 없이 분리 식별되지 않는다. 스칼라 구속 하나로 그립, 이탈 시점과 힘을 동시에 보정하면 여러 해가 가능하므로 이를 실제 동작 수정 효과로 해석하지 않는다. 정적 파일은 근력·외부 어깨 일 예산·편안함도 알려주지 않는다. 현재 모델의 개선 후보 최적화는 계속 보류한다.


## 같은 실행에서 이어서 수행한 정적 템플릿 실험

`hand_geometry.py --template-source static`을 추가하고 세션별 정적 C3D의 유효 프레임을 손목 중점 기준 로컬 좌표로 변환한 중앙값을 사용했다. 정확한 이름 또는 일관된 `Skeleton_001_` 이름을 허용하되 두 체계가 동시에 존재하면 거부한다. 기본값 trial은 유지했다. 동일한 10명·29개 투구 전부 성공했다.

다섯 조건의 선수 내 변화 MAE는 raw 9.104, Cartesian Butterworth20 1.770, Cartesian Gaussian 2.442, 정적 강체 적합 8.319, 정적 강체 평활화 2.167km/h였다. 무변화 기준 0.861보다 모두 못했다. 강체 평활화의 절대 구속 MAE는 14.988km/h, 다른 선수 평균 기준은 6.637이었다. 정적 강체 적합의 마커 RMSE 평균은 3.247mm였다. 기존 투구별 템플릿의 평활화 변화 MAE 2.368과 적합 RMSE 1.621mm도 함께 보존한다. 정적 자료의 손 모양과 투구 중 손 모양이 일치한다고 가정할 수 없으며 차이를 특정 조직 또는 원인으로 확정하지 않는다.

결과: `artifacts/static-hand/report.json`, `geometry.csv`, `predictions.csv`. 이는 해석적 가정 공 중심 속도 비교이며 물리적 동작 변경이나 실제 구속 증가의 증거가 아니다. 다음에는 BR 주변 사전 지정 ±1/±2프레임을 모두 평가해 시간 정렬 가정에 대한 민감도를 검사한다. 관측 속도에 맞는 시점을 골라 최종 예측으로 사용하지 않는다.
