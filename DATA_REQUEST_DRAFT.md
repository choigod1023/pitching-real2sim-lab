# 데이터 협조 요청 초안 — 미발송

## 저자에게 요청할 내용

Subject: Request for anonymized synchronized baseball finger–ball data for non-commercial research

Dear authors,

I am developing a non-commercial research prototype to test whether measured pitching mechanics can predict ball release states on held-out pitches. I would like to ask whether anonymized data underlying your finger–ball pitching study can be shared for this purpose, and what permission and citation requirements apply.

The most useful subset would contain repeated pitches from the same participants, synchronized hand/finger and ball trajectories (or calibrated videos), timestamps, release-event definitions, and measured pitch velocity. Ball orientation/spin, arm motion and any measured contact forces would be valuable if available. Please distinguish measured and estimated quantities and provide trial identifiers linking the files, coordinate/unit definitions, sampling rates, synchronization/calibration information, and filtering procedures.

We would fit parameters on a subset of pitches and evaluate the remaining pitches without using their measured ball outcomes to tune the model. We would not claim that observational prediction demonstrates a training benefit. We can work with a limited anonymized subset, subject to your sharing conditions.

Thank you for considering the request.

[사용자 이름·소속·연구 목적 확인 후 발송할 초안. 현재 발송하지 않음.]

## 대학 연계 시 필요한 최소 측정 묶음

- 같은 투구에 공통 trial_id: 손목/손 자세, 검지·중지·엄지 접촉 근처 관측, 공 중심 및 가능하면 회전, 독립 구속. 가능한 항목만 나열하고 미측정을 0으로 채우지 않는다.
- 반복 투구와 선수 ID, 측정일/구종/노력지시/휴식/조건 기록. 서로 다른 날 자료는 일반화 검증에 유용하다. 최종 표본 수는 파일럿의 반복 변동·측정오차를 보고 정하며 임의 개수를 충분한 통계력으로 선언하지 않는다.
- 서로 다른 시야의 동기화 고속카메라와 공통 보정 공간, 실제 타임스탬프/노출/누락 프레임 기록. 문헌의960–2000fps는 참고 설계이며 장비만 같으면1km/h정확도가 보장되는 것은 아니다.
- 구속 장비의 모델·측정 위치·시점·출력 단위, 공 이탈 이벤트 판독 규칙과 판독자 간 차이. 영상 속도를 레이더에 강제로 맞추지 않는다.
- 팔/몸통 또는 전신 동작을 연계할 때 동일 trial_id와 동기화 정보. 접촉만 측정한 연구와 전신 자료를 서로 다른 선수 사이에서 이어 붙이지 않는다.
- 힘/일 제약을 다루려면 가능한 힘판·근활성·별도 능력 평가와 그 한계를 함께 기록한다. EMG를 근력 정답으로 취급하지 않는다.

이 초안은 측정 협의용이다. 선수 모집·인체 개입·장비 구매·외부 연락은 실행하지 않았다.
