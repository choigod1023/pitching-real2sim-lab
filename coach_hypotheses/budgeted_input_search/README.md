# 입력 시간 분배·일 예산 최적화 결과

2026-10-30. PR134에서 제안한 제약 정의 점검과 실제 입력 최적화를 구현했다. **기존 A 비교 기준을 통과한 조건부 +0.647360km/h 후보를 얻었다. 그러나 전체 순일과 외부 힘 벡터 peak까지 모두 불증가한 결과는 아니며, 실제 선수의 같은 능력·성능 향상을 검증한 것도 아니다.**

## 무엇을 바꿨는가

정해 둔 +2° 자세를 추적시키는 방식 대신, 마지막60ms의 일반화입력11개를 각3개의 시간기저로 바꾸는33변수의 직접 single-shooting 최적제어를 구현했다. 상태를 재설정하지 않고 실제 힘 입력으로RK4 적분한다. 각변경은 원래입력peak의2% 이내이며 원래성분/관절peak상한으로 제한한다. 이2%는공학적탐색범위이고 생리적허용량이아니다. 초기상태·그립15cm·관성·공칭BR을고정했고 레이더속도는 적합/선택에 쓰지 않았다.

목적함수는모델이탈구속, 부등식은실제적분한일·힘·자세다. 일 제약은 최적화 안에서 평가하며, 최적화 결과의 success와 무관하게 기존gate로 다시검사한다. 원래동작0변수를 항상가능한대안으로유지한다. 실제근력용량과관절별일상한의차이는 [CAPACITY_REVIEW.md](CAPACITY_REVIEW.md)에정리했다. 기존 같은선수4구가한투구의상한에서탈락했던2026-10-26결과는재계산하지않고재사용했다.

## A와 B는 서로 다른 가정이다

- **A, 기존 주분석:** 모든좌표별양/음의일·성분입력peak·정의된관절노름peak·어깨병진/회전순일이기준을넘지않아야한다. 기존gate epsilon1e−9를그대로유지했다. 전체순일이나어깨병진힘벡터노름의상한은이기존gate에없다.
- **B, 별도 탐색:** 내부관절별일은재분배할수있지만,5그룹의힘·속도내적을먼저합한총양/음의일·전체순일과외부어깨각그룹의양/음의일·순일을제한한다. 기존성분/관절peak도제한한다. 이는검증된근력곡선이아니다.

두정책은단순포함관계가아니다. A통과를B통과로바꾸거나B실패를A실패로덮어쓰지않는다. 둘모두최대회전3°/공중심이동20mm/방출방향1°를유지한다. 기준을본뒤완화한것이아니라 [PROTOCOL.md](PROTOCOL.md)를결과전에commit `d1a4bb1`로고정했다.

## 첫 탐색과 수치 복원 이력

초기SLSQP는A/B각360개고유평가예산에처음시작점에서모두도달했다. 나머지사전시작점은실행하지못했다. 기준외의통과후보를얻지못했지만, **미수렴이므로증가가능성이없다는결론이아니다.** A에서약+0.648km/h와작은일초과가남은점을찾아수치적제약복원으로이어갔다.

- [RECOVERY_PROTOCOL.md](RECOVERY_PROTOCOL.md), `e3ecd48`: 원A의evaluation296을고정시작점으로선택하고,두해상도의일·자세조건과gain≥0.5를동시에검사했다.0.5는이번복원의공학적목표이며실제최소유효차이가아니다.378쌍평가에서solver success였지만실제gate를약4e−12만큼넘어통과0개로보존했다.
- [NUMERICAL_TARGET_NOTE.md](NUMERICAL_TARGET_NOTE.md), `2a8fc73`: 최적화가비교epsilon경계가아닌실제비용상한을목표로삼도록엄격하게수정했다.400쌍평가예산에도달했고통과0개였다.
- [REDUNDANT_PEAK_NOTE.md](REDUNDANT_PEAK_NOTE.md), `bc4f8b9`: 입력생성기에서이미제한한peak의재계산에약7e−15Nm의반올림차이가남아,평평한중복부등식이raw복원을방해함을확인했다. 입력생성기의상한과최종gate는그대로두고최적화의중복peak항만제거했다. 실제일·자세·증가량조건은두해상도에서유지했다. **102쌍평가,3iteration에서수렴했고실제기존gate통과점1개를얻었다.**

후보조건·그립·이탈·힘허용량을늘리지않았고,최종판정epsilon도바꾸지않았다. 모든실패보고서와각실행코드를보존한다. `results/*.json.gz`는원JSON의무손실압축이며해제후SHA256을검사했다. 이전실패를삭제하지않고중복제약의수치적수정과새결과를구분한다.

## 선택된 조건부 후보

같은3034_1의같은기준과비교한세밀한해상도결과다.

- 모델구속 **113.466581973 → 114.113941654km/h**, 증가 **+0.647359680km/h**.
- 두해상도의증가량차이 **0.000000016041km/h**. 이는적분검산값이며실제예측오차가아니다.
- 최대회전차 **1.640645°**, 공중심최대이동 **5.831945mm**, 방출방향차 **0.601681°**. 기존3°/20mm/1°검사를두해상도에서통과했다.
- 회전차최댓값: 어깨모델좌표0.510242°, 팔꿈치힌지0.469683°, 전완marker-roll1.326708°, 손목마커회전1.640645°. 해부학적손목굽힘/어깨내회전 처방량이아니다.
- 기준대비모델어깨병진위치의최대차3.079628mm. 몸통·다리를새로예측한것은아니다.
- 양의성분별일합 **−1.074523J**, 음의일크기합 **−2.199193J**. 각좌표별양/음의일과기존성분/관절peak는기존A상한을지켰다.
- 어깨병진순일차약1.4e−12J, 회전순일 **−0.000174J**로A의비교범위안이다.

### 반드시 함께 읽어야 할 제한

흡수한에너지가더줄어들어 **전체순일은+1.124670J** 늘었다. 어깨병진힘의3축벡터peak도 **+0.010509N** 늘었다. 원래A는축별peak와정의된관절노름을검사하므로이물리량을완전하게제한하지않는다. 따라서 **“모든힘·일이동일하거나감소한증가후보”라고말하면틀리다.**

그룹정의로다시검사하면외부어깨회전그룹의양의일약+0.004481J, 음의일크기약+0.004656J도남는다. 이A후보를B기준에넣으면외부어깨그룹일·전체순일에서실패한다. B자체의초기탐색은미수렴이었으므로B에서가능한모든후보가없다고주장하지않는다.

끝상태의모델운동에너지는+1.248305J, 위치에너지는−0.123630J로합계+1.124675J다. 공의병진운동에너지차는약+0.824164J다. 단순표시만바꾸어구속을높인것이아니라,일·흡수·에너지분배가달라진계산결과다. 이는일반강체모델의BR까지의회계이며,실제공회전이나이후감속부담을검증하지않았다. **이탈전흡수를줄였다는이유로실제팔부담이나부상위험이낮아졌다고말하지않는다.** 후속감속에부담이옮겨갈가능성도별도검증대상이다.

## 불확실성과 실제 적용

고정한새A후보를기준과짝지어그립14/15/16cm, BR/−1/−2 source frame에서평가한9개기구학조건은 **+0.456743~+0.700882km/h**로모두양수였다. +1/+2 frame은저장범위밖이어서외삽하지않고제외했다. 이것은같은상태의사이트기하진단으로payload관성을바꾼재적분도아니고,실제이탈오차의분포도아니다. **95%구간이나실제예측오차로표시하지않는다.** [selected_sensitivity.json](results/selected_sensitivity.json) 참조.

기준실측기록130.195930km/h와기준모델의차이약16.729348km/h는여전히남는다. 수정동작은실제로관측하지않았으므로수정모델구속을원래실측기록에대입해'오차가줄었다'고말하지않는다. 새후보의실제개입효과구간도UNKNOWN이다.

이번결과는A의특정기계적비교기준에서가능한입력재분배를찾았다는것이다. 근력용량·근육경로·해부학축·직접손–공관측검증을대체하지않는다. 모델의입력33변수는그대로선수에게실행하라고지시할수있는훈련메뉴가아니다. 실제회전벡터와목표/도착점이연결돼있지않아RPM·구위·제구향상량은미검증이다.

다음연구에서확정해야할것은'같은일'을어떤물리량으로판정할지다. 순일증가를허용하는A비교결과와전체순일도제한하는B결과는목표가다르다. A결과를보고주목적을사후변경하지않으며,후속분석은원래/수정의외부힘벡터·그룹양/음의일·전체순일·이탈후부담까지명시해야한다. 최종목표인실제같은능력의성능향상을완료했다고처리하지않는다.

## 검증과 영상

실제선택입력을0부터독립재생한qpos/qvel은두해상도모두저장상태와bitwise일치했다. 실제입력과상태로다시계산한일·구속·peak와기존A gate,자세판정도일치했다. [selected_verification.json](results/selected_verification.json)에A/B판정차이와어깨힘벡터증가를포함했다. 원JSON·해시·실패이력은 [summary.json](results/summary.json)의압축파일목록으로연결한다.

[겹친 투구 시뮬레이션](media/budgeted_pitch_overlay.mp4)은원래/수정팔·손·공을같은시각·좌표·축척으로표시한다. 몸통·다리는같은투구의관찰좌표재생이고팔은실제적분이다. 숫자맞춤애니메이션이나실사촬영영상이아니다. 영상안에도A통과의범위,전체순일/어깨힘벡터증가,기준실측차이,실제효과미검증을표시한다. 전체디코드와대표프레임2개를검사했고초안의화면잘림을수정한뒤다시검사했다. 초안은로컬artifacts에보존했다.

새테스트16개와전체suite결과는 [suite_verification.json](results/suite_verification.json)에기록했다. **464개중452개통과,기존환경오류10개(OpenSim9/로컬파일1),건너뜀2개이며새오류/실패0개**다. 기존과학적gate를advance하지않았고자동루틴도재개하지않았다.

## 재현

lab의`.venv/Scripts/python.exe`를사용해이번worktree에서실행한다.

```powershell
python -m unittest coach_hypotheses.budgeted_input_search.test_search coach_hypotheses.budgeted_input_search.test_restore coach_hypotheses.budgeted_input_search.test_render_candidate -v
python -m coach_hypotheses.budgeted_input_search.search --source ../grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz --reference-report ../coordinated-tracking-20261030/artifacts/coordinated-tracking/report.json --out artifacts/budgeted-input-search
python -m coach_hypotheses.budgeted_input_search.restore --source ../grip-identifiability/artifacts/grip-identifiability/fine-reference/reference-0.npz --prior-report artifacts/budgeted-input-search/report.json --out artifacts/budgeted-input-restore-clean
python -m coach_hypotheses.budgeted_input_search.render_candidate --report artifacts/budgeted-input-restore-clean/report.json --state-dir artifacts/budgeted-input-restore-clean --dataset ../openbiomechanics/baseball_pitching/data --out coach_hypotheses/budgeted_input_search/media
```

현재restore.py는중복peak를제거한버전이다. 앞선실행코드는각보고서와함께보존했다. 실제선수의개선효과는미검증이다.

원본C3D·촬영영상은추가하지않았다. 파생결과·영상은CC BY-NC-SA4.0과저장소NOTICE.md의조건을따르며 [Driveline/OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics)를인용한다. 실제선수모집·외부연락·배포·PR병합·훈련/의료조언은이번범위가아니다.
