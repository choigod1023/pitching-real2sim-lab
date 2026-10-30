# MLB 공식 투구 JSON 표본 확보·스키마 감사 사전 기준

2026-10-30. parser구현/저장/관계분석전에커밋한다. 병렬읽기전용접근확인에서공식schedule과그응답의feed링크를실제로확인했다. Mendeley/Statcast CSV403실패는그대로보존한다. 이번대안은공식JSON이며원래CSV를확보했다고하지않는다.

## 고정 접근·선택

https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2025-06-01 의완료된정규시즌경기중gamePk숫자오름차순첫경기를선택한다. 성적/구위값으로경기를고르지않는다. 접근감사에서선택된경기는777677(DET@KC)이며schedule응답의공식link=/api/v1.1/game/777677/feed/live를사용한다. sourceURL=https://statsapi.mlb.com/api/v1.1/game/777677/feed/live, 초기확인응답725682bytes/SHA2567f40e379d47c491aa1318190a030fe6ba5ff9f609dd9f42ebc2c852a692d0fc7이다. 동적피드라응답이달라지면현재해시와차이를기록하며이전해시를가짜로재사용하지않는다. 공식pitchCodes목록도sourceURL/해시를보존한다.

isPitch=true인모든playEvent를추출하고(gamePk,atBatIndex,pitchIndex/playId)의중복/신원/필드유형·비유한·결측을감사한다. 초기읽기전용감사는69allPlays/256투구/8투수,boxscore투구합256과일치했다. 이것은사전품질확인이지구위관련성분석이아니다.

## 값/신원/시점 정의

startSpeed/endSpeed,spinRate/spinDirection,breakHorizontal/Vertical/VerticalInduced,pfxX/Z,pX/Z,x0/y0/z0,vX0/vY0/vZ0,aX/aY/aZ를원필드명과원값으로보존한다. JSON자체에는단위가없다. 공식metrics목록의유사한이름과같아보인다는것만으로별칭의단위를확정하지않고,각필드의단위/좌표/시점근거를확인/가정/UNKNOWN으로분리한다. 기본적으로단위변환·ballflight계산을하지않는다. spinDirection은3D회전벡터가아니고arm_angle/손가락/심/의도목표가없으면None/미관측으로남긴다. 데이터범위만으로단위를맞추지않는다.

allPlay.matchup의pitcher/batter가AB도중교체를정확히대표하는지검사한다. 첫pitch후투수/타자교체가발견되면그AB의연결을uncertain으로격리·기록하고임의신원배정을하지않는다. 첫pitch이전교체와구분한다. isPitch=false의자동볼/스트라이크같은count업데이트도순서대로반영한다. event.count는사후값이므로이전이벤트의사후count로사전balls/strikes를복원하고AB시작0/0과이상범위를검사한다.

공식pitchCodes의swingStatus/swingMissStatus/contactStatus/buntAttemptStatus를원문bool로별도보존한다. T=foul tip에swingMissStatus=true인경우가있어이를연구용whiff라고즉시rename하지않는다. 명시적별도정의없이헛스윙률/성적연관모델을만들지않는다.

## 완료·범위

rawJSON·투구별CSV는자기전용worktree artifacts에만저장하고Git에올리지않는다. 코드·테스트·aggregate audit/manifest/접근실패·한국어보고서만커밋한다. 공개접근은이자료가CC BY나OBP라이선스라는뜻이아니며재배포허용을추정하지않는다. 공식출처/접근일/해시를보존하고원문재배포를피한다.

parser의필드/숫자/중복/신원/사전사후count/교체격리/단위미확인/공식코드bool/boxscore검산/JSON응답검사/캐시테스트를작성한다. 실제256투구의coverage와누락을보고하고fixture는연구데이터로세지않는다. technical_access_ready와conditional_study_ready,body_to_spin_ready/3Dspin/근력·일·개입효과는구분한다. 이번은첫공식실자료확보/감사이며한경기만으로선수처방이나구위향상효과를주장하지않는다. 관계분석은이후표본/정의/평가기준을별도고정한뒤진행한다. 외부연락/유료/선수모집/접근우회/배포/PR병합/자동화재개는하지않는다.
