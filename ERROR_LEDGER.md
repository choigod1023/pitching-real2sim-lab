# 오차 장부

선수2의 같은 세션3034 투구5개. 원자료 식별자와 해시는 각 report.json의 ledger에 보존했다. 레이더 위치·측정 시각과 실제 BR 검출 방식은 미확인이다.

원 C3D360Hz와 처리 CSV의 반올림 시간표를 대조했다. 길이 m, 속도 m/s에서 km/h로 환산, 모멘트 Nm. 손/전완 마커는4차20Hz 양방향 필터, 관절 중심은 공급된 처리 좌표를 그대로 사용한다. 일반 팔/손 관성과145g 공, 손목에서 마커 x축15cm 그립 가정은 유지했다.

자료별 파일 경로와 길이:
- 3034_1: FP10=1.4306, BR=1.6, 관찰 130.195930km/h, 길이=[0.35626962154046415, 0.30600300598665986]; C3D C:\Users\user\Documents\Codex\2026-10-20\new-chat-2\outputs\pitching-lab\openbiomechanics\baseball_pitching\data\c3d\000002\000002_003034_73_207_002_FF_809.c3d
- 3034_2: FP10=1.4, BR=1.5583, 관찰 131.000602km/h, 길이=[0.3637799956152826, 0.2988906430453653]; C3D C:\Users\user\Documents\Codex\2026-10-20\new-chat-2\outputs\pitching-lab\openbiomechanics\baseball_pitching\data\c3d\000002\000002_003034_73_207_005_FF_814.c3d
- 3034_3: FP10=1.3, BR=1.4583, 관찰 129.230323km/h, 길이=[0.3616876863748521, 0.298982871135121]; C3D C:\Users\user\Documents\Codex\2026-10-20\new-chat-2\outputs\pitching-lab\openbiomechanics\baseball_pitching\data\c3d\000002\000002_003034_73_207_006_FF_803.c3d
- 3034_4: FP10=1.2972, BR=1.4528, 관찰 128.425651km/h, 길이=[0.3632259232246646, 0.2999674777640511]; C3D C:\Users\user\Documents\Codex\2026-10-20\new-chat-2\outputs\pitching-lab\openbiomechanics\baseball_pitching\data\c3d\000002\000002_003034_73_207_008_FF_798.c3d
- 3034_5: FP10=1.2333, BR=1.4139, 관찰 128.586586km/h, 길이=[0.3569225554704957, 0.30000204131424324]; C3D C:\Users\user\Documents\Codex\2026-10-20\new-chat-2\outputs\pitching-lab\openbiomechanics\baseball_pitching\data\c3d\000002\000002_003034_73_207_011_FF_799.c3d

주요 미확정: 실제 공 중심·그립·손가락 접촉, 해부학적 손목/회내외 보정, 근육 능력, 레이더와 BR속도의 동일성.

이 장부는 자료 존재와 처리 정합성을 기록하며 미확정 항목을 추정 정답으로 채우지 않는다.
