# 단계1 구현 전 원문 확인

확인일:2026-10-29(KST). 논문 결과를 자세 처방이나 개입 효과로 해석하지 않습니다.
전체 논문을 확인한 경우와 저자 초록만 확인한 경우를 구분합니다. 접근 실패 응답은
로컬 artifacts/sources/access_manifest.json에 보존하고 논문 본문으로 취급하지 않습니다.

1. Scarborough 등(2020), DOI10.1080/14763141.2018.1503321.
   [저자 초록/PubMed](https://pubmed.ncbi.nlm.nih.gov/30213227/)에서22명208구와
   시퀀스 변동 연구임을 확인했습니다. 출판사 본문403으로 **전체 원문 확인 필요**입니다.
   이 연구가 큰 골반–몸통 지연을 보편적 최적값으로 입증했다고 쓰지 않습니다.
2. Liu 등 ISBS2022, [저자 학회 페이지](https://commons.nmu.edu/isbs/vol40/iss1/92/).
   초록은4명, 축/벡터 크기에 따른 순서 차이와 필터의 예비적 영향을 설명합니다.
   PDF403으로 **전체 원문 확인 필요**입니다. 본 분석의 제공z축 정의를 이 논문의
   특정 글로벌 축 분석과 동일시하지 않습니다.
3. Dowling 등 OJSM2024, DOI10.1177/23259671241257539.
   [전체 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC11329978/)과 EuropePMC XML을
   확인했습니다. 프로50명/고교50명의 후향적 관찰입니다. 고신전 프로 집단의
   팔꿈치 varus 토크는 저신전 집단보다 낮았지만 고교 집단은 반대였습니다.
   따라서 신전이 큰 모든 투수의 토크가 높았다는 요약은 정확하지 않습니다.
4. Matsuda 등 Frontiers2025, DOI10.3389/fspor.2025.1534596.
   [전체 논문](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1534596/full)을
   확인했습니다. 대학 남자20명, 정상 보폭 대비±20% 조건입니다. 긴 보폭과
   짧은 보폭 간 구속 차이는 없었지만 정상 보폭은 두 변형 조건보다 높았습니다.
   모든 조건 간 구속 차이가 없었다는 요약은 수정해야 합니다. 몸통 관절로 나가는
   총 에너지의 비유의 차이와 개별 단계의 변화를 구분합니다.
5. Jump 등 SEACSM2026 P129,
   [학회 초록 원문PDF,46번째PDF페이지](https://southeast.acsm.org/wp-content/uploads/2026/02/SEACSM-2026-program-ABSTRACTS-2-21-26.pdf).
   표본은335구가 아니라 **335명 투수**입니다. 선수 간/선수 내 관계와 개인별
   방향·크기 차이를 다룹니다. **학회 초록이며 전체 논문 확인 필요**입니다.
6. [TACSM2026 초록집](https://texas.acsm.org/wp-content/uploads/2026/02/2026-TACSM-Research-Abstract-Booklet-1.pdf)의
   OBP 기반 분석은 착지 분리와 최대 분리의 부하 관계 방향이 달랐다고 보고합니다.
   **학회 초록이며 전체 논문 확인 필요**입니다. 이를 근거로 안전 한계를 만들지 않습니다.
7. Slowik 등 JAT2019, DOI10.4085/1062-6050-558-17,
   [전체 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC6485842/).
   웹 본문을 확인했습니다(직접 다운로드403/EuropePMC500과 구분).
   64명의 선택된 프로 투수에서 선수 간 관계와 random-intercept 혼합모형의
   선수 내 관계를 구분했습니다. 해당 R²를 우리 중심화 상관계수와 같은 통계량으로
   취급하지 않습니다. 순 관절 토크는 UCL 조직 하중이나 실제 부상률이 아닙니다.
8. 릴리스 일관성은 위치/각도/시점/도착점의 서로 다른 질문입니다.
   [Whiteside 등 release variability 연구](https://onlinelibrary.wiley.com/doi/10.1111/sms.12443),
   [운동학 변동과 도착점 연구](https://pubmed.ncbi.nlm.nih.gov/31449438/)를 찾았지만
   이번에는 **전체 원문 확인 필요**로 남깁니다. OBP에 의도 목표/도착점 및 직접 공 중심이
   없으므로 타이밍 일관성이 제구를 설명한다는 검증을 수행하지 않습니다.

OBP 정의는 [baseball_pitching README](https://github.com/drivelineresearch/openbiomechanics/tree/main/baseball_pitching)와
로컬 data_dictionary.csv, DATASHEET.md를 대조했습니다. full_sig는 이미20Hz
4차 Butterworth 처리가 된 자료입니다. 추가10/20/30Hz 처리를 원시 신호의
차단주파수 재분석으로 부르지 않습니다. 자료의 구속을 이용해 이벤트/축/필터를 선택하지 않습니다.
