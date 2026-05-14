## rules

Read docs/TECHSPEC.md, docs/PROJECT_DESIGN.md, and docs/ARCHITECTURE.md before conducting your work.
개발 후 디렉토리 구조가 변경될 시 반드시 docs/ARCHITECTURE.md에 변경 내용 명시

1. 구현 작업 원칙
    - SOLID 원칙 사용
    - 핵심 로직은 TDD로 구현할 것
    - Clean Architecture를 사용해서 구현 : 책임과 관심사를 명확히 분리하여 구현
2. 코드 품질 원칙
    - 단순성 : 언제나 복잡한 솔루션보다 가장 단순한 솔루션을 우선시할 것
    - 중복 방지 : 코드 중복을 피하고, 가능한 기존 기능을 재사용할 것
    - 가드레일 : 테스트 외에는 개발이나 프로덕션 환경에서 모의 데이터를 사용하지 말 것
    - 효율성 : 명확성을 희생하지 않으면서 토큰 사용을 최소화하도록 출력을 최적화할 것
3. 언어
    - 문서와 주석 한국어로 작성
    - 기술적인 용어나 라이브러리 이름 등은 원문 유지
4. 문서화
    - 문서는 코드와 함께 업데이트
    - 복잡한 로직이나 알고리즘은 주석으로 설명할 것
    - docs/TECHSPEC.md에는 사용 언어, 도구, 라이브러리, 모델, 실행 환경 등 기술 스펙만 작성할 것
    - 문제 정의, 데이터 검증, 라벨링, 평가 설계는 docs/PROJECT_DESIGN.md에 작성할 것
    - 일정, 산출물, 리스크, 실행 계획은 docs/PORTFOLIO_EXECUTION_PLAN.md에 작성할 것
