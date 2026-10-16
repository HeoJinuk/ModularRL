# GitHub 공개 전 저장소 감사

감사일: 2026-10-16 (Asia/Seoul). 대상 저장소: [HeoJinuk/ModularRL](https://github.com/HeoJinuk/ModularRL).

**판정: REVIEW REQUIRED.** 개인 Git 이메일의 공개 여부는 사용자가 “공개 전 별도 검토”로 남겼다.
또한 이 Mac의 독립 PyPI 설치에서 NumPy 연동 실패를 재현했다. 사용자는 의존성을 유지하고 다른
컴퓨터에서 확인하기로 했다. 기본 설치의 모든 플랫폼 검증이 통과했다는 뜻은 아니다.
이 감사는 Push, 원격 등록, 이력 재작성, 저자 변경, 원본 삭제를 수행하지 않는다.

## 기준점과 공개 범위

- 시작 Branch: main, HEAD `8a22d68df3c736bd52a107de5ab158d2ee048211`.
- 시작 main 이력: 47개 Commit, 고유 Blob 469개. 최초 Commit `ba79ff960bfb`까지 검사했다.
- 시작 시 Tag와 Remote는 없었다. main 외에는 로컬 `refs/codex/turn-diffs/` 보관 Ref가 있다.
  그 Ref의 main 외 고유 Blob 19개도 별도로 검사했으며 공개 Branch로 취급하지 않았다.
- 시작 시 추적 257개, 미추적 19개, Ignore 354개로 총 630개 파일이었다.
  `.git/` 내부 파일은 이 수에 넣지 않고 Git Object로 별도 검사했다.
- 기존 사용자 변경은 추적 Notebook 15개와 미추적 파일 19개다. Notebook 15개에는 출력/Metadata 외에 코드 셀의 배치 변경도
  포함돼 있어 출력 재생성으로 간주하지 않았다. 전부 읽기 전용으로 검사하고 원본 바이트를 보존한다.
- 이번 Commit에는 감사의 안전한 변경만 명시적으로 넣는다. main의 이미 Commit된 Notebook과
  현재 Working Tree의 사용자 Notebook은 서로 다를 수 있다. 현재 Notebook 실행 결과가 그 사용자
  변경을 자동으로 공개 Commit에 포함시키지는 않는다.

## 파일 인벤토리와 분류

전체 행은 [PRE_PUBLICATION_INVENTORY.csv](PRE_PUBLICATION_INVENTORY.csv)에 있다.
경로·크기·Git 추적/Working Tree 상태·종류·정적 참조·wheel/sdist 포함 여부·A/B/C/D·공개 필요성·권장 조치를 기록한다.
Git 상태는 이번 감사 파일을 Stage하기 전 스냅샷이다. 최종 Commit 후 감사 파일만 tracked로 바뀐다.

|분류|파일 수|처리|
|---|---:|---|
|A 유지|246|실제 구현·교육/실험 자산·설정·최신 문서|
|B 구분해 유지 / 개선 검토|13|역사 문서·독립 회귀 참조·오래된 개발 범위 설명|
|C 로컬 생성 산출물|371|Ignore 유지, 삭제 없음|
|D 사용자 결정|2|개인 경로가 있는 미추적 practice Notebook|
|합계|632|감사 문서/CSV 2개 추가 후 스냅샷|

이 시점의 Git 상태는 tracked 257, untracked 4(감사 파일 2 + 사용자 practice 2), ignored 371이다.

참조 분석은 Python/Notebook 셀의 import, 동적 테스트 모듈의 명시적 이름, 문서 상대 링크와 경로
문자열을 사용한다. pytest·CLI·Workflow 진입점은 별도로 표시한다. 정적 참조가 없다는 이유만으로
파일을 사용하지 않는 것으로 판정하지 않았다. 이미지·JSON은 해당 보고서와 재현 스크립트까지 확인했다.

|영역|판정과 조치|
|---|---|
|modular_rl|추론·학습·Buffer·Target·Update·Planning의 현재 구현 유지. 공개 API나 알고리즘 변경 없음|
|examples / 공식 notebooks|교육 자산 A. 예제 진입점과 Notebook 회귀를 유지하며 출력 일괄 삭제 안 함|
|tests|계약·수치·통합 회귀 A. GitHub에는 유지하고 배포 Artifact에는 제외|
|tests/_legacy_td 및 _legacy_planning.py|승격 전 독립 수학 참조 B. 현재 구현의 재현성 비교에 사용하므로 중복 삭제 안 함|
|docs/history|이전 설계와 결과 B. 현재 API 문서와 역사 기록을 구분한 기존 위치 유지|
|AGENTS.md|B. 일부 범위 문장이 현재 Dyna/Double DQN/A2C 구현보다 오래됨. 덮어쓰지 않았으며 병합 검토 대상으로 남김|
|experiments 및 결과·문서 그림|A. 재현 스크립트·모의 RL 통계·Q 진단·PPO 안정화 결과이며 실측 개인정보 데이터셋으로 확인되지 않음|
|build / dist / cache / work / checkpoint|C. 생성·로컬 QA 자료로 Ignore 유지. 로컬 파일을 삭제하지 않음|
|notebooks/practice 두 파일|D. 사용자 실습 자료이며 개인 경로가 있어 이번 Commit에서 제외. 별도 공개 전 사용자 검토 필요|

가장 큰 추적 파일은 `notebooks/dlfs4/06_td_modularrl.ipynb` 약 2.06 MB다.
약 1.05 MB의 `experiments/results/dyna_dqn_diagnosis_summary.json`과 교육용 그래프도 의도된
결과 자료다. 대용량 데이터셋·모델 가중치·설치에 필요한 미추적 구현 파일은 발견하지 않았다.
유사한 이름의 예제 Wrapper, 공개 Alias, 독립 회귀 참조는 역할이 달라 삭제·이동하지 않았다.

## .gitignore와 실제 정리

- Python·테스트·분석 캐시, Jupyter checkpoint, 로컬 IDE, 빌드·환경·로그·훈련 출력·다운로드 경로를 정리했다.
- `.env` 계열, Private Key와 인증서 묶음·로컬 자격증명을 제외하며 `.env.example` 예외를 유지했다.
- Notebook 전체, JSON/CSV/PNG 전체, 테스트의 작은 Torch Fixture를 일괄 Ignore하지 않는다.
  훈련 출력은 루트의 명시적 디렉터리 규칙을 사용한다.
- 기존 미추적 19개 중 Jupyter checkpoint 17개는 새 규칙으로 ignored가 됐다. 파일은 모두 남아 있다.
  나머지 사용자 practice Notebook 2개는 계속 untracked다.
- `git ls-files -ci --exclude-standard` 결과는 0개다. 과거 main의 모든 파일 경로를 현재 규칙에
  대조해도 해당 규칙으로 숨겨진 추적 이력은 없었다. 추적 해제가 필요한 생성 파일은 없었다.
- 합성 경로로 checkpoint·캐시·루트 로그/훈련 출력의 제외와 공식 결과 JSON·문서 PNG·실제 Notebook·
  테스트 Fixture의 보존을 확인했다. `.gitignore`가 기존 Git 이력을 지운다고 주장하지 않는다.
- Repository URL을 패키지 Metadata에 추가하고 설치 Smoke Test에 NumPy 입력과 Tensor 행동의
  NumPy 변환을 보강했다. README 19줄과 MIT 표준 LICENSE, Copyright (c) 2026 Jinuk Huh는 유지했다.
- 오래된 공개 준비 문서의 저장소 생성 여부와 URL 설명을 바로잡고 이 감사로 연결했다.

## 비밀정보·개인정보 및 Git 이력

현재 파일의 원시 바이트와 main에서 도달 가능한 모든 Commit/Blob을 검사했다. 로컬 보관 Ref의
추가 Blob도 분리했다. Token·Private Key·인증 Header·DB 자격증명 URL·사설 주소·홈 경로·이메일
패턴 검사에 더해 파일명과 설정·소스·Notebook 텍스트를 검토했다.

임시 도구 경로에 설치한 detect-secrets 1.5.0의 27개 탐지기를 `--all-files --no-verify`로 실행했다.
현재/역사 UTF-8 파일 사본과 Commit Metadata/메시지를 대상으로 비밀정보 후보는 0개였으며,
합성 Password 양성 대조는
KeywordDetector가 탐지했다. 의심 자격증명의 외부 유효성 확인은 수행하지 않았다. Notebook은
검사용 사본에서 이미지 MIME payload만 제외하고 소스·텍스트 출력·Metadata를 모두 검사했다. Cookie·Password 할당·인증/Token 이름도 별도 검색했으며
확인된 후보는 테스트 이름의 environment/monkeypatch 같은 일반 문자열이었다.
원본 Notebook을 정규화하거나 출력 삭제하지 않았다. 개인정보 패턴은 별도로 검사했다.

|후보|위치·증거|main 최초 공개에 포함되는가|조치|
|---|---|---|---|
|개인 author/committer 이메일|시작 main 47개 Commit 모두 동일한 `b***@g***.com`; GitHub noreply 아님|예. 파일 Blob에 없어도 Commit Metadata로 노출|사용자 “공개 전 별도 검토”. 기존 이력·Git 저자 설정 유지|
|개인 홈 경로|practice/ModularRL_API_HandsOn.ipynb JSON 42행, practice/ModularRL_Practice_Workbook.ipynb 37–38행|아니오. 현재 미추적이며 main 이력에서 도달 불가|읽기 전용 보존, 이번 Stage 제외|
|같은 개인 경로의 로컬 보관본|Blob `99830212c6de722e8d1b3e91b4af3458014fe8c0`, `254bb318df5f733a27b7f6b5bda4d0fc5a8827f9`|아니오. `refs/codex/turn-diffs/` 전용|로컬 Ref 유지. 나중에 별도 공개할 때 다시 검사|
|로컬 QA의 개인 이메일 / bytecode의 홈 경로|ignored work의 QA JSON 6개에 이메일 292회, 주로 pyc에 홈 경로 199회|아니오. main 이력에 해당 파일 없음|로컬 보존·Ignore 유지. raw 진단은 저장소 밖에 둠|

개인 이메일이 있는 최초 main Commit은 `ba79ff960bfb`, 감사 시작의 마지막은 `8a22d68df3c7`다.
새 감사 Commit도 기존 Git Identity를 사용하므로 공개 전 검토 범위에 포함한다.
author/committer 표시 이름은 한 종류이며 LICENSE의 영문 이름과 표기가 다르다. 이를 잘못된 저자로
단정하거나 변경하지 않았고, 공개 전 Git Identity 검토에 함께 포함한다.
보고서에는 개인 경로·이메일 원문·원시 QA 파일을 복사하지 않는다. 로컬 이메일 후보 파일은
`work/action-selector-validation-before.json`, `work/estimator_cleanup/before.json`,
`work/estimator_stabilization/before.json`, `work/experience_replay/before.json`,
`work/model_integration/before.json`, `work/network_injection/before.json`이다.

현재 추적 파일과 main Blob에서 위 패턴의 자격증명·개인 경로·개인 이메일 후보는 없었다.
Commit Metadata의 개인 이메일은 별개다. 폐기·재발급해야 할 실제 Credential은 확인하지 못했다.
이메일 비공개를 원하면 승인된 별도 작업에서 공개할 Ref 범위를 정하고 Commit Metadata 정리 후
전체 재감사해야 한다. Credential이 추가 발견되면 먼저 폐기·재발급하고 이력 정리를 별도 승인받아야 한다.
이번에는 이력 재작성, Ref 삭제, `--mirror`나 `--all` Push를 실행하지 않았다.

## 코드와 CI 보안 검토

Bandit 1.9.4는 modular_rl/examples/experiments/tests를 검사해 High 0, Medium 3, Low 1,718개를
보고했다. Low 대부분은 pytest assert(B101 1,676개)와 고정 argv subprocess다. 분석 오류는 0개,
`modular_rl/`의 Bandit 결과는 0개였다. Medium의 실제 맥락은 다음과 같다.

- `tests/test_dyna_q_composition.py:309`, `tests/test_gridworld_dyna_dqn.py:189`의
  `torch.load(weights_only=False)`는 같은 테스트가 tmp에 직접 저장한 RNG/경험 Checkpoint다.
  이미 trusted local 주석이 있으며 외부 파일을 받는 공개 로더가 아니다.
- `tests/test_public_api.py:77`의 `pickle.loads`는 직전에 같은 메모리에서 만든 Model을
  `pickle.dumps`한 왕복 검증이다. 신뢰 경계가 다른 임의 파일 로드로 일반화하지 않는다.

Python과 Notebook 코드 셀 AST/문자열 검토에서 builtin eval/exec, shell=True, os.system/popen,
SSL 검증 해제와 외부 데이터 기반 명령 실행은 발견하지 못했다. nn.Module.eval()은 동적 실행과
구분했다. subprocess는 고정 argv와 로컬 테스트·패키지 도구 실행이다. torch/pickle 역직렬화
20개 호출의 출처를 확인했으며 17개 torch 호출은 weights_only=True다.
설치는 선언적 setuptools Backend를 사용하며 별도 setup.py/hook의 임의 명령이나 소스의 불필요한
네트워크 접근은 발견하지 못했다. import/소스 검사에 설치 후 저장소 밖 실행을 함께 사용했다.

두 Workflow는 contents:read, checkout persist-credentials:false이며 비밀값 출력·쓰기/배포 권한·
continue-on-error나 실패 무시 구문이 없다. Push/PR의 짧은 CPU 검사와 수동 전체 Notebook 검사를
분리하고 Python 3.12, pyproject 기반 pip cache를 사용한다. 고정 CPU PyTorch 인덱스도 확인했다.
현재 공식 checkout/setup-python의 v7 사용은 [각 공식 저장소](https://github.com/actions/checkout),
[setup-python 문서](https://github.com/actions/setup-python)와 대조했다. Major Tag는 변경 가능하며
Commit SHA 고정은 향후 강화 선택지다. 이번에 불필요한 Workflow 변경은 하지 않았다.
YAML 구조와 로컬 명령을 검증하지만 실제 GitHub 호스팅 CI는 첫 Push 후 확인해야 한다.

## Notebook·실험 자료

추적 Notebook 27개, 미추적 practice 2개, ignored checkpoint 17개로 JSON 46개를 검사했다.
모든 JSON을 읽을 수 있었고 추적 Notebook의 저장된 Traceback 출력은 0개였다. 텍스트 출력과
Metadata에서도 위 자격증명/개인 경로 후보가 없었다. 그래프와 현재 사용자 출력은 그대로 유지한다. Notebook PNG 82개와 문서 그림 5개를
모아보기 이미지로 시각 검토했으며 학습 곡선·가치표·정책/환경 도식이었다. Notebook PNG의 부가정보는
Software와 dpi 필드뿐이었다. 축소 시각 검토를 이미지 OCR이나 숨은 데이터 검사로 간주하지 않는다.
PPO/Dyna 등 정리된 실험 결과는 학습 설정·Seed·합성 환경 통계로 검토했고 개인정보 원본 데이터나
내부 운영 로그로 판단할 근거가 없었다. 그래프의 교육적 역할을 확인했다.

실행 회귀는 전달된 Notebook의 사본을 별도 커널에서 두 번 실행해 출력·수치·그림을 비교한다.
Gymnasium 22개는 FAST 예산이고 기존 DLFS4 5개는 FAST 분기가 없는 원래 예산을 사용한다.
원본 파일을 덮어쓰지 않는다. CUDA는 장치가 없어 실행하지 않는다.
추적 Notebook 중 HEAD와 소스 배치가 다른 파일은 gymnasium/00_api_basics 한 개다.
이 파일은 Commit된 HEAD 버전도 별도 사본으로 두 번 실행해 출력과 전체 정밀도 수치가 같음을 확인했다.
다른 추적 Notebook의 소스 셀은 HEAD와 동일하다.

## 패키징과 독립 설치

- Python 3.12.15, Conda PyTorch 2.13.0, NumPy 2.5.3, Gymnasium 1.4.0, macOS Intel/CPU에서 검증한다.
  requires-python>=3.10은 유지하지만 다른 Python/플랫폼 전체를 검증했다고 주장하지 않는다.
- 필수 NumPy/PyTorch/Gymnasium은 실제 코드에서 사용한다. Matplotlib/Jupyter/pytest 등은 Extra다.
  Distribution `modularrl-model-contract-demo==0.1.0`, import `modular_rl`과 기존 공개 Alias를 유지한다.
- wheel 103개 Record, sdist 109개 파일이며 라이브러리 Python 소스 98개가 서로 일치한다.
  표준 LICENSE·MIT Metadata·README·Repository URL을 확인했다. tests/examples/notebooks/experiments/
  docs/cache/로컬 설정은 배포에 포함되지 않는다. wheel에 README 파일이 없더라도 설명 Metadata로
  포함되는 점과 물리 파일 인벤토리를 구분한다.
- system_site_packages=True인 Artifact별 별도 venv에서는 설치 경로가 venv 아래인지,
  examples 없이 공개 알고리즘이 실행되는지, NumPy 관측 입력과 행동 변환이 가능한지 확인한다.
- 별도로 system_site_packages=False의 새 venv 두 개에서 PyPI 기본 의존성을 실제 해결했다.
  **wheel과 sdist 모두 설치/pip check는 통과했지만 NumPy 연동 Smoke는 실패했다.**
  이 Mac에서 선택된 조합은 PyTorch 2.2.2 / NumPy 2.5.4다. 오류는
  `Could not infer dtype of numpy.float32`와 NumPy ABI 초기화 실패다.
- 원인 분리용 세 번째 임시 venv에만 NumPy 1.26.4를 지정했을 때 같은 wheel의 NumPy 입력 및
  알고리즘 Smoke가 통과했다. 이는 기본 설치 통과나 프로젝트의 NumPy 상한 적용을 뜻하지 않는다.
  [NumPy의 공식 ABI 설명](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)과
  재현 결과를 함께 확인했다.
- 사용자는 이 Mac에 맞춘 numpy<2 Metadata 변경을 원하지 않았다. runtime 의존성과 사용자 Conda
  환경은 유지한다. 다른 컴퓨터의 새 의존성 설치 결과와 GitHub Linux CI는 미확인이다.
  기존 Tensor 중심 설치 Smoke가 놓친 문제를 앞으로 발견하도록 NumPy 입출력 검증만 보강했다.

## 실제 검증 결과

|검사|결과|
|---|---|
|전체 pytest / Notebook 27개 × Run All 2회|2459 passed / 229 skipped, 3627.85초(1:00:27); 전부 통과|
|기본 CPU CI 명령(Notebook 두 모듈 제외)|2432 passed / 229 skipped, 172.56초|
|Commit된 HEAD의 API Notebook 별도 검사|Run All 2회; 출력·수치 동일|
|기존 CLI 예제 Smoke|18개 통과: quickstart, DP, FrozenLake, CartPole, MC/TD/Control, n-step, Q(sigma), Dyna|
|Build / Twine|wheel/sdist 빌드 성공, 둘 모두 Twine PASSED|
|Artifact 파일 목록·소스·MIT·Metadata|일치 및 배포 제외 경계 확인|
|기존 Conda 의존성 공유 venv wheel/sdist|둘 모두 통과; NumPy 입력/출력 포함|
|새 의존성 venv wheel/sdist|install/pip check 통과, NumPy 연동 둘 모두 실패(위 제한)|
|NumPy 1.26.4 임시 원인 분리|동일 wheel NumPy/알고리즘 Smoke 통과; 기본 의존성 변경 없음|
|보안 Scanner / 패턴 / AST|위 범위 검사 완료; 개인 Git 이메일은 검토 미해결|
|Workflow YAML / 호스팅 CI|로컬 구조 검증; 호스팅 실행은 Push 이후|
|기존 사용자 파일 SHA-256 보존|기존 15개 수정 Notebook 및 미추적 19개 모두 동일; 추적 Notebook 전체도 바이트 동일|
|README / LICENSE|19줄·코드 예제 실행 통과 / MIT 표준 원문·2026 Jinuk Huh 유지|

## 공개 전에 남은 일과 감사의 한계

1. 사용자가 개인 Git 이메일을 기존 이력에 공개할지 별도 검토한다. 아직 공개 승인이 아니다.
2. 사용자가 다른 컴퓨터에서 기본 의존성의 독립 설치를 확인한다. 알려진 Mac Intel 실패는 위에 남긴다.
3. 사용자 수정 Notebook을 나중에 Commit할지는 별도로 결정한다. practice 두 파일과 로컬 Ref는
   현재 main에 포함되지 않으며, 향후 공개하려면 개인 경로를 먼저 검토한다.
4. 사용자가 Push를 결정한 뒤 첫 GitHub CI와 실제 Repository 접근·설정 상태를 확인한다.

이는 전체 소프트웨어 보안 인증이나 모든 의존성 CVE·저작권·모든 플랫폼의 설치 보장이 아니다.
이미지 OCR, 비텍스트 바이너리의 숨은 정보, 저장소 밖 비공개 서비스까지 검사한 것은 아니다.
암호화되거나 알려지지 않은 비밀정보는 패턴/엔트로피 검사로 놓칠 수 있다. 도구 설명은
[detect-secrets 공식 문서](https://github.com/Yelp/detect-secrets)를 참고했다.
의도된 교재 출처 링크와 기존 결과를 보존했으며 MIT는 제3자 저작물·의존성의 권리를 변경하지 않는다.

최종 Commit 후 실제 main 도달 범위를 다시 검사한다. 안전한 감사 파일만 Commit하고 Push하지 않는다.
