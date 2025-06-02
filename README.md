---
title: "Jupyter Notebook 사용법"
tags: [jupyter, python, 데이터분석]
date: 2025-06-02
author: "easywoo8080@github.com"
---

# 2. 주피터 노트북

## 2-1. 개요
Jupyter Notebook은 코드, 설명, 그래프, 수식을 한 문서에서 통합 관리할 수 있는 웹 기반 인터랙티브 개발 환경입니다. 특히 데이터 분석, 머신러닝, 과학 계산, 문서화에 널리 사용됩니다.

## 2-2. 핵심 특징

1. 인터랙티브 실행
	1. 셀 단위로 코드를 실행하고 즉시 결과 확인
2. 마크다운 지원
	1. 설명, 수식(LaTeX), 제목 등을 마크다운 문법으로 작성 가능
3. 시각화 통합
	1. `matplotlib`, `seaborn` 등의 그래프 결과를 셀 아래에 바로 표시
4. 다양한 커널
	1. 기본은 Python 이지만,  R, Julia 등 다른 언어 커널도 지원
5. 파일 확장자
	1. `.ipynb` (JSON 기반 구조로 셀 단위 구성)
6. 노트 형식 문서화
	1. 분석 보고서나 실험 로그를 한 문서에 기록 가능


- **Google Colab**: 클라우드 기반 Jupyter 환경, 무료 GPU 지원
- **VSCode Jupyter 확장**: VSCode 내에서 `.ipynb` 실행 가능
---
# 3. 설치

## 3-1. 최초 설치

1. 가상환경 설정
2. 가상환경 접속
3. uv init
4. uv add로 주피터 노트북 설치와 패키지 관리

```powershell
python -m venv .venv
.venv\Scripts\activate
uv init
uv add --dev ipykernel notebook
```

## 3-2. 새환경 및 재설치

1. `uv run`으로 가상환경부터 패키지가지 전부 설치
2. 가상환경 접속
```powershell
un run
.venv\Scripts\activate
```


## 3-3. 주피터 노트북 실행
1. `jupyter notebook`실행하면 브라우저에 주피터 노트북이 실행
```powershell
jupyter notebook
```

---
# 4. 주피터 사용

## 4-1. 주피터 실행

```powershell
jupyter notebook
```

브라우저가 출력되면 디렉토리 목록이 나옴

## 4-2. 디렉토리 관리와 사용 규칙

### 4-2-1. 트리 구조

```text
root/
├── jupyter/
│   └── your_modules_1/
│	│	└── option.ipynb
│   └── your_modules_2/
│	│	└── option_2.ipynb
│	└── ipynb2py.py
└── src/

```

### 4-2-2. 작업 영역
최소한의 폴더 구조는 root폴더와 그 하위에 jupyter폴더 그 안에 ipynb2py.py이다.

> [!info]- ipynb2py.py의 기능
> ipynb2py.py의 기능은 해당 파일의 현재 디렉토리 부터 그 하위 디렉토리에 있는 모든 ipynb확장자 파일들을 jupyter폴더와 같은 위치에 src라는 폴더를 생성하고 그 안에 디렉토리 구조를 복사 ipynb파이을 py로 변환하여 저장한다.

1. jupyter폴더 안에는 ipynb2py.py 파일만 존재한다.
2. jupyter폴더 하위 폴더를 생성하여 ipynb파일 작업을 한다.
3. 작업이 모두 끝나면 jupyter/inpynb2py.py를 실행해서 py로 변환한다.
4. 실제 사용 코드는 src에 있는 코드를 사용한다.

### 4-2-3. 컨버팅 코드
```powershell
python .\jupyter\ipynb2py.py
```

