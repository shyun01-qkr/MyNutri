# MyNutri

개인 건강정보와 생활습관 정보를 바탕으로 부족 가능 영양소를 분석하고, 건강 목표에 맞는 건강기능식품을 추천하는 Flask 기반 웹 프로젝트입니다.

> **주의**: 본 프로젝트의 분석 및 추천 결과는 학습·프로젝트 목적의 참고 정보이며, 질병의 진단이나 치료를 위한 의료적 판단을 제공하지 않습니다.

## 주요 기능

- 건강정보 및 생활습관 입력
- 규칙 기반 부족 가능 영양소 분석
- Random Forest 기반 부족 가능 영양소 예측
- 건강 목표, 부족 가능 영양소, 안전성, 생활습관 적합도 등을 반영한 제품 점수 계산
- 건강 목표별 추천 및 최대 3개 제품의 최종 조합 추천
- Flask 웹 화면과 분석 API 제공

## 기술 스택

- Python
- Flask
- pandas / NumPy
- scikit-learn
- joblib
- Matplotlib

## 프로젝트 구조

```text
MyNutri/
├── app.py
├── train_models.py
├── train_random_forest.py
├── requirements.txt
├── config/
│   └── health_goal_mapping.json
├── templates/
│   └── index.html
├── models/
│   └── .gitkeep
├── 01_product_items_100_clean.csv
├── 02_functional_ingredients_100_clean.csv
├── 03_nutrition_facts_100_clean.csv
├── 04_label_information_100_clean.csv
├── SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER.csv
└── nutrition_model_report.txt
```

## 실행 방법

### 1. 가상환경 생성 및 패키지 설치

Windows 기준:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 모델 생성

대용량 학습 모델(`*.pkl`)은 GitHub 저장소에 포함하지 않습니다. 아래 스크립트를 실행하면 필요한 모델 파일이 생성됩니다.

```bash
python train_models.py
python train_random_forest.py
```

`train_models.py`는 `models/` 아래에 예측 모델과 인코더 정보를 생성하고, `train_random_forest.py`는 `nutrition_random_forest.pkl` 및 평가 결과를 생성합니다.

### 3. Flask 앱 실행

```bash
python app.py
```

실행 후 브라우저에서 `http://127.0.0.1:5000`으로 접속합니다.

## 데이터 및 모델 관련 안내

- 프로젝트에는 합성 PHR 데이터와 정리된 건강기능식품 데이터가 포함되어 있습니다.
- 부족 가능 영양소 예측용 학습 라벨은 프로젝트에서 정의한 규칙을 바탕으로 생성된 라벨입니다.
- 따라서 `nutrition_model_report.txt`의 모델 평가 수치는 실제 임상 데이터에서의 진단 성능을 의미하지 않습니다.
- 학습된 `.pkl` 모델 파일은 파일 크기와 저장소 관리 편의를 위해 제외하고, 재학습 가능한 코드를 제공합니다.

## 모델 평가 결과

현재 저장된 `nutrition_model_report.txt` 기준:

- 데이터 수: 12,650
- Feature 수: 40
- Label 수: 17
- Macro F1: 0.9368
- Micro F1: 0.9984

위 결과는 프로젝트에서 생성한 합성 데이터 및 규칙 기반 라벨을 대상으로 한 평가 결과입니다.
