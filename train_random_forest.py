"""
Random Forest 기반 PHR 부족 가능 영양소/기능성 원료 예측 파이프라인

주의: 본 모델은 질병 진단, 치료, 예방 목적이 아닙니다.
건강기능식품 추천을 위한 참고용 '부족 가능성/보충 고려' 예측 모델입니다.

실행:
    python train_random_forest.py

입력:
    SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER.csv
    또는 SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER(1).csv

출력:
    SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER_labeled.csv
    nutrition_random_forest.pkl
    nutrition_model_report.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==============================
# 파일 경로 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parent
CANDIDATE_INPUTS = [
    BASE_DIR / "SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER.csv",
    BASE_DIR / "SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER(1).csv",
]
LABELED_CSV = BASE_DIR / "SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER_labeled.csv"
MODEL_PATH = BASE_DIR / "nutrition_random_forest.pkl"
REPORT_PATH = BASE_DIR / "nutrition_model_report.txt"

LABEL_COLUMNS = [
    "vitamin_d_risk",
    "vitamin_c_risk",
    "vitamin_b_risk",
    "magnesium_risk",
    "calcium_risk",
    "zinc_risk",
    "iron_risk",
    "folate_risk",
    "omega3_need",
    "probiotics_need",
    "lutein_need",
    "red_ginseng_need",
    "milk_thistle_need",
    "coq10_need",
    "policosanol_need",
    "garcinia_need",
    "saw_palmetto_need",
]

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "waist",
    "sbp",
    "dbp",
    "glucose",
    "hdl",
    "ldl",
    "triglyceride",
    "sleep_hours",
    "heart_age",
    "lifestyle_risk_score",
    "cardio_risk_score",
    "metabolic_syndrome_count",
    "exercise_score",
    "sleep_score",
    "stress_score",
    "smoking_score",
    "drinking_score",
    "cardio_risk_encoded",
    "ascvd_risk_encoded",
    "medication_flag",
    "takes_antihypertensive",
    "takes_diabetes_med",
    "takes_lipid_med",
]

CATEGORICAL_FEATURES = [
    "sex",
    "smoking",
    "drinking",
    "exercise",
    "stress_level",
    "subjective_health",
    "hypertension",
    "diabetes",
    "dyslipidemia",
    "metabolic_syndrome",
    "family_history",
    "medication",
    "cardio_risk",
    "menopause",
    "ascvd_risk_level",
]

# 모델 출력용 한글 이름
LABEL_DISPLAY_NAMES = {
    "vitamin_d_risk": "Vitamin D 부족 가능성",
    "vitamin_c_risk": "Vitamin C 부족 가능성",
    "vitamin_b_risk": "Vitamin B Complex 부족 가능성",
    "magnesium_risk": "Magnesium 부족 가능성",
    "calcium_risk": "Calcium 부족 가능성",
    "zinc_risk": "Zinc 부족 가능성",
    "iron_risk": "Iron 부족 가능성",
    "folate_risk": "Folate 부족 가능성",
    "omega3_need": "EPA/DHA(Omega3) 보충 고려",
    "probiotics_need": "Probiotics 보충 고려",
    "lutein_need": "Lutein 보충 고려",
    "red_ginseng_need": "Red Ginseng 보충 고려",
    "milk_thistle_need": "Milk Thistle 보충 고려",
    "coq10_need": "Coenzyme Q10 보충 고려",
    "policosanol_need": "Policosanol 보충 고려",
    "garcinia_need": "Garcinia Cambogia 보충 고려",
    "saw_palmetto_need": "Saw Palmetto 보충 고려",
}


def find_input_file() -> Path:
    for path in CANDIDATE_INPUTS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "PHR CSV 파일을 찾지 못했습니다. train_random_forest.py와 같은 폴더에 "
        "SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER.csv 파일을 넣어주세요."
    )


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_phr_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 첫 번째 행이 '만나이', '성별' 같은 한글 설명행이면 제거
    df["AGE_NUM_CHECK"] = pd.to_numeric(df["AGE"], errors="coerce")
    df = df[df["AGE_NUM_CHECK"].notna()].drop(columns=["AGE_NUM_CHECK"])
    df = df.reset_index(drop=True)
    return df


def risk_level_to_num(value: Any) -> int:
    text = str(value).upper()
    if "HIGH" in text or "높" in text:
        return 2
    if "MED" in text or "중" in text:
        return 1
    return 0


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """원본 PHR 컬럼을 Random Forest 입력용 Feature로 변환한다."""
    out = pd.DataFrame(index=df.index)

    # 원본 컬럼 표준화
    out["age"] = to_number(df.get("AGE", pd.Series(index=df.index)))
    out["sex"] = df.get("SEX", "unknown").astype(str)
    out["smoking"] = df.get("SMOKING", "unknown").astype(str)
    out["drinking"] = df.get("DRINKING", "unknown").astype(str)
    out["exercise"] = df.get("EXERCISE", "unknown").astype(str)
    out["sleep_hours"] = to_number(df.get("SLEEP_HOURS", pd.Series(index=df.index)))
    out["stress_level"] = df.get("STRESS_LEVEL", "unknown").astype(str)
    out["subjective_health"] = df.get("SUBJECTIVE_HEALTH", "unknown").astype(str)
    out["lifestyle_risk_score"] = to_number(df.get("LIFESTYLE_RISK_SCORE", pd.Series(index=df.index)))
    out["bmi"] = to_number(df.get("BMI", pd.Series(index=df.index)))
    out["waist"] = to_number(df.get("WAIST", pd.Series(index=df.index)))
    out["sbp"] = to_number(df.get("SBP", pd.Series(index=df.index)))
    out["dbp"] = to_number(df.get("DBP", pd.Series(index=df.index)))
    out["glucose"] = to_number(df.get("GLUCOSE", pd.Series(index=df.index)))
    out["hdl"] = to_number(df.get("HDL", pd.Series(index=df.index)))
    out["ldl"] = to_number(df.get("LDL", pd.Series(index=df.index)))
    out["triglyceride"] = to_number(df.get("TRIGLYCERIDE", pd.Series(index=df.index)))
    out["hypertension"] = df.get("HYPERTENSION", "unknown").astype(str)
    out["diabetes"] = df.get("DIABETES", "unknown").astype(str)
    out["dyslipidemia"] = df.get("DYSLIPIDEMIA", "unknown").astype(str)
    out["metabolic_syndrome_count"] = to_number(df.get("METABOLIC_SYNDROME_COUNT", pd.Series(index=df.index)))
    out["metabolic_syndrome"] = df.get("METABOLIC_SYNDROME", "unknown").astype(str)
    out["family_history"] = df.get("FAMILY_HISTORY", "unknown").astype(str)
    out["medication"] = df.get("MEDICATION", "없음").astype(str)
    out["takes_antihypertensive"] = to_number(df.get("TAKES_ANTIHYPERTENSIVE", pd.Series(index=df.index))).fillna(0)
    out["takes_diabetes_med"] = to_number(df.get("TAKES_DIABETES_MED", pd.Series(index=df.index))).fillna(0)
    out["takes_lipid_med"] = to_number(df.get("TAKES_LIPID_MED", pd.Series(index=df.index))).fillna(0)
    out["cardio_risk_score"] = to_number(df.get("CARDIO_RISK_SCORE", pd.Series(index=df.index)))
    out["cardio_risk"] = df.get("CARDIO_RISK", "unknown").astype(str)
    out["menopause"] = df.get("MENOPAUSE", "unknown").astype(str)
    out["ascvd_risk_level"] = df.get("ASCVD_RISK_LEVEL", "unknown").astype(str)
    out["heart_age"] = to_number(df.get("HEART_AGE", pd.Series(index=df.index)))

    # 점수형 Feature
    out["exercise_score"] = out["exercise"].map(lambda x: 0 if x in ["없음", "거의 안함", "부족"] else (1 if "1" in x or "2" in x or x == "보통" else 2))
    out["sleep_score"] = out["sleep_hours"].map(lambda h: 100 if h >= 8 else (90 if h >= 7 else (70 if h >= 6 else (50 if h >= 5 else 30))))
    out["stress_score"] = out["stress_level"].map(lambda x: 2 if "높" in str(x) else (1 if "보통" in str(x) else 0))
    out["smoking_score"] = out["smoking"].map(lambda x: 0 if "비흡연" in str(x) else 2)
    out["drinking_score"] = out["drinking"].map(lambda x: 0 if "비음주" in str(x) or "없" in str(x) else (2 if "자주" in str(x) or "주 3" in str(x) else 1))
    out["cardio_risk_encoded"] = out["cardio_risk"].map(risk_level_to_num)
    out["ascvd_risk_encoded"] = out["ascvd_risk_level"].map(risk_level_to_num)
    out["medication_flag"] = out["medication"].map(lambda x: 0 if str(x) in ["없음", "nan", "None"] else 1)

    return out


def generate_nutrition_labels(features: pd.DataFrame) -> pd.DataFrame:
    """의학적 근거 기반 규칙으로 학습용 0/1 라벨을 생성한다.
    이 라벨은 진단이 아니라 건강기능식품 추천 참고용 '부족 가능성/보충 고려' 라벨이다.
    """
    y = pd.DataFrame(index=features.index)

    age = features["age"].fillna(0)
    sex = features["sex"].astype(str)
    bmi = features["bmi"].fillna(0)
    sleep_hours = features["sleep_hours"].fillna(7)
    sleep_score = features["sleep_score"].fillna(70)
    stress_score = features["stress_score"].fillna(0)
    smoking_score = features["smoking_score"].fillna(0)
    drinking_score = features["drinking_score"].fillna(0)
    exercise_score = features["exercise_score"].fillna(0)
    tg = features["triglyceride"].fillna(0)
    ldl = features["ldl"].fillna(0)
    hdl = features["hdl"].fillna(999)
    glucose = features["glucose"].fillna(0)
    cardio_risk = features["cardio_risk_encoded"].fillna(0)
    meds = features["medication_flag"].fillna(0)
    menopause = features["menopause"].astype(str)
    metabolic_count = features["metabolic_syndrome_count"].fillna(0)

    # 현재 CSV에는 야외활동/채소/생선/건강목표가 없으므로, PHR 검사 지표와 생활습관 지표 중심으로 대체한다.
    # 웹 입력에서는 동일한 함수에 outdoor_score, vegetable_score, fish_score, health_goals를 추가하면 더 정확해진다.

    # Vitamin D: 고령, BMI, 낮은 활동성/운동 부족, 뼈 건강 대리지표(폐경 여성) 중심
    vitamin_d_score = (age >= 50).astype(int) * 2 + (bmi >= 25).astype(int) + (exercise_score == 0).astype(int) + ((sex == "여성") & (menopause == "YES")).astype(int) * 2
    y["vitamin_d_risk"] = (vitamin_d_score >= 3).astype(int)

    # Vitamin C: 흡연, 식습관 위험점수/생활습관 위험, 스트레스
    vitamin_c_score = smoking_score * 2 + (features["lifestyle_risk_score"].fillna(0) >= 3).astype(int) + (stress_score >= 2).astype(int)
    y["vitamin_c_risk"] = (vitamin_c_score >= 3).astype(int)

    # Vitamin B: 피로/에너지 대리지표로 수면 부족, 음주, 스트레스, 생활습관 위험 사용
    vitamin_b_score = (sleep_hours < 6).astype(int) * 2 + drinking_score + stress_score + (features["lifestyle_risk_score"].fillna(0) >= 3).astype(int)
    y["vitamin_b_risk"] = (vitamin_b_score >= 3).astype(int)

    # Magnesium: 수면 부족, 스트레스, 운동 부족
    magnesium_score = (sleep_score <= 50).astype(int) * 2 + stress_score + (exercise_score == 0).astype(int)
    y["magnesium_risk"] = (magnesium_score >= 3).astype(int)

    # Calcium: 고령, 여성/폐경, Vitamin D 위험
    calcium_score = (age >= 50).astype(int) * 2 + (sex == "여성").astype(int) + ((sex == "여성") & (menopause == "YES")).astype(int) * 2 + y["vitamin_d_risk"]
    y["calcium_risk"] = (calcium_score >= 3).astype(int)

    # Zinc: 면역/영양 대리지표로 고령, 생활습관 위험, 흡연/스트레스
    zinc_score = (age >= 60).astype(int) + (features["lifestyle_risk_score"].fillna(0) >= 3).astype(int) * 2 + (smoking_score > 0).astype(int) + (stress_score >= 2).astype(int)
    y["zinc_risk"] = (zinc_score >= 3).astype(int)

    # Iron: Hb/Ferritin이 없으므로 보수적으로 여성 가임기 + 피로 대리지표(수면 부족) 중심
    iron_score = ((sex == "여성") & (age < 50)).astype(int) * 2 + (sleep_hours < 6).astype(int) + (features["subjective_health"].astype(str).str.contains("나쁨", na=False)).astype(int)
    y["iron_risk"] = (iron_score >= 3).astype(int)

    # Folate: 여성 가임기, 음주, 식습관 위험 대리지표
    folate_score = ((sex == "여성") & (age.between(15, 49))).astype(int) * 2 + (drinking_score >= 2).astype(int) + (features["lifestyle_risk_score"].fillna(0) >= 3).astype(int)
    y["folate_risk"] = (folate_score >= 3).astype(int)

    # EPA/DHA: 중성지방, LDL, HDL, 심혈관위험 중심
    omega3_score = (tg >= 150).astype(int) * 3 + (ldl >= 130).astype(int) + (hdl < 40).astype(int) + (cardio_risk >= 1).astype(int) * 2
    y["omega3_need"] = (omega3_score >= 3).astype(int)

    # Probiotics: 직접 장 증상 없음. 생활습관/대사 위험 대리지표로 보수적 생성
    probiotics_score = (features["lifestyle_risk_score"].fillna(0) >= 3).astype(int) * 2 + (metabolic_count >= 2).astype(int)
    y["probiotics_need"] = (probiotics_score >= 2).astype(int)

    # Lutein: 눈 건강 직접 feature 부족. 고령 중심으로만 약하게 생성
    lutein_score = (age >= 50).astype(int) * 2
    y["lutein_need"] = (lutein_score >= 2).astype(int)

    # Red Ginseng: 피로/면역 목표가 없으므로 스트레스, 수면부족, 주관건강 대리지표
    red_ginseng_score = (stress_score >= 2).astype(int) + (sleep_hours < 6).astype(int) + (features["subjective_health"].astype(str).str.contains("나쁨", na=False)).astype(int) * 2
    y["red_ginseng_need"] = (red_ginseng_score >= 2).astype(int)

    # Milk Thistle: 잦은 음주 중심
    milk_thistle_score = (drinking_score >= 2).astype(int) * 3
    y["milk_thistle_need"] = (milk_thistle_score >= 3).astype(int)

    # CoQ10: 중장년, 심혈관 위험, 혈압약/지질약 복용 대리지표
    coq10_score = (age >= 40).astype(int) + (cardio_risk >= 1).astype(int) * 2 + (features["takes_antihypertensive"].fillna(0) == 1).astype(int) + (features["takes_lipid_med"].fillna(0) == 1).astype(int)
    y["coq10_need"] = (coq10_score >= 3).astype(int)

    # Policosanol: LDL/HDL/지질약/이상지질혈증 중심
    policosanol_score = (ldl >= 130).astype(int) * 2 + (hdl < 40).astype(int) + (features["takes_lipid_med"].fillna(0) == 1).astype(int) + (features["dyslipidemia"].astype(str).str.contains("이상", na=False)).astype(int) * 2
    y["policosanol_need"] = (policosanol_score >= 3).astype(int)

    # Garcinia: BMI/허리둘레/대사증후군 위험. 간 안전성은 추천 단계에서 주의 처리
    garcinia_score = (bmi >= 25).astype(int) * 3 + (features["waist"].fillna(0) >= 90).astype(int) + (metabolic_count >= 2).astype(int)
    y["garcinia_need"] = (garcinia_score >= 3).astype(int)

    # Saw Palmetto: 남성 50세 이상 중심. 전립선 증상 feature 없어서 보수적
    saw_score = ((sex == "남성") & (age >= 50)).astype(int) * 3
    y["saw_palmetto_need"] = (saw_score >= 3).astype(int)

    return y[LABEL_COLUMNS]


def build_model() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    clf = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])


def get_probabilities(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """MultiOutputClassifier에서 각 label의 class=1 확률을 추출한다."""
    transformed = pipeline.named_steps["preprocessor"].transform(X)
    estimators = pipeline.named_steps["model"].estimators_
    probs = []
    for estimator in estimators:
        proba = estimator.predict_proba(transformed)
        # class 1이 없는 드문 경우 처리
        if len(estimator.classes_) == 1:
            p = np.ones(transformed.shape[0]) if estimator.classes_[0] == 1 else np.zeros(transformed.shape[0])
        else:
            class_one_idx = list(estimator.classes_).index(1)
            p = proba[:, class_one_idx]
        probs.append(p)
    return np.vstack(probs).T


def predict_nutrition_risk(pipeline: Pipeline, user_input: Dict[str, Any], threshold: float = 0.6) -> List[Dict[str, Any]]:
    """웹에서 사용자 1명의 입력을 받아 부족 가능 영양소 예측 결과를 반환한다."""
    raw = pd.DataFrame([user_input])
    features = feature_engineering(raw)
    # 누락된 feature 컬럼 보강
    for col in NUMERIC_FEATURES:
        if col not in features:
            features[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in features:
            features[col] = "unknown"
    features = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    probs = get_probabilities(pipeline, features)[0]
    results = []
    for label, prob in zip(LABEL_COLUMNS, probs):
        results.append(
            {
                "label": label,
                "name": LABEL_DISPLAY_NAMES[label],
                "probability": round(float(prob), 4),
                "selected": bool(prob >= threshold),
            }
        )
    return sorted(results, key=lambda x: x["probability"], reverse=True)


def main() -> None:
    input_path = find_input_file()
    print(f"[INFO] 입력 CSV: {input_path}")
    raw_df = load_phr_csv(input_path)
    print(f"[INFO] 설명행 제거 후 데이터 크기: {raw_df.shape}")

    features = feature_engineering(raw_df)
    # 필요한 컬럼 순서 보장
    X = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = generate_nutrition_labels(features)

    labeled = pd.concat([raw_df.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    labeled.to_csv(LABELED_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] 라벨 추가 CSV 저장: {LABELED_CSV}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_model()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report_lines = []
    report_lines.append("Random Forest 기반 PHR 부족 가능 영양소 예측 모델 평가\n")
    report_lines.append("주의: 본 모델은 건강기능식품 추천 참고용이며 질병 진단 목적이 아닙니다.\n")
    report_lines.append(f"데이터 수: {len(raw_df)}")
    report_lines.append(f"Feature 수: {X.shape[1]}")
    report_lines.append(f"Label 수: {len(LABEL_COLUMNS)}\n")

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    report_lines.append(f"Macro F1: {macro_f1:.4f}")
    report_lines.append(f"Micro F1: {micro_f1:.4f}\n")

    for idx, label in enumerate(LABEL_COLUMNS):
        acc = accuracy_score(y_test.iloc[:, idx], y_pred[:, idx])
        report_lines.append(f"===== {label} ({LABEL_DISPLAY_NAMES[label]}) =====")
        report_lines.append(f"Accuracy: {acc:.4f}")
        report_lines.append(classification_report(y_test.iloc[:, idx], y_pred[:, idx], zero_division=0))

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[INFO] 평가 리포트 저장: {REPORT_PATH}")

    bundle = {
        "pipeline": pipeline,
        "label_columns": LABEL_COLUMNS,
        "label_display_names": LABEL_DISPLAY_NAMES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "disclaimer": "본 모델은 건강기능식품 추천 참고용 부족 가능성 예측 모델이며 질병의 진단, 치료, 예방 목적이 아닙니다.",
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"[INFO] 모델 저장: {MODEL_PATH}")

    # 샘플 예측
    sample_user = raw_df.iloc[0].to_dict()
    results = predict_nutrition_risk(pipeline, sample_user, threshold=0.6)
    print("\n[샘플 사용자 예측 TOP 5]")
    for item in results[:5]:
        print(f"- {item['name']}: {item['probability'] * 100:.1f}%")


if __name__ == "__main__":
    main()
