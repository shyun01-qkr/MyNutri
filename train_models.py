import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

print("=== 데이터 로딩 ===")
df = pd.read_csv('SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER.csv', encoding='utf-8')
df.columns = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)

print(f"전체 데이터: {len(df)}건")

eng_cols = [
    'AGE', 'SEX', 'SMOKING', 'DRINKING', 'EXERCISE', 'SLEEP_HOURS',
    'STRESS_LEVEL', 'SUBJECTIVE_HEALTH', 'LIFESTYLE_RISK_SCORE', 'BMI', 'WAIST',
    'SBP', 'DBP', 'GLUCOSE', 'HDL', 'LDL', 'TRIGLYCERIDE',
    'HYPERTENSION', 'DIABETES', 'DYSLIPIDEMIA', 'METABOLIC_SYNDROME_COUNT',
    'METABOLIC_SYNDROME', 'FAMILY_HISTORY', 'MEDICATION',
    'TAKES_ANTIHYPERTENSIVE', 'TAKES_DIABETES_MED', 'TAKES_LIPID_MED',
    'CARDIO_RISK_SCORE', 'CARDIO_RISK', 'MENOPAUSE', 'ASCVD_RISK_LEVEL', 'HEART_AGE'
]
df.columns = eng_cols

for col in ['AGE', 'SLEEP_HOURS', 'LIFESTYLE_RISK_SCORE', 'BMI', 'WAIST',
            'SBP', 'DBP', 'GLUCOSE', 'HDL', 'LDL', 'TRIGLYCERIDE',
            'CARDIO_RISK_SCORE', 'HEART_AGE', 'METABOLIC_SYNDROME_COUNT',
            'TAKES_ANTIHYPERTENSIVE', 'TAKES_DIABETES_MED', 'TAKES_LIPID_MED']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n=== 전처리: 범주형 변수 인코딩 ===")
categorical_cols = ['SEX', 'SMOKING', 'DRINKING', 'EXERCISE', 'STRESS_LEVEL',
                     'SUBJECTIVE_HEALTH', 'FAMILY_HISTORY', 'MEDICATION',
                     'MENOPAUSE']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

target_encoders = {}
for col in ['HYPERTENSION', 'DIABETES', 'DYSLIPIDEMIA', 'METABOLIC_SYNDROME', 'CARDIO_RISK', 'ASCVD_RISK_LEVEL']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    target_encoders[col] = le
    print(f"  Target {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\n=== 특성/타겟 정의 ===")
base_features = ['AGE', 'SEX', 'SMOKING', 'DRINKING', 'EXERCISE', 'SLEEP_HOURS',
                 'STRESS_LEVEL', 'SUBJECTIVE_HEALTH', 'LIFESTYLE_RISK_SCORE',
                 'BMI', 'WAIST', 'FAMILY_HISTORY', 'MENOPAUSE',
                 'TAKES_ANTIHYPERTENSIVE', 'TAKES_DIABETES_MED', 'TAKES_LIPID_MED']

vital_features = ['SBP', 'DBP', 'GLUCOSE', 'HDL', 'LDL', 'TRIGLYCERIDE']

all_features = base_features + vital_features

X_all = df[all_features].copy()

models = {}

def train_classifier(target_name, use_features, model_name=None):
    if model_name is None:
        model_name = target_name
    print(f"\n--- {model_name} 모델 학습 ---")
    y = df[target_name]
    X = df[use_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    num_cols = [c for c in use_features if c not in categorical_cols]
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  정확도: {acc:.4f}")
    print(f"  분류보고서:\n{classification_report(y_test, y_pred, zero_division=0)}")

    return model, scaler, acc

def train_regressor(target_name, use_features, model_name=None):
    if model_name is None:
        model_name = target_name
    print(f"\n--- {model_name} 모델 학습 (회귀) ---")
    y = df[target_name]
    X = df[use_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    num_cols = [c for c in use_features if c not in categorical_cols]
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"  RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return model, scaler

print("\n========================================")
print("만성질환 위험도 예측 모델 학습 시작")
print("========================================")

# 1. 고혈압 예측 (SBP/DBP 제외)
hyp_features = [c for c in all_features if c not in ['SBP', 'DBP', 'TAKES_ANTIHYPERTENSIVE']]
model_hyp, scaler_hyp, acc_hyp = train_classifier('HYPERTENSION', hyp_features, '고혈압')
models['hypertension'] = {'model': model_hyp, 'scaler': scaler_hyp, 'features': hyp_features, 'accuracy': acc_hyp}

# 2. 당뇨 예측 (GLUCOSE 제외)
dia_features = [c for c in all_features if c not in ['GLUCOSE', 'TAKES_DIABETES_MED']]
model_dia, scaler_dia, acc_dia = train_classifier('DIABETES', dia_features, '당뇨')
models['diabetes'] = {'model': model_dia, 'scaler': scaler_dia, 'features': dia_features, 'accuracy': acc_dia}

# 3. 이상지질혈증 예측 (HDL/LDL/TRIGLYCERIDE 제외)
dys_features = [c for c in all_features if c not in ['HDL', 'LDL', 'TRIGLYCERIDE', 'TAKES_LIPID_MED']]
model_dys, scaler_dys, acc_dys = train_classifier('DYSLIPIDEMIA', dys_features, '이상지질혈증')
models['dyslipidemia'] = {'model': model_dys, 'scaler': scaler_dys, 'features': dys_features, 'accuracy': acc_dys}

# 4. 대사증후군 예측
ms_features = [c for c in all_features if c not in ['WAIST', 'SBP', 'DBP', 'GLUCOSE', 'HDL', 'TRIGLYCERIDE', 'METABOLIC_SYNDROME_COUNT']]
model_ms, scaler_ms, acc_ms = train_classifier('METABOLIC_SYNDROME', ms_features, '대사증후군')
models['metabolic_syndrome'] = {'model': model_ms, 'scaler': scaler_ms, 'features': ms_features, 'accuracy': acc_ms}

# 5. 심혈관 위험등급 예측
cardio_features = [c for c in all_features if c not in ['CARDIO_RISK_SCORE', 'CARDIO_RISK']]
model_cardio, scaler_cardio, acc_cardio = train_classifier('CARDIO_RISK', cardio_features, '심혈관위험등급')
models['cardio_risk'] = {'model': model_cardio, 'scaler': scaler_cardio, 'features': cardio_features, 'accuracy': acc_cardio}

# 6. ASCVD 위험도 예측
ascvd_features = [c for c in all_features if c not in ['ASCVD_RISK_LEVEL']]
model_ascvd, scaler_ascvd, acc_ascvd = train_classifier('ASCVD_RISK_LEVEL', ascvd_features, 'ASCVD위험도')
models['ascvd_risk'] = {'model': model_ascvd, 'scaler': scaler_ascvd, 'features': ascvd_features, 'accuracy': acc_ascvd}

# 7. HEART_AGE 예측 (회귀)
heart_features = [c for c in all_features if c not in ['HEART_AGE', 'CARDIO_RISK_SCORE']]
model_heart, scaler_heart = train_regressor('HEART_AGE', heart_features, '심장나이')
models['heart_age'] = {'model': model_heart, 'scaler': scaler_heart, 'features': heart_features}

# 8. 대사증후군 구성요소 수 예측 (회귀)
msc_features = [c for c in all_features if c not in ['WAIST', 'SBP', 'DBP', 'GLUCOSE', 'HDL', 'TRIGLYCERIDE', 'METABOLIC_SYNDROME', 'METABOLIC_SYNDROME_COUNT']]
model_msc, scaler_msc = train_regressor('METABOLIC_SYNDROME_COUNT', msc_features, '대사증후군구성요소수')
models['ms_count'] = {'model': model_msc, 'scaler': scaler_msc, 'features': msc_features}

print("\n========================================")
print("모델 저장 중...")
print("========================================")

os.makedirs('models', exist_ok=True)

joblib.dump(models, 'models/all_models.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(target_encoders, 'models/target_encoders.pkl')

feature_info = {
    'all_features': all_features,
    'categorical_cols': categorical_cols,
    'base_features': base_features,
    'vital_features': vital_features
}
joblib.dump(feature_info, 'models/feature_info.pkl')

print("\n=== 모델 학습 완료! ===")
print(f"저장된 모델: {list(models.keys())}")
for name, info in models.items():
    if 'accuracy' in info:
        print(f"  {name}: 정확도 {info['accuracy']:.4f}")
