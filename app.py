import os, io, base64, json, re
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from flask import Flask, render_template, jsonify, request, send_file
from train_random_forest import predict_nutrition_risk

app = Flask(__name__)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────
# 1. Load supplement product data from CSVs
# ──────────────────────────────────────────────
print("제품 데이터 로딩 중...")
df_products = pd.read_csv('01_product_items_100_clean.csv', encoding='utf-8')
df_ingredients = pd.read_csv('02_functional_ingredients_100_clean.csv', encoding='utf-8')
df_nutrition = pd.read_csv('03_nutrition_facts_100_clean.csv', encoding='utf-8')
df_labels = pd.read_csv('04_label_information_100_clean.csv', encoding='utf-8')

# Build unified product lookup with full JOIN
products_by_id = {}
for _, row in df_products.iterrows():
    pid = row['product_id']
    products_by_id[pid] = {
        'product_id': pid,
        'product_name': row['product_name'],
        'company_name': row['company_name'],
        'product_type': row['product_type'],
        'manufacture_type': row['manufacture_type'],
        'form_type': row['form_type'],
        'status': row['status'],
        'report_no': row['report_no'],
        'nutrition_facts': {},
        'functional_ingredients': [],
    }

# Add nutrition data as nested dict
for _, row in df_nutrition.iterrows():
    pid = row['product_id']
    if pid in products_by_id:
        products_by_id[pid]['serving_size'] = row['serving_size']
        products_by_id[pid]['servings_per_day'] = int(row['servings_per_day']) if pd.notna(row['servings_per_day']) else 1
        products_by_id[pid]['nutrition_facts'] = {
            'energy_kcal': float(row['energy_kcal']) if pd.notna(row['energy_kcal']) else 0,
            'vitamin_c_mg': float(row['vitamin_c_mg']) if pd.notna(row['vitamin_c_mg']) else 0,
            'vitamin_d_ug': float(row['vitamin_d_ug']) if pd.notna(row['vitamin_d_ug']) else 0,
            'calcium_mg': float(row['calcium_mg']) if pd.notna(row['calcium_mg']) else 0,
            'magnesium_mg': float(row['magnesium_mg']) if pd.notna(row['magnesium_mg']) else 0,
            'zinc_mg': float(row['zinc_mg']) if pd.notna(row['zinc_mg']) else 0,
            'iron_mg': float(row['iron_mg']) if pd.notna(row['iron_mg']) else 0,
            'epa_dha_mg': float(row['epa_dha_mg']) if pd.notna(row['epa_dha_mg']) else 0,
            'lutein_mg': float(row['lutein_mg']) if pd.notna(row['lutein_mg']) else 0,
            'probiotics_cfu': float(row['probiotics_cfu']) if pd.notna(row['probiotics_cfu']) else 0,
            'folic_acid_ug': float(row['folic_acid_ug']) if pd.notna(row['folic_acid_ug']) else 0,
        }

# Add functional ingredients as array (product_id JOIN)
for _, row in df_ingredients.iterrows():
    pid = row['product_id']
    if pid in products_by_id:
        products_by_id[pid]['functional_ingredients'].append({
            'ingredient_id': row['ingredient_id'],
            'ingredient_name': row['ingredient_name'],
            'ingredient_category': row['ingredient_category'],
            'function_category': row['function_category'],
            'functional_claim': row['functional_claim'],
            'daily_intake': row['daily_intake'],
        })

# Add label info
for _, row in df_labels.iterrows():
    pid = row['product_id']
    if pid in products_by_id:
        products_by_id[pid].update({
            'intake_method': row['intake_method'],
            'intake_time': row['intake_time'],
            'warning': row['warning'],
            'allergen_info': row['allergen_info'],
            'raw_materials': row['raw_materials'],
            'functional_display': row['functional_display'],
            'storage_method': row['storage_method'],
            'barcode': row['barcode'],
        })

product_list = list(products_by_id.values())

# ──────────────────────────────────────────────
# 1.5 Load health goal mapping from config
# ──────────────────────────────────────────────
with open('config/health_goal_mapping.json', 'r', encoding='utf-8') as f:
    goal_config = json.load(f)

GOAL_INGREDIENT_MAP = {}
for gk, gv in goal_config['health_goals'].items():
    GOAL_INGREDIENT_MAP[gk] = gv.get('ingredient_names_for_match', [])

GOAL_DEFICIENCY_MAP = goal_config['goal_deficiency_map']

DEFICIENCY_TO_NUTRITION = {}
for dk, dv in goal_config['deficiency_to_nutrition'].items():
    DEFICIENCY_TO_NUTRITION[dk] = (dv['key'], dv['display'], dv['unit'])

GOAL_NUTRITION_SCORE_MAP = goal_config['goal_nutrition_score_map']

FUNCTION_CATEGORY_TO_GOAL = goal_config['function_category_to_goal']

GOAL_ANALYSIS_DESCRIPTIONS = goal_config['goal_analysis_descriptions']

print(f"  로드된 건강 목표: {list(GOAL_INGREDIENT_MAP.keys())}")

# Random Forest ��� �ε�
from sklearn.compose import _column_transformer
if not hasattr(_column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    _column_transformer._RemainderColsList = _RemainderColsList

nutrition_pipeline = None
try:
    nutrition_bundle = joblib.load('nutrition_random_forest.pkl')
    if isinstance(nutrition_bundle, dict) and 'pipeline' in nutrition_bundle:
        nutrition_pipeline = nutrition_bundle['pipeline']
        print('  Random Forest ���Ҽ� �м� ��� �ε� ����')
    else:
        nutrition_pipeline = nutrition_bundle
        print('  Random Forest ���Ҽ� �м� ��� �ε� ����')
except Exception as e:
    nutrition_pipeline = None
    print(f'  Random Forest ��� �ε� ����, Rule-based �м� ���: {e}')

LABEL_TO_KOREAN = {
    'vitamin_d_risk': '비타민D',
    'vitamin_c_risk': '비타민C',
    'vitamin_b_risk': '비타민B군',
    'magnesium_risk': '마그네슘',
    'calcium_risk': '칼슘',
    'zinc_risk': '아연',
    'iron_risk': '철분',
    'folate_risk': '엽산',
    'omega3_need': 'EPA/DHA',
    'probiotics_need': '프로바이오틱스',
    'lutein_need': '루테인',
    'red_ginseng_need': '홍삼',
    'milk_thistle_need': '밀크씨슬',
    'coq10_need': '코엔자임Q10',
    'policosanol_need': '폴리코사놀',
    'garcinia_need': '가르시니아',
    'saw_palmetto_need': '쏘팔메토',
}

def predict_deficiencies_ml(data):
    try:
        user_input = {
            'AGE': float(data.get('age', 30)),
            'SEX': data.get('gender', '남성'),
            'BMI': 23 if not (data.get('height') and data.get('weight')) else round(float(data['weight']) / ((float(data['height'])/100)**2), 1),
            'WAIST': float(data.get('waist', 0)) or 0,
            'SBP': float(data.get('sbp', 0)) or 120,
            'DBP': float(data.get('dbp', 0)) or 80,
            'GLUCOSE': float(data.get('glucose', 0)) or 0,
            'HDL': float(data.get('hdl', 0)) or 0,
            'LDL': float(data.get('ldl', 0)) or 0,
            'TRIGLYCERIDE': float(data.get('triglyceride', 0)) or 0,
            'SLEEP_HOURS': float(data.get('sleep_hours', 7)),
            'HEART_AGE': float(data.get('heart_age', 0)) or 0,
            'LIFESTYLE_RISK_SCORE': float(data.get('lifestyle_risk_score', 0)) or 0,
            'CARDIO_RISK_SCORE': float(data.get('cardio_risk_score', 0)) or 0,
            'METABOLIC_SYNDROME_COUNT': float(data.get('metabolic_syndrome_count', 0)) or 0,
            'TAKES_ANTIHYPERTENSIVE': float(data.get('takes_antihypertensive', 0)) or 0,
            'TAKES_DIABETES_MED': float(data.get('takes_diabetes_med', 0)) or 0,
            'TAKES_LIPID_MED': float(data.get('takes_lipid_med', 0)) or 0,
            'SMOKING': data.get('smoking', '비흡연'),
            'DRINKING': data.get('alcohol', '비음주'),
            'EXERCISE': data.get('exercise_freq', '거의 안함'),
            'STRESS_LEVEL': data.get('stress_level', '보통'),
            'SUBJECTIVE_HEALTH': data.get('subjective_health', '보통'),
            'HYPERTENSION': data.get('hypertension', '없음'),
            'DIABETES': data.get('diabetes', '없음'),
            'DYSLIPIDEMIA': data.get('dyslipidemia', '없음'),
            'METABOLIC_SYNDROME': data.get('metabolic_syndrome', '아니요'),
            'FAMILY_HISTORY': data.get('family_history', '없음'),
            'MEDICATION': data.get('medication', '없음'),
            'CARDIO_RISK': data.get('cardio_risk', 'LOW'),
            'MENOPAUSE': data.get('menopause', 'NO'),
            'ASCVD_RISK_LEVEL': data.get('ascvd_risk_level', 'LOW'),
        }
        results = predict_nutrition_risk(nutrition_pipeline, user_input, threshold=0.6)
        deficiencies = []
        reasons = []
        caution_ingredients = []
        caution_reasons = []
        for item in results:
            if item['selected']:
                label = item['label']
                korean_name = LABEL_TO_KOREAN.get(label, label)
                deficiencies.append(korean_name)
                name_val = item['name']
                prob_val = item['probability'] * 100
                reasons.append(f'Random Forest 분석 결과 {name_val} 부족 가능성이 {prob_val:.1f}%로 확인되었습니다.')
        ml_predictions_data = []
        for item in results:
            label = item['label']
            korean_name = LABEL_TO_KOREAN.get(label, label)
            ml_predictions_data.append({
                'label': label,
                'nutrient': korean_name,
                'probability': item['probability'],
                'percentage': round(item['probability'] * 100, 1),
                'selected': item['selected'],
            })
        if not deficiencies:
            rule_result = analyze_deficiencies(data)
            result = {
                'deficiencies': rule_result.get('deficiencies', []),
                'reasons': rule_result.get('reasons', []),
                'caution_ingredients': rule_result.get('caution_ingredients', []),
                'caution_reasons': rule_result.get('caution_reasons', []),
                'ml_predictions': ml_predictions_data,
                'analysis_source': 'random_forest',
            }
        else:
            result = {
                'deficiencies': deficiencies,
                'reasons': reasons,
                'caution_ingredients': caution_ingredients,
                'caution_reasons': caution_reasons,
                'ml_predictions': ml_predictions_data,
                'analysis_source': 'random_forest',
            }
        print("analysis_source:", result.get("analysis_source"))
        print("ml_predictions:", result.get("ml_predictions"))
        return result
    except Exception as e:
        print(f'ML 분석 오류, Rule-based로 폴백: {e}')
        result = analyze_deficiencies(data)
        result['analysis_source'] = 'rule_based'
        result['ml_predictions'] = []
        return result

# ��������������������������������������������������������������������������������������������
# 2. Deficiency analysis
# ��������������������������������������������������������������������������������������������

def analyze_deficiencies(data):
    """Analyze user data and return potential nutrient deficiencies and cautions."""
    deficiencies = []
    reasons = []
    caution_ingredients = []
    caution_reasons = []

    try:
        sleep = float(data.get('sleep_hours', 7))
    except:
        sleep = 7
    if sleep < 6:
        deficiencies.extend(['마그네슘', '비타민B군'])
        reasons.append('수면 시간이 6시간 미만으로 마그네슘과 비타민B군이 부족할 수 있습니다.')

    outdoor = data.get('outdoor_activity', '보통')
    if outdoor == '낮음':
        deficiencies.append('비타민D')
        reasons.append('야외활동이 적어 비타민D 합성이 부족할 수 있습니다.')

    veg = data.get('vegetable_intake', '보통')
    if veg == '부족':
        deficiencies.append('비타민C')
        reasons.append('채소 섭취가 부족하여 비타민C 보충이 필요할 수 있습니다.')

    fish = data.get('fish_intake', '보통')
    if fish == '부족':
        deficiencies.append('EPA/DHA')
        reasons.append('생선 섭취가 부족하여 오메가3(EPA/DHA) 보충이 필요할 수 있습니다.')

    alc = data.get('alcohol', '비음주')
    if alc in ['주3~4회', '매일']:
        deficiencies.append('밀크씨슬')
        deficiencies.append('비타민B군')
        reasons.append('잦은 음주로 간 건강 관리와 비타민B군 보충이 필요할 수 있습니다.')
    elif alc in ['주1~2회']:
        if '비타민B군' not in deficiencies:
            deficiencies.append('비타민B군')
        reasons.append('음주로 인해 비타민B군 보충이 필요할 수 있습니다.')

    smoke = data.get('smoking', '비흡연')
    if smoke == '현재흡연':
        deficiencies.append('비타민C')
        reasons.append('흡연으로 인한 항산화 보호를 위해 비타민C 보충이 필요할 수 있습니다.')
    elif smoke == '과거흡연':
        if '비타민C' not in deficiencies:
            deficiencies.append('비타민C')
        reasons.append('과거 흡연 이력이 있어 항산화 영양소 보충이 도움이 될 수 있습니다.')

    stress = data.get('stress_level', '보통')
    if stress == '높음':
        if '비타민B군' not in deficiencies:
            deficiencies.append('비타민B군')
        if '마그네슘' not in deficiencies:
            deficiencies.append('마그네슘')
        reasons.append('스트레스 수준이 높아 비타민B군과 마그네슘 보충이 도움이 될 수 있습니다.')

    caffeine = data.get('caffeine', '하루1잔')
    if caffeine in ['하루3잔 이상']:
        if '마그네슘' not in deficiencies:
            deficiencies.append('마그네슘')
        reasons.append('카페인 섭취가 많아 마그네슘 보충이 필요할 수 있습니다.')

    health_goals = data.get('health_goals', [data.get('health_goal', '')])
    if isinstance(health_goals, str):
        health_goals = [health_goals]
    for goal in health_goals:
        goal_deficiencies = GOAL_DEFICIENCY_MAP.get(goal, [])
        for d in goal_deficiencies:
            if d not in deficiencies:
                deficiencies.append(d)
        desc = GOAL_ANALYSIS_DESCRIPTIONS.get(goal, '')
        if desc:
            reasons.append(desc)
        # Legacy keyword matching for backward compatibility
        if goal not in GOAL_DEFICIENCY_MAP:
            if '뼈' in goal or '골밀도' in goal:
                for d in ['칼슘', '비타민D', '마그네슘']:
                    if d not in deficiencies: deficiencies.append(d)
                reasons.append('뼈 건강을 위해 칼슘, 비타민D, 마그네슘 보충이 도움이 될 수 있습니다.')
            if '혈행' in goal or '혈관' in goal or '콜레스테롤' in goal:
                if 'EPA/DHA' not in deficiencies: deficiencies.append('EPA/DHA')
                reasons.append('혈행 개선을 위해 오메가3(EPA/DHA) 보충이 도움이 될 수 있습니다.')
            if '피로' in goal or '활력' in goal:
                if '비타민B군' not in deficiencies: deficiencies.append('비타민B군')
                if '코엔자임Q10' not in deficiencies: deficiencies.append('코엔자임Q10')
                reasons.append('피로 개선을 위해 비타민B군과 코엔자임Q10 보충이 도움이 될 수 있습니다.')
            if '면역' in goal:
                if '비타민C' not in deficiencies: deficiencies.append('비타민C')
                if '아연' not in deficiencies: deficiencies.append('아연')
                reasons.append('면역 기능을 위해 비타민C와 아연 보충이 도움이 될 수 있습니다.')
            if '장' in goal or '소화' in goal:
                if '프로바이오틱스' not in deficiencies: deficiencies.append('프로바이오틱스')
                reasons.append('장 건강을 위해 프로바이오틱스 보충이 도움이 될 수 있습니다.')
            if '눈' in goal or '시력' in goal:
                if '루테인' not in deficiencies: deficiencies.append('루테인')
                reasons.append('눈 건강을 위해 루테인 보충이 도움이 될 수 있습니다.')
            if '체지방' in goal or '다이어트' in goal:
                if '가르시니아' not in deficiencies: deficiencies.append('가르시니아')
                reasons.append('체지방 감소에 가르시니아가 도움이 될 수 있습니다.')

    ex = data.get('exercise_freq', '거의 안함')
    if ex in ['거의 안함']:
        reasons.append('운동 빈도가 낮아 전반적인 영양 관리가 필요합니다.')

    try:
        age_val = int(float(data.get('age', 30)))
        if age_val >= 50:
            if '칼슘' not in deficiencies:
                deficiencies.append('칼슘')
            if '비타민D' not in deficiencies:
                deficiencies.append('비타민D')
            reasons.append('연령대가 높아짐에 따라 칼슘과 비타민D 보충이 중요합니다.')
    except:
        pass

    med = data.get('medication', '없음')
    if '항응고제' in med or '와파린' in med:
        caution_ingredients.append('오메가3(EPA/DHA)')
        caution_reasons.append('항응고제 복용 시 오메가3를 고용량 섭취하면 출혈 위험이 증가할 수 있으므로 전문가와 상담이 필요합니다.')
        caution_ingredients.append('비타민K')
        caution_reasons.append('항응고제 복용 시 비타민K 함유 제품은 일관된 섭취량을 유지하는 것이 중요합니다.')
    if '고혈압약' in med:
        caution_ingredients.append('혈압관련 기능성 원료')
        caution_reasons.append('고혈압약 복용 시 혈압에 영향을 주는 기능성 원료는 주의가 필요합니다.')
    if '당뇨약' in med:
        caution_ingredients.append('혈당관련 성분')
        caution_reasons.append('당뇨약 복용 시 혈당에 영향을 줄 수 있는 성분은 주의가 필요합니다.')
    if '갑상선' in med:
        caution_ingredients.append('칼슘, 철분')
        caution_reasons.append('갑상선약 복용 시 칼슘, 철분 섭취는 4시간 이상 간격을 두고 섭취하세요.')
    if '이뇨제' in med:
        if '마그네슘' not in deficiencies:
            deficiencies.append('마그네슘')
        reasons.append('이뇨제 복용으로 마그네슘 배출이 증가할 수 있습니다.')

    deficiencies = list(dict.fromkeys(deficiencies))
    reasons = list(dict.fromkeys(reasons))
    caution_ingredients = list(dict.fromkeys(caution_ingredients))
    caution_reasons = list(dict.fromkeys(caution_reasons))

    return {
        'deficiencies': deficiencies,
        'reasons': reasons,
        'caution_ingredients': caution_ingredients,
        'caution_reasons': caution_reasons,
    }

# ──────────────────────────────────────────────
# 3. Enhanced recommendation engine
# ──────────────────────────────────────────────



def _get_ingredient_names(product):
    return [i['ingredient_name'] for i in product.get('functional_ingredients', [])] + \
           [i['function_category'] for i in product.get('functional_ingredients', [])] + \
           [i['functional_claim'] for i in product.get('functional_ingredients', [])]

def calculate_recommendation_score(product, analysis, data):
    """Calculate recommendation score (0-100) using CSV data."""
    scores = {}
    nf = product.get('nutrition_facts', {})
    ing_names = _get_ingredient_names(product)
    all_text = ' '.join(ing_names).lower()

    # ── ① 건강 적합도 (30점) ──
    goal = data.get('health_goal', '')
    health_score = 0
    func_ings = product.get('functional_ingredients', [])

    # (a) function_category 정확 매칭 (15점)
    for fi in func_ings:
        fc = fi.get('function_category', '')
        if not fc:
            continue
        if goal in fc:
            health_score += 15
            break
        mapped_goal = FUNCTION_CATEGORY_TO_GOAL.get(fc, '')
        if mapped_goal == goal:
            health_score += 15
            break

    # (b) functional_claim 매칭 (10점)
    if health_score < 30:
        for fi in func_ings:
            claim = fi.get('functional_claim', '')
            if claim and goal.replace(' ', '') in claim.replace(' ', ''):
                health_score += 10
                break

    # (c) ingredient_name 매칭 (5점)
    if health_score < 30:
        goal_ingredients = GOAL_INGREDIENT_MAP.get(goal, [])
        for gi in goal_ingredients:
            gi_lower = gi.lower().replace(' ', '')
            if any(gi_lower in x.lower().replace(' ', '') for x in ing_names):
                health_score += 5

    # (d) nutrition_facts 보너스 (5점)
    if health_score < 30:
        for nk in GOAL_NUTRITION_SCORE_MAP.get(goal, []):
            if nf.get(nk, 0) > 0:
                health_score += 5
                break

    health_score = min(health_score, 30)
    scores['건강적합도'] = health_score

    # ── ② 부족 영양소 보충 (25점) ──
    supp_score = 0
    deficiency_matches = []
    for d in analysis.get('deficiencies', []):
        mapping = DEFICIENCY_TO_NUTRITION.get(d)
        if mapping:
            key, _, _ = mapping
            val = nf.get(key, 0)
            if val > 0:
                supp_score += 6
                deficiency_matches.append(d)
        elif d == '비타민B군':
            if nf.get('folic_acid_ug', 0) > 0:
                supp_score += 6
                deficiency_matches.append(d)
        elif d == '코엔자임Q10':
            if any('코엔자임' in x for x in ing_names):
                supp_score += 6
                deficiency_matches.append(d)
        elif d == '밀크씨슬':
            if any('밀크씨슬' in x for x in ing_names):
                supp_score += 6
                deficiency_matches.append(d)
        elif d == '가르시니아':
            if any('가르시니아' in x for x in ing_names):
                supp_score += 6
                deficiency_matches.append(d)
    supp_score = min(supp_score, 25)
    scores['부족영양소보충'] = supp_score

    # ── ③ 안전성 (20점) ──
    safety_score = 20
    med = data.get('medication', '').lower()
    allergen = data.get('allergies', '').lower()

    epa_related = any('epa' in x.lower() or 'dha' in x.lower() or '오메가3' in x or '혈행' in x for x in ing_names)
    calcium_related = any('칼슘' in x for x in ing_names) or nf.get('calcium_mg', 0) > 0

    if '항응고제' in med and epa_related:
        safety_score -= 10
    if '갑상선' in med and calcium_related:
        safety_score -= 5

    raw = (product.get('raw_materials', '') or '').lower()
    product_allergen = (product.get('allergen_info', '') or '').lower()
    if '알레르기' in allergen or ('있음' in allergen):
        if any(a in raw for a in ['우유', '대두', '밀', '난류', '땅콩', '복숭아']):
            safety_score -= 5

    warning = product.get('warning', '')
    if warning and str(warning) != 'nan' and len(str(warning)) > 5:
        safety_score += 5
    safety_score = max(safety_score, 0)
    safety_score = min(safety_score, 20)
    scores['안전성'] = safety_score

    # ── ④ 생활습관 적합도 (10점) ──
    lifestyle_score = 0
    try:
        sleep = float(data.get('sleep_hours', 7))
    except:
        sleep = 7
    if sleep < 6:
        has_b = nf.get('folic_acid_ug', 0) > 0
        has_mg = nf.get('magnesium_mg', 0) > 0
        if has_b or has_mg:
            lifestyle_score += 4

    outdoor = data.get('outdoor_activity', '보통')
    if outdoor == '낮음' and nf.get('vitamin_d_ug', 0) > 0:
        lifestyle_score += 4

    veg = data.get('vegetable_intake', '보통')
    if veg == '부족' and nf.get('vitamin_c_mg', 0) > 0:
        lifestyle_score += 4

    fish = data.get('fish_intake', '보통')
    if fish == '부족' and nf.get('epa_dha_mg', 0) > 0:
        lifestyle_score += 4

    smoking = data.get('smoking', '비흡연')
    if smoking in ['현재흡연', '과거흡연'] and nf.get('vitamin_c_mg', 0) > 0:
        lifestyle_score += 3

    alcohol = data.get('alcohol', '비음주')
    if alcohol in ['주3~4회', '매일'] and nf.get('folic_acid_ug', 0) > 0:
        lifestyle_score += 3

    exercise = data.get('exercise_freq', '거의 안함')
    if exercise == '거의 안함':
        if nf.get('magnesium_mg', 0) > 0 or nf.get('vitamin_d_ug', 0) > 0:
            lifestyle_score += 2

    lifestyle_score = min(lifestyle_score, 10)
    scores['생활습관적합도'] = lifestyle_score

    # ── ⑤ 가성비 (10점) ──
    func_count = len(product.get('functional_ingredients', []))
    nutrient_count = sum(1 for k in ['vitamin_c_mg', 'vitamin_d_ug', 'calcium_mg', 'magnesium_mg',
                                     'zinc_mg', 'iron_mg', 'epa_dha_mg', 'lutein_mg',
                                     'probiotics_cfu', 'folic_acid_ug'] if nf.get(k, 0) > 0)
    total_count = func_count + nutrient_count
    cost_score = 0
    if total_count >= 10:
        cost_score = 10
    elif total_count >= 8:
        cost_score = 8
    elif total_count >= 6:
        cost_score = 6
    elif total_count >= 4:
        cost_score = 4
    elif total_count >= 2:
        cost_score = 2
    scores['가성비'] = cost_score

    # ── ⑥ 복용 편의성 (5점) ──
    sv = product.get('servings_per_day', 1)
    if sv <= 1:
        conv_score = 5
    elif sv <= 2:
        conv_score = 4
    else:
        conv_score = 2
    scores['복용편의성'] = conv_score

    total = sum(scores.values())
    return total, scores, deficiency_matches

def generate_recommendation_reasons(product, scores, deficiency_matches, analysis, data):
    """Generate minimum 3 recommendation reasons from CSV data."""
    reasons = []
    nf = product.get('nutrition_facts', {})
    ing_names = _get_ingredient_names(product)

    # Reason 1: Health goal match
    goal = data.get('health_goal', '')
    if goal and scores.get('건강적합도', 0) >= 10:
        reasons.append(f"사용자의 건강 목표 '{goal}'와(과) 가장 잘 맞는 제품입니다.")

    # Reason 2: Deficiency match
    for d in deficiency_matches:
        reasons.append(f"부족한 {d} 보충에 적합합니다.")

    # Reason 3: Functional ingredient
    for ing in product.get('functional_ingredients', []):
        claim = ing.get('functional_claim', '')
        if claim and str(claim) != 'nan':
            reasons.append(f"{claim} 기능성 원료({ing['ingredient_name']})를 포함합니다.")
            break

    # Reason 4: Convenience
    sv = product.get('servings_per_day', 1)
    if sv <= 1:
        reasons.append("하루 1회 섭취로 복용이 편리합니다.")
    elif sv <= 2:
        reasons.append(f"하루 {sv}회 섭취로 복용이 간편합니다.")

    # Reason 5: Lifestyle fit
    try:
        sleep = float(data.get('sleep_hours', 7))
    except:
        sleep = 7
    if sleep < 6 and (nf.get('magnesium_mg', 0) > 0 or nf.get('folic_acid_ug', 0) > 0):
        reasons.append("생활습관 분석 결과에 적합한 영양성분(마그네슘, 비타민B)을 포함합니다.")

    # Reason 6: Nutrient richness
    nutrient_count = sum(1 for k in ['vitamin_c_mg', 'vitamin_d_ug', 'calcium_mg', 'magnesium_mg',
                                     'zinc_mg', 'iron_mg', 'epa_dha_mg', 'lutein_mg',
                                     'probiotics_cfu', 'folic_acid_ug'] if nf.get(k, 0) > 0)
    if nutrient_count >= 5:
        reasons.append(f"총 {nutrient_count}종의 주요 영양성분을 함유하고 있습니다.")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique[:5]

def generate_caution_reasons(product, analysis, data):
    """Generate caution reasons from CSV and user data."""
    cautions = []
    med = data.get('medication', '')

    # Medication interactions
    ing_names = _get_ingredient_names(product)
    nf = product.get('nutrition_facts', {})

    if '항응고제' in med:
        epa = nf.get('epa_dha_mg', 0)
        if epa > 0:
            cautions.append('항응고제 복용 중이라면 EPA/DHA 섭취 전 전문가와 상담하세요.')

    if '갑상선' in med:
        if nf.get('calcium_mg', 0) > 0 or nf.get('iron_mg', 0) > 0:
            cautions.append('갑상선약과는 시간 간격을 두고 섭취하세요.')

    # Allergen
    allergen_info = product.get('allergen_info', '')
    if allergen_info and str(allergen_info) != 'nan' and str(allergen_info).strip():
        user_allergies = data.get('allergies', '').lower()
        if '있음' in user_allergies or user_allergies:
            cautions.append(f"{allergen_info} 알레르기가 있다면 섭취에 주의하세요.")

    # Warning from CSV
    warning = product.get('warning', '')
    if warning and str(warning) != 'nan' and len(str(warning)) > 3:
        cautions.append(str(warning))

    # Caution from functional ingredients
    for ing in product.get('functional_ingredients', []):
        caution_text = ing.get('caution', '')
        if caution_text and str(caution_text) != 'nan' and len(str(caution_text)) > 3:
            cautions.append(str(caution_text))

    seen = set()
    unique = []
    for c in cautions:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:5]

def _product_contains_nutrient(product, deficiency):
    """Check if a product contains a specific deficiency nutrient."""
    nf = product.get('nutrition_facts', {})
    mapping = DEFICIENCY_TO_NUTRITION.get(deficiency)
    if mapping:
        key, _, _ = mapping
        if key is not None:
            return nf.get(key, 0) > 0
    # Non-nutrition ingredients (checked by ingredient_name)
    if deficiency == '비타민B군':
        return nf.get('folic_acid_ug', 0) > 0
    if deficiency == '코엔자임Q10':
        return any('코엔자임' in x['ingredient_name'] for x in product.get('functional_ingredients', []))
    if deficiency == '밀크씨슬':
        return any('밀크씨슬' in x['ingredient_name'] for x in product.get('functional_ingredients', []))
    if deficiency == '가르시니아':
        return any('가르시니아' in x['ingredient_name'] for x in product.get('functional_ingredients', []))
    # General fallback: check if deficiency name appears in ingredient names
    ings = ' '.join([x['ingredient_name'] for x in product.get('functional_ingredients', [])])
    return deficiency in ings

def _add_to_seen(seen, pid, product, analysis, data, MIN_SCORE):
    total, scores, def_matches = calculate_recommendation_score(product, analysis, data)
    if total < MIN_SCORE:
        return
    if pid in seen and total <= seen[pid]['total_score']:
        return
    reasons = generate_recommendation_reasons(product, scores, def_matches, analysis, data)
    cautions = generate_caution_reasons(product, analysis, data)
    if total >= 85:
        level = '적극 추천'
    elif total >= 70:
        level = '추천'
    else:
        level = '조건부 추천'
    seen[pid] = {
        'product': product,
        'total_score': round(total, 1),
        'score_breakdown': scores,
        'recommendationReasons': reasons,
        'cautionReasons': cautions,
        'deficiency_matches': def_matches,
        'recommendation_level': level,
    }

def generate_recommendations(analysis, data):
    """Generate recommendations grouped by health goal. Top 3 per group."""
    MIN_SCORE = 60
    MAX_PER_GROUP = 3
    health_goals = data.get('health_goals', [data.get('health_goal', '')])
    if isinstance(health_goals, str):
        health_goals = [health_goals]
    groups = []

    for goal in health_goals:
        goal_deficiencies = GOAL_DEFICIENCY_MAP.get(goal, [])
        seen = {}

        if goal_deficiencies:
            # Find products by deficiency match
            for deficiency in goal_deficiencies:
                for product in product_list:
                    if product.get('status') == '판매중지':
                        continue
                    if not _product_contains_nutrient(product, deficiency):
                        continue
                    pid = product['product_id']
                    _add_to_seen(seen, pid, product, analysis, data, MIN_SCORE)
        else:
            # Find products by function_category or ingredient match
            goal_ings = GOAL_INGREDIENT_MAP.get(goal, [])
            for product in product_list:
                if product.get('status') == '판매중지':
                    continue
                func_cats = [fi.get('function_category', '') for fi in product.get('functional_ingredients', [])]
                ing_names = [fi.get('ingredient_name', '') for fi in product.get('functional_ingredients', [])]
                mapped_cats = [FUNCTION_CATEGORY_TO_GOAL.get(fc, '') for fc in func_cats]
                all_text = ' '.join(func_cats + mapped_cats + ing_names)
                if not any(gi.lower().replace(' ','') in all_text.lower().replace(' ','') for gi in goal_ings):
                    if goal not in all_text:
                        continue
                pid = product['product_id']
                _add_to_seen(seen, pid, product, analysis, data, MIN_SCORE)

        candidates = sorted(seen.values(), key=lambda x: x['total_score'], reverse=True)
        top = candidates[:MAX_PER_GROUP]

        groups.append({
            'goal': goal,
            'deficiencies': goal_deficiencies,
            'products': top,
        })

    return {'health_goal_groups': groups}

NUTRITION_KEYS = ['vitamin_c_mg', 'vitamin_d_ug', 'calcium_mg', 'magnesium_mg',
                  'zinc_mg', 'iron_mg', 'epa_dha_mg', 'lutein_mg',
                  'probiotics_cfu', 'folic_acid_ug']

def _product_covers_goal(product, goal):
    """Check if a product covers a specific health goal."""
    goal_ings = GOAL_INGREDIENT_MAP.get(goal, [])
    ing_names = _get_ingredient_names(product)
    func_cats = [fi.get('function_category', '') for fi in product.get('functional_ingredients', [])]
    mapped_cats = [FUNCTION_CATEGORY_TO_GOAL.get(fc, '') for fc in func_cats]
    all_text = ' '.join(ing_names + func_cats + mapped_cats)
    if goal in all_text:
        return True
    for gi in goal_ings:
        if any(gi.lower().replace(' ','') in x.lower().replace(' ','') for x in ing_names):
            return True
    return False

def _product_covers_deficiency(product, deficiency):
    """Check if a product contains a specific deficiency nutrient/ingredient."""
    nf = product.get('nutrition_facts', {})
    mapping = DEFICIENCY_TO_NUTRITION.get(deficiency)
    if mapping:
        key, _, _ = mapping
        if key is not None and nf.get(key, 0) > 0:
            return True
    ings = ' '.join([x['ingredient_name'] for x in product.get('functional_ingredients', [])])
    return deficiency in ings

def _get_nutrients_set(product):
    """Return set of nutrition keys that are present (>0) in this product."""
    nf = product.get('nutrition_facts', {})
    return {k for k in NUTRITION_KEYS if nf.get(k, 0) > 0}

def _compute_nutrient_overlap(products):
    """Count nutrients that appear in more than one product in the combination."""
    all_nutrient_lists = [_get_nutrients_set(p) for p in products]
    overlap_count = 0
    for i in range(len(all_nutrient_lists)):
        for j in range(i + 1, len(all_nutrient_lists)):
            overlap = all_nutrient_lists[i] & all_nutrient_lists[j]
            overlap_count += len(overlap)
    return overlap_count

def _generate_final_reasons(product, covered_goals, def_matches, analysis, data):
    """Generate dynamic reasons for final recommendation per product."""
    reasons = []
    nf = product.get('nutrition_facts', {})
    ing_names = _get_ingredient_names(product)

    # Reason 1: Which health goals this product covers
    if covered_goals:
        reasons.append(f"선택한 건강 목표({', '.join(covered_goals)})를 충족합니다.")

    # Reason 2: Which deficiency nutrients it replenishes
    actual_defs = [d for d in def_matches if _product_covers_deficiency(product, d)]
    if actual_defs:
        reasons.append(f"부족 가능 영양소({', '.join(actual_defs[:3])})를 보충할 수 있습니다.")

    # Reason 3: Complex product (multiple nutrients together)
    nutrient_set = _get_nutrients_set(product)
    func_count = len(product.get('functional_ingredients', []))
    total_count = len(nutrient_set) + func_count
    if total_count >= 4:
        present = []
        for k in sorted(nutrient_set):
            display_map = {
                'vitamin_c_mg': '비타민C', 'vitamin_d_ug': '비타민D',
                'calcium_mg': '칼슘', 'magnesium_mg': '마그네슘',
                'zinc_mg': '아연', 'iron_mg': '철분',
                'epa_dha_mg': 'EPA/DHA', 'lutein_mg': '루테인',
                'probiotics_cfu': '프로바이오틱스', 'folic_acid_ug': '엽산',
            }
            present.append(display_map.get(k, k))
        reasons.append(f"복합 제품으로 {', '.join(present[:4])} 등을 동시에 보충할 수 있습니다.")

    # Reason 4: Lifestyle fit
    try:
        sleep = float(data.get('sleep_hours', 7))
    except:
        sleep = 7
    outdoor = data.get('outdoor_activity', '보통')
    veg = data.get('vegetable_intake', '보통')
    lifestyle_fits = []
    if sleep < 6 and (nf.get('magnesium_mg', 0) > 0 or nf.get('folic_acid_ug', 0) > 0):
        lifestyle_fits.append('수면 부족')
    if outdoor == '낮음' and nf.get('vitamin_d_ug', 0) > 0:
        lifestyle_fits.append('야외활동 부족')
    if veg == '부족' and nf.get('vitamin_c_mg', 0) > 0:
        lifestyle_fits.append('채소 섭취 부족')
    if lifestyle_fits:
        reasons.append(f"생활습관 분석({', '.join(lifestyle_fits)}) 결과와 일치합니다.")

    # Reason 5: Functional claim
    for ing in product.get('functional_ingredients', []):
        claim = ing.get('functional_claim', '')
        if claim and str(claim) != 'nan':
            reasons.append(f"{claim} 기능성 원료({ing['ingredient_name']})를 포함합니다.")
            break

    seen = set()
    unique = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique[:5]

def _score_product_for_goals(product, health_goals, analysis, data):
    """Score a product against all health goals and return combined candidate info."""
    best_total = 0
    best_scores = {}
    best_def_matches = []
    covered_goals = []

    for goal in health_goals:
        temp_data = dict(data)
        temp_data['health_goal'] = goal
        total, scores, def_matches = calculate_recommendation_score(product, analysis, temp_data)

        if total > best_total:
            best_total = total
            best_scores = scores
            best_def_matches = def_matches

        if _product_covers_goal(product, goal):
            covered_goals.append(goal)

    all_defs = list(set(best_def_matches + [
        d for d in analysis.get('deficiencies', [])
        if _product_covers_deficiency(product, d) and d not in best_def_matches
    ]))

    return best_total, best_scores, best_def_matches, covered_goals, all_defs


def _greedy_select(health_goals, candidates, analysis):
    """Select products greedily to cover health goals, evaluating by criteria order:
    1) new goals covered 2) new defs covered 3) multi-goal bonus
    4) -overlap (lower better) 5) original score (tiebreaker)
    Returns (selected_candidates, remaining_goals_list)
    """
    remaining_goals = set(health_goals)
    selected = []
    used_ids = set()

    while remaining_goals and len(selected) < 3:
        best = None
        best_criteria = (-1, -1, 0, float('inf'), -1)

        for c in candidates:
            if c['product_id'] in used_ids:
                continue
            new_goals = [g for g in remaining_goals if g in c['covered_goals']]
            if not new_goals:
                continue

            new_goals_count = len(new_goals)
            new_defs = [d for d in c['covered_deficiencies']]
            new_defs_count = len(new_defs)
            is_multi = len(c['covered_goals']) >= 2

            combo_products = [x['product'] for x in selected] + [c['product']]
            overlap = _compute_nutrient_overlap(combo_products)

            criteria = (new_goals_count, new_defs_count, 1 if is_multi else 0, -overlap, c['total_score'])

            if criteria > best_criteria:
                best = c
                best_criteria = criteria

        if best is None:
            break

        selected.append(best)
        used_ids.add(best['product_id'])
        remaining_goals -= set(best['covered_goals'])

    return selected, list(remaining_goals)


def generate_final_recommendation(analysis, data):
    """
    Find the best COMBINATION of up to 3 products that most efficiently
    covers the user's selected health goals and deficiency nutrients.
    Uses original engine scores (no bonuses, never exceeds 100).

    STEP 1: Score all products using existing engine.
    STEP 2: Classify by score thresholds (80+, 70-79, 60-69, <60 excluded).
    STEP 3: Greedy selection prioritizing goal coverage.
    """
    health_goals = data.get('health_goals', [data.get('health_goal', '')])
    if isinstance(health_goals, str):
        health_goals = [health_goals]
    if not health_goals:
        return [], {}

    MIN_SCORE = 60

    # STEP 1 + 2: Score all products and classify candidates
    candidate_map = {}
    goals_with_zero_candidates = []

    for goal in health_goals:
        goal_has_candidate = False
        for product in product_list:
            if product.get('status') == '판매중지':
                continue
            pid = product['product_id']
            total, scores, def_matches, covered_goals, all_defs = _score_product_for_goals(
                product, [goal], analysis, data
            )
            if total < MIN_SCORE:
                continue

            goal_has_candidate = True

            if pid not in candidate_map:
                candidate_map[pid] = {
                    'product': product,
                    'product_id': pid,
                    'total_score': total,
                    'score_breakdown': scores,
                    'deficiency_matches': def_matches,
                    'covered_goals': covered_goals,
                    'covered_deficiencies': all_defs,
                }
            else:
                existing = candidate_map[pid]
                if total > existing['total_score']:
                    existing['total_score'] = total
                    existing['score_breakdown'] = scores
                    existing['deficiency_matches'] = def_matches
                if product['product_id'] == pid:
                    for g in covered_goals:
                        if g not in existing['covered_goals']:
                            existing['covered_goals'].append(g)
                    for d in all_defs:
                        if d not in existing['covered_deficiencies']:
                            existing['covered_deficiencies'].append(d)

        if not goal_has_candidate:
            goals_with_zero_candidates.append(goal)

    # Re-score all candidates: run against ALL goals to get accurate covered_goals and total score
    for pid in candidate_map:
        product = products_by_id[pid]
        full_total, full_scores, full_def_matches, full_covered_goals, full_all_defs = _score_product_for_goals(
            product, health_goals, analysis, data
        )
        entry = candidate_map[pid]
        entry['total_score'] = full_total
        entry['score_breakdown'] = full_scores
        entry['deficiency_matches'] = full_def_matches
        entry['covered_goals'] = full_covered_goals
        entry['covered_deficiencies'] = full_all_defs

    candidates = list(candidate_map.values())

    if not candidates:
        combo_info = {
            'total_goals': len(health_goals),
            'covered_goals_count': 0,
            'uncovered_goals': health_goals,
            'uncovered_goal_warnings': [
                f"현재 보유한 제품 데이터에서는 {g}를 충분히 충족하는 제품이 없습니다."
                for g in goals_with_zero_candidates
            ],
            'medication_considered': data.get('medication', '') not in ['', '없음'],
        }
        return [], combo_info

    # Tier classification
    tier80 = [c for c in candidates if c['total_score'] >= 80]
    tier70 = [c for c in candidates if 70 <= c['total_score'] < 80]
    tier60 = [c for c in candidates if 60 <= c['total_score'] < 70]

    # Phase 1: Try 70+ pool first
    pool70 = tier80 + tier70
    selected, remaining_goals = _greedy_select(health_goals, pool70, analysis)

    # If not all goals covered, try with alternatives (60+)
    if remaining_goals:
        pool_all = pool70 + tier60
        selected2, remaining2 = _greedy_select(health_goals, pool_all, analysis)
        # Compare coverage: prefer the one with fewer remaining goals
        if len(remaining2) < len(remaining_goals):
            selected = selected2
            remaining_goals = remaining2
        elif len(remaining2) == len(remaining_goals) and selected2:
            # Same coverage, prefer smaller combination or higher scores
            if len(selected2) < len(selected) or (
                len(selected2) == len(selected) and
                sum(c['total_score'] for c in selected2) > sum(c['total_score'] for c in selected)
            ):
                selected = selected2
                remaining_goals = remaining2

    # Build result
    result = []
    for item in selected:
        product = item['product']
        reasons = _generate_final_reasons(
            product, item['covered_goals'], item['covered_deficiencies'],
            analysis, data
        )
        cautions = generate_caution_reasons(product, analysis, data)
        score = item['total_score']
        level = '적극 추천' if score >= 80 else ('추천' if score >= 70 else '대체 가능')

        result.append({
            'product': product,
            'total_score': score,
            'score_breakdown': item['score_breakdown'],
            'recommendationReasons': reasons,
            'cautionReasons': cautions,
            'deficiency_matches': item['covered_deficiencies'],
            'recommendation_level': level,
            'goals_covered': len(item['covered_goals']),
            'total_goals': len(health_goals),
            'covered_goal_names': item['covered_goals'],
            'covered_deficiency_names': item['covered_deficiencies'],
        })

    # Build uncovered goal warnings
    uncovered_warnings = []
    for g in goals_with_zero_candidates:
        uncovered_warnings.append(
            f"현재 보유한 제품 데이터에서는 {g}를 충분히 충족하는 제품이 없습니다."
        )
    for g in remaining_goals:
        if g not in goals_with_zero_candidates:
            uncovered_warnings.append(
                f"현재 보유한 제품 데이터에서는 {g}를 충분히 충족하는 제품이 없습니다."
            )

    combo_info = {
        'total_goals': len(health_goals),
        'covered_goals_count': len(set(health_goals) - set(remaining_goals)),
        'covered_goal_names': list(set(health_goals) - set(remaining_goals)),
        'uncovered_goals': remaining_goals,
        'uncovered_goal_warnings': uncovered_warnings,
        'medication_considered': data.get('medication', '') not in ['', '없음'],
    }

    return result, combo_info

# ──────────────────────────────────────────────

def generateAiHealthReport(analysis, recommendations, final_recommendation, combination_summary, data):
    try:
        nutritional_goals = data.get("health_goals", [data.get("health_goal", "")])
        if isinstance(nutritional_goals, str):
            nutritional_goals = [nutritional_goals]

        ml_predictions = []
        if nutrition_pipeline is not None:
            try:
                user_input_bmi = 23
                if data.get("height") and data.get("weight"):
                    try:
                        h = float(data["height"]) / 100
                        w = float(data["weight"])
                        user_input_bmi = round(w / (h * h), 1)
                    except:
                        pass
                ml_input = {
                    "age": float(data.get("age", 30)),
                    "sex": data.get("gender", "남성"),
                    "bmi": user_input_bmi,
                    "waist": float(data.get("waist", 0)) or 0,
                    "sbp": float(data.get("sbp", 0)) or 120,
                    "dbp": float(data.get("dbp", 0)) or 80,
                    "glucose": float(data.get("glucose", 0)) or 0,
                    "hdl": float(data.get("hdl", 0)) or 0,
                    "ldl": float(data.get("ldl", 0)) or 0,
                    "triglyceride": float(data.get("triglyceride", 0)) or 0,
                    "sleep_hours": float(data.get("sleep_hours", 7)),
                    "heart_age": float(data.get("heart_age", 0)) or 0,
                    "lifestyle_risk_score": float(data.get("lifestyle_risk_score", 0)) or 0,
                    "cardio_risk_score": float(data.get("cardio_risk_score", 0)) or 0,
                    "metabolic_syndrome_count": float(data.get("metabolic_syndrome_count", 0)) or 0,
                    "exercise_score": float(data.get("exercise_score", 0)) or 0,
                    "sleep_score": float(data.get("sleep_score", 0)) or 0,
                    "stress_score": float(data.get("stress_score", 0)) or 0,
                    "smoking_score": float(data.get("smoking_score", 0)) or 0,
                    "drinking_score": float(data.get("drinking_score", 0)) or 0,
                    "cardio_risk_encoded": float(data.get("cardio_risk_encoded", 0)) or 0,
                    "ascvd_risk_encoded": float(data.get("ascvd_risk_encoded", 0)) or 0,
                    "medication_flag": float(data.get("medication_flag", 0)) or 0,
                    "takes_antihypertensive": float(data.get("takes_antihypertensive", 0)) or 0,
                    "takes_diabetes_med": float(data.get("takes_diabetes_med", 0)) or 0,
                    "takes_lipid_med": float(data.get("takes_lipid_med", 0)) or 0,
                    "smoking": data.get("smoking", "비흡연"),
                    "drinking": data.get("alcohol", "비음주"),
                    "exercise": data.get("exercise_freq", "거의 안함"),
                    "stress_level": data.get("stress_level", "보통"),
                    "subjective_health": data.get("subjective_health", "보통"),
                    "hypertension": data.get("hypertension", "없음"),
                    "diabetes": data.get("diabetes", "없음"),
                    "dyslipidemia": data.get("dyslipidemia", "없음"),
                    "metabolic_syndrome": data.get("metabolic_syndrome", "아니요"),
                    "family_history": data.get("family_history", "없음"),
                    "medication": data.get("medication", "없음"),
                    "cardio_risk": data.get("cardio_risk", "LOW"),
                    "menopause": data.get("menopause", "NO"),
                    "ascvd_risk_level": data.get("ascvd_risk_level", "LOW"),
                }
                raw_results = predict_nutrition_risk(nutrition_pipeline, ml_input, threshold=0.3)
                selected = [r for r in raw_results if r["selected"]]
                selected_sorted = sorted(selected, key=lambda x: x["probability"], reverse=True)
                for r in selected_sorted[:5]:
                    ml_predictions.append({
                        "name": r["name"],
                        "probability": round(r["probability"] * 100, 1),
                        "label": r["label"],
                    })
            except Exception as e:
                print(f"AiHealthReport ML error: {e}")

        deficiencies = analysis.get("deficiencies", [])
        reasons = analysis.get("reasons", [])
        caution_ingredients = analysis.get("caution_ingredients", [])
        caution_reasons = analysis.get("caution_reasons", [])

        goals_str = ", ".join(nutritional_goals) if nutritional_goals else "일반"

        if ml_predictions:
            top_def_names = [p["name"] for p in ml_predictions[:3]]
            ml_summary = f"사용자의 PHR 데이터와 생활습관을 종합 분석한 결과, {", ".join(top_def_names)} 보충 고려 가능성이 높게 나타났습니다."
        else:
            def_names = deficiencies[:3] if deficiencies else ["특별한"]
            ml_summary = f"사용자의 PHR 데이터와 생활습관을 종합 분석한 결과, {", ".join(def_names)} 관련 영양소 부족 가능성이 확인되었습니다."

        rec_product_names = []
        if final_recommendation:
            for r in final_recommendation[:3]:
                p = r.get("product", {})
                rec_product_names.append(p.get("product_name", "이름 없음"))
        if not rec_product_names:
            groups = recommendations.get("health_goal_groups", [])
            for g in groups[:2]:
                for r in g.get("products", [])[:2]:
                    p = r.get("product", {})
                    rec_product_names.append(p.get("product_name", "이름 없음"))

        combined_reasons = []
        if final_recommendation:
            for r in final_recommendation[:2]:
                combined_reasons.extend(r.get("recommendationReasons", [])[:2])
        if not combined_reasons:
            groups = recommendations.get("health_goal_groups", [])
            for g in groups[:2]:
                for r in g.get("products", [])[:1]:
                    combined_reasons.extend(r.get("recommendationReasons", [])[:2])

        lifestyle_notes = []
        sleep = data.get("sleep_hours", 7)
        try:
            sleep = float(sleep)
        except:
            sleep = 7
        outdoor = data.get("outdoor_activity", "보통")
        veg = data.get("vegetable_intake", "보통")
        fish = data.get("fish_intake", "보통")
        exercise = data.get("exercise_freq", "거의 안함")
        stress = data.get("stress_level", "보통")

        if outdoor == "낮음" or outdoor == "거의 안함":
            lifestyle_notes.append("야외활동이 낮은 경우 비타민D 보충이 도움이 될 수 있습니다.")
        if fish == "낮음" or fish == "거의 안 먹음":
            lifestyle_notes.append("생선 섭취가 부족한 경우 EPA/DHA 보충이 도움이 될 수 있습니다.")
        if sleep < 6:
            lifestyle_notes.append("수면 부족이 있는 경우 Magnesium, Vitamin B군 보충 고려 가능성이 있습니다.")
        if "흔함" in str(exercise) or "안함" in str(exercise):
            lifestyle_notes.append("운동 부족은 골격력 저하와 관련될 수 있으며 비타민D, 칼슘 보충이 도움이 될 수 있습니다.")

        med = data.get("medication", "")
        caution_lines = []
        if med and med not in ["", "없음"]:
            caution_lines.append(f"복용 약물({med})과의 상호작용을 고려하여 제품을 선택했습니다.")
        allergy = data.get("allergies", "")
        if allergy and allergy not in ["", "없음"]:
            caution_lines.append(f"알레르기 정보({allergy})를 고려하여 추천했습니다.")
        if caution_reasons:
            caution_lines.extend(caution_reasons[:2])
        if not caution_lines:
            caution_lines.append("복용 약물, 알레르기, 제품별 섭취 주의사항을 확인한 뒤 섭취하세요.")

        report = {
            "ml_summary": ml_summary,
            "ml_predictions": ml_predictions[:5],
            "deficiencies": deficiencies[:5],
            "reasons": reasons[:3],
            "rec_product_names": rec_product_names[:3],
            "rec_reasons": combined_reasons[:3],
            "lifestyle_notes": lifestyle_notes[:3],
            "caution_notes": caution_lines[:3],
            "goals": nutritional_goals,
        }
        return report
    except Exception as e:
        print(f"generateAiHealthReport error: {e}")
        return {
            "ml_summary": "사용자의 데이터를 분석하여 개별화된 건강 리포트를 준비하는 중입니다.",
            "ml_predictions": [],
            "deficiencies": analysis.get("deficiencies", [])[:3] if analysis else [],
            "reasons": [],
            "rec_product_names": [],
            "rec_reasons": [],
            "lifestyle_notes": [],
            "caution_notes": [],
            "goals": [],
        }

# 3. Flask routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health-goals')
def api_health_goals():
    return jsonify(goal_config)

@app.route('/api/products')
def api_products():
    return jsonify(product_list)

@app.route('/api/product/<pid>')
def api_product(pid):
    p = products_by_id.get(pid)
    if p:
        return jsonify(p)
    return jsonify({'error': 'not found'}), 404

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    body = request.get_json()
    data = dict(body.get('direct_input', body))

    try:
        if nutrition_pipeline is not None:
            analysis = predict_deficiencies_ml(data)
        else:
            analysis = analyze_deficiencies(data)
            analysis['analysis_source'] = 'rule_based'
            analysis['ml_predictions'] = []
    except Exception:
        analysis = analyze_deficiencies(data)
        analysis['analysis_source'] = 'rule_based'
        analysis['ml_predictions'] = []
    print("analysis_source:", analysis.get("analysis_source"))
    print("ml_predictions:", analysis.get("ml_predictions"))
    recommendations = generate_recommendations(analysis, data)
    final_recommendation, combination_summary = generate_final_recommendation(analysis, data)

    ai_health_report = generateAiHealthReport(analysis, recommendations, final_recommendation, combination_summary, data)

    health_summary = {
        'has_checkup': data.get('health_checkup_data', False),
        'has_wearable': data.get('wearable_data', False),
        'age': data.get('age', None),
        'gender': data.get('gender', '남성'),
        'health_goal': data.get('health_goal', ''),
        'bmi': None,
    }
    if data.get('height') and data.get('weight'):
        try:
            h = float(data['height']) / 100
            w = float(data['weight'])
            health_summary['bmi'] = round(w / (h * h), 1)
        except:
            pass

    # Convert NaN values in recommendations for JSON serialization
    def clean_val(v):
        if isinstance(v, float) and (pd.isna(v) or np.isnan(v)):
            return None
        if isinstance(v, dict):
            return {k: clean_val(v) for k, v in v.items()}
        if isinstance(v, list):
            return [clean_val(x) for x in v]
        return v

    groups_clean = []
    for group in recommendations.get('health_goal_groups', []):
        cleaned_products = []
        for r in group['products']:
            rec = {
                'product': clean_val(r['product']),
                'total_score': r['total_score'],
                'score_breakdown': r['score_breakdown'],
                'recommendationReasons': r['recommendationReasons'],
                'cautionReasons': r['cautionReasons'],
                'deficiency_matches': r['deficiency_matches'],
                'recommendation_level': r['recommendation_level'],
            }
            cleaned_products.append(rec)
        groups_clean.append({
            'goal': group['goal'],
            'deficiencies': group['deficiencies'],
            'products': cleaned_products,
        })

    final_clean = []
    if final_recommendation:
        for r in final_recommendation:
            r['product'] = clean_val(r['product'])
            final_clean.append(r)

    return jsonify({
        'health_summary': health_summary,
        'analysis': analysis,
        'recommendations': groups_clean,
        'final_recommendation': final_clean,
        'combination_summary': combination_summary,
        'ai_health_report': ai_health_report,
        'debug_pipeline_loaded': nutrition_pipeline is not None,
    })

# ── Keep existing health prediction routes ──
print("건강 데이터 로딩 중...")
df_health = pd.read_csv('SYNTHETIC_PHR_MEDICAL_v4_FINAL_KR_HEADER.csv', encoding='utf-8')
df_health.columns = df_health.iloc[0]
df_health = df_health.iloc[1:].reset_index(drop=True)

eng_cols = [
    'AGE', 'SEX', 'SMOKING', 'DRINKING', 'EXERCISE', 'SLEEP_HOURS',
    'STRESS_LEVEL', 'SUBJECTIVE_HEALTH', 'LIFESTYLE_RISK_SCORE', 'BMI', 'WAIST',
    'SBP', 'DBP', 'GLUCOSE', 'HDL', 'LDL', 'TRIGLYCERIDE',
    'HYPERTENSION', 'DIABETES', 'DYSLIPIDEMIA', 'METABOLIC_SYNDROME_COUNT',
    'METABOLIC_SYNDROME', 'FAMILY_HISTORY', 'MEDICATION',
    'TAKES_ANTIHYPERTENSIVE', 'TAKES_DIABETES_MED', 'TAKES_LIPID_MED',
    'CARDIO_RISK_SCORE', 'CARDIO_RISK', 'MENOPAUSE', 'ASCVD_RISK_LEVEL', 'HEART_AGE'
]
df_health.columns = eng_cols

for col in ['AGE', 'SLEEP_HOURS', 'LIFESTYLE_RISK_SCORE', 'BMI', 'WAIST',
            'SBP', 'DBP', 'GLUCOSE', 'HDL', 'LDL', 'TRIGLYCERIDE',
            'CARDIO_RISK_SCORE', 'HEART_AGE', 'METABOLIC_SYNDROME_COUNT',
            'TAKES_ANTIHYPERTENSIVE', 'TAKES_DIABETES_MED', 'TAKES_LIPID_MED']:
    df_health[col] = pd.to_numeric(df_health[col], errors='coerce')

print("모델 로딩 중...")
models = joblib.load('models/all_models.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
target_encoders = joblib.load('models/target_encoders.pkl')
feature_info = joblib.load('models/feature_info.pkl')

categorical_cols = feature_info['categorical_cols']
all_features = feature_info['all_features']

KOREAN_NAMES = {
    'AGE': '나이', 'SEX': '성별', 'SMOKING': '흡연', 'DRINKING': '음주',
    'EXERCISE': '운동', 'SLEEP_HOURS': '수면시간', 'STRESS_LEVEL': '스트레스',
    'SUBJECTIVE_HEALTH': '주관적 건강', 'LIFESTYLE_RISK_SCORE': '생활습관 위험점수',
    'BMI': 'BMI', 'WAIST': '허리둘레', 'SBP': '수축기혈압', 'DBP': '이완기혈압',
    'GLUCOSE': '공복혈당', 'HDL': 'HDL', 'LDL': 'LDL', 'TRIGLYCERIDE': '중성지방',
    'HYPERTENSION': '고혈압', 'DIABETES': '당뇨', 'DYSLIPIDEMIA': '이상지질혈증',
    'METABOLIC_SYNDROME': '대사증후군', 'METABOLIC_SYNDROME_COUNT': '대사증후군구성요소수',
    'FAMILY_HISTORY': '가족력', 'MEDICATION': '복약정보',
    'CARDIO_RISK': '심혈관위험등급', 'ASCVD_RISK_LEVEL': 'ASCVD위험도',
    'HEART_AGE': '심장나이', 'CARDIO_RISK_SCORE': '심혈관위험점수',
    'MENOPAUSE': '폐경여부'
}

PREDICTION_TARGETS = [
    {'key': 'hypertension', 'name': '고혈압', 'levels': ['정상', '고혈압 전단계', '고혈압']},
    {'key': 'diabetes', 'name': '당뇨', 'levels': ['정상', '전당뇨', '당뇨']},
    {'key': 'dyslipidemia', 'name': '이상지질혈증', 'levels': ['정상', '경계', '이상지질혈증']},
    {'key': 'metabolic_syndrome', 'name': '대사증후군', 'levels': ['아니오', '예']},
    {'key': 'cardio_risk', 'name': '심혈관위험등급', 'levels': ['HIGH', 'LOW', 'MEDIUM']},
    {'key': 'ascvd_risk', 'name': 'ASCVD위험도', 'levels': ['HIGH', 'LOW', 'MODERATE', 'VERY_HIGH']},
]

def get_chart_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data

@app.route('/api/analysis')
def api_analysis():
    charts = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_health['AGE'], bins=30, edgecolor='white', color='steelblue', alpha=0.8)
    ax.set_xlabel('나이'); ax.set_ylabel('빈도'); ax.set_title('연령 분포')
    charts['age_dist'] = get_chart_base64(fig)

    sex_counts = df_health['SEX'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sex_counts.values, labels=sex_counts.index.tolist(), autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'], startangle=90)
    ax.set_title('성별 분포')
    charts['sex_pie'] = get_chart_base64(fig)

    hyp_counts = df_health['HYPERTENSION'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_hyp = ['#ff9999', '#ffcc99', '#99cc99']
    bars = ax.bar(hyp_counts.index.tolist(), hyp_counts.values, color=colors_hyp[:len(hyp_counts)])
    for bar, val in zip(bars, hyp_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val} ({val/len(df_health)*100:.1f}%)', ha='center', fontsize=10)
    ax.set_title('고혈압 상태 분포'); ax.set_ylabel('인원 수')
    charts['hypertension'] = get_chart_base64(fig)

    dia_counts = df_health['DIABETES'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_dia = ['#99ccff', '#ffcc99', '#ff9999']
    bars = ax.bar(dia_counts.index.tolist(), dia_counts.values, color=colors_dia[:len(dia_counts)])
    for bar, val in zip(bars, dia_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val} ({val/len(df_health)*100:.1f}%)', ha='center', fontsize=10)
    ax.set_title('당뇨 상태 분포'); ax.set_ylabel('인원 수')
    charts['diabetes'] = get_chart_base64(fig)

    ms_counts = df_health['METABOLIC_SYNDROME'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(ms_counts.values, labels=ms_counts.index.tolist(), autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'], startangle=90)
    ax.set_title('대사증후군 유병률')
    charts['metabolic'] = get_chart_base64(fig)

    msc = df_health['METABOLIC_SYNDROME_COUNT'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(msc.index.tolist(), msc.values, color='mediumseagreen', alpha=0.8)
    for bar, val in zip(bars, msc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(val), ha='center', fontsize=10)
    ax.set_xlabel('대사증후군 구성요소 수'); ax.set_ylabel('인원 수'); ax.set_title('대사증후군 구성요소 수 분포')
    charts['ms_count'] = get_chart_base64(fig)

    cr_counts = df_health['CARDIO_RISK'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_cr = {'LOW': '#99cc99', 'MEDIUM': '#ffcc99', 'HIGH': '#ff9999'}
    bars = ax.bar(cr_counts.index.tolist(), cr_counts.values, color=[colors_cr.get(x, '#999999') for x in cr_counts.index.tolist()])
    for bar, val in zip(bars, cr_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val} ({val/len(df_health)*100:.1f}%)', ha='center', fontsize=10)
    ax.set_title('심혈관 위험등급 분포'); ax.set_ylabel('인원 수')
    charts['cardio_risk'] = get_chart_base64(fig)

    ascvd_counts = df_health['ASCVD_RISK_LEVEL'].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_ascvd = {'LOW': '#99cc99', 'MODERATE': '#ffcc99', 'HIGH': '#ff9966', 'VERY_HIGH': '#ff6666'}
    bars = ax.bar(ascvd_counts.index.tolist(), ascvd_counts.values, color=[colors_ascvd.get(x, '#999999') for x in ascvd_counts.index.tolist()])
    for bar, val in zip(bars, ascvd_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val} ({val/len(df_health)*100:.1f}%)', ha='center', fontsize=10)
    ax.set_title('ASCVD 위험도 분포'); ax.set_ylabel('인원 수')
    charts['ascvd_risk'] = get_chart_base64(fig)

    smoke_counts = df_health['SMOKING'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(smoke_counts.index.tolist(), smoke_counts.values, color=['#ff9999', '#99cc99', '#66b3ff'])
    for bar, val in zip(bars, smoke_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val} ({val/len(df_health)*100:.1f}%)', ha='center', fontsize=10)
    ax.set_title('흡연 상태 분포'); ax.set_ylabel('인원 수')
    charts['smoking'] = get_chart_base64(fig)

    stats = {
        'total_count': len(df_health),
        'avg_age': round(df_health['AGE'].mean(), 1),
        'avg_bmi': round(df_health['BMI'].mean(), 1),
        'avg_sbp': round(df_health['SBP'].mean(), 1),
        'avg_dbp': round(df_health['DBP'].mean(), 1),
        'avg_glucose': round(df_health['GLUCOSE'].mean(), 1),
        'avg_hdl': round(df_health['HDL'].mean(), 1),
        'avg_ldl': round(df_health['LDL'].mean(), 1),
        'avg_triglyceride': round(df_health['TRIGLYCERIDE'].mean(), 1),
        'avg_waist': round(df_health['WAIST'].mean(), 1),
        'avg_sleep': round(df_health['SLEEP_HOURS'].mean(), 1),
        'avg_heart_age': round(df_health['HEART_AGE'].mean(), 1),
        'hypertension_rate': round((df_health['HYPERTENSION'].isin(['고혈압', '고혈압 전단계']).sum() / len(df_health)) * 100, 1),
        'diabetes_rate': round((df_health['DIABETES'].isin(['당뇨', '전당뇨']).sum() / len(df_health)) * 100, 1),
        'ms_rate': round((df_health['METABOLIC_SYNDROME'] == '예').sum() / len(df_health) * 100, 1),
        'high_cardio_rate': round((df_health['CARDIO_RISK'] == 'HIGH').sum() / len(df_health) * 100, 1),
        'medium_cardio_rate': round((df_health['CARDIO_RISK'] == 'MEDIUM').sum() / len(df_health) * 100, 1),
        'male_count': int((df_health['SEX'] == '남성').sum()),
        'female_count': int((df_health['SEX'] == '여성').sum()),
        'age_min': int(df_health['AGE'].min()),
        'age_max': int(df_health['AGE'].max()),
    }

    display_df = df_health.head(100)[['AGE', 'SEX', 'BMI', 'SBP', 'DBP', 'GLUCOSE',
                                         'HYPERTENSION', 'DIABETES', 'METABOLIC_SYNDROME',
                                         'CARDIO_RISK', 'HEART_AGE']].copy()
    display_data = json.loads(display_df.to_json(orient='records', force_ascii=False))

    return jsonify({'charts': charts, 'stats': stats, 'data': display_data})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    input_data = request.get_json()
    record_index = input_data.get('record_index')
    if record_index is not None:
        record = df_health.iloc[int(record_index)].to_dict()
    else:
        record = {}
        for col in all_features:
            if col in input_data:
                record[col] = input_data[col]
    results = {}
    for target in PREDICTION_TARGETS:
        key = target['key']
        model_info = models[key]
        model = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features']
        input_vector = []
        for f in features:
            val = record.get(f, 0)
            if f in categorical_cols:
                try: val = label_encoders[f].transform([str(val)])[0]
                except: val = 0
            else:
                try: val = float(val)
                except: val = 0
            input_vector.append(val)
        input_array = np.array(input_vector).reshape(1, -1)
        num_cols = [c for c in features if c not in categorical_cols]
        if num_cols:
            idx_map = {f: i for i, f in enumerate(features)}
            num_idx = [idx_map[c] for c in num_cols]
            input_array[0, num_idx] = scaler.transform(input_array[0, num_idx].reshape(1, -1)).flatten()
        if key in ['heart_age', 'ms_count']:
            pred = model.predict(input_array)[0]
            results[key] = {'name': target['name'], 'prediction': round(float(pred), 1), 'type': 'regression'}
        else:
            pred_class = model.predict(input_array)[0]
            proba = model.predict_proba(input_array)[0]
            pred_label = target['levels'][int(pred_class)]
            confidence = float(proba[int(pred_class)])
            all_probs = {target['levels'][i]: float(proba[i]) for i in range(len(target['levels'])) if i < len(proba)}
            results[key] = {'name': target['name'], 'prediction': pred_label, 'confidence': round(confidence * 100, 1), 'probabilities': all_probs, 'type': 'classification'}
    selected_record = {}
    for col in all_features:
        if col in record:
            val = record[col]
            if col in categorical_cols and col in label_encoders:
                try: val = label_encoders[col].inverse_transform([int(val)])[0]
                except: pass
            selected_record[KOREAN_NAMES.get(col, col)] = val
    return jsonify({'predictions': results, 'record': selected_record, 'record_index': record_index})

@app.route('/api/record/<int:idx>')
def api_record(idx):
    if idx < 0 or idx >= len(df_health):
        return jsonify({'error': 'Index out of range'}), 404
    record = df_health.iloc[idx].to_dict()
    result = {}
    for col in all_features:
        val = record[col]
        if col in categorical_cols and col in label_encoders:
            try: val = label_encoders[col].inverse_transform([int(val)])[0]
            except: pass
        result[KOREAN_NAMES.get(col, col)] = val
    return jsonify({'record': result, 'index': idx})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

