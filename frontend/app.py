# -*- coding: utf-8 -*-
"""
信贷风险预测系统 - 后端API服务
完整保留全部模型，仅懒加载防内存溢出，预测效果与本地一致
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# 模型容器，启动为空，首次请求再载入全部模型
models = {}
model_info = {}
model_loaded = False

# 申请记录存储
application_records = []
RECORDS_FILE = 'application_records.json'

def load_records():
    global application_records
    if os.path.exists(RECORDS_FILE):
        try:
            with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
                application_records = json.load(f)
        except:
            application_records = []

def save_records():
    with open(RECORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(application_records, f, ensure_ascii=False, indent=2)

# ====================== 懒加载全套原始模型 ======================
def load_models():
    global models, model_info, model_loaded
    if model_loaded:
        return True

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'output_v35')
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return False

    try:
        print(f"开始加载全套模型...")

        # 模型特征信息
        info_path = os.path.join(model_dir, 'model_info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                global model_info
                model_info = pickle.load(f)

        # 1. LightGBM
        lgb_path = os.path.join(model_dir, 'lgb_model.txt')
        if os.path.exists(lgb_path):
            import lightgbm as lgb
            models['lgb'] = lgb.Booster(model_file=lgb_path)

        # 2. XGBoost
        xgb_path = os.path.join(model_dir, 'xgb_model.json')
        if os.path.exists(xgb_path):
            import xgboost as xgb
            models['xgb'] = xgb.Booster()
            models['xgb'].load_model(xgb_path)

        # 3. CatBoost
        cat_path = os.path.join(model_dir, 'cat_model.cbm')
        if os.path.exists(cat_path):
            from catboost import CatBoostClassifier
            models['cat'] = CatBoostClassifier()
            models['cat'].load_model(cat_path)

        # 4. Stacking融合模型
        stack_path = os.path.join(model_dir, 'stacking_model.pkl')
        if os.path.exists(stack_path):
            with open(stack_path, 'rb') as f:
                models['stacking'] = pickle.load(f)

        # 5. 概率校准器
        calib_path = os.path.join(model_dir, 'calibrator.pkl')
        if os.path.exists(calib_path):
            with open(calib_path, 'rb') as f:
                models['calibrator'] = pickle.load(f)

        model_loaded = True
        print(f"全套模型加载完成，可用模型：{list(models.keys())}")
        return True
    except Exception as e:
        print(f"模型加载异常: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== 以下所有业务、特征、预测函数 完全保留你原版逻辑 ==========
def safe_float(value, default=0, min_val=None, max_val=None):
    try:
        v = float(value) if value is not None else default
        if np.isnan(v) or np.isinf(v):
            v = default
        if min_val is not None:
            v = max(v, min_val)
        if max_val is not None:
            v = min(v, max_val)
        return v
    except:
        return default

def safe_int(value, default=0, min_val=None, max_val=None):
    try:
        v = int(value) if value is not None else default
        if min_val is not None:
            v = max(v, min_val)
        if max_val is not None:
            v = min(v, max_val)
        return v
    except:
        return default

def calculate_derived_features(data):
    annual_income = safe_float(data.get('annualIncome'), default=100000, min_val=1000, max_val=100000000)
    loan_amnt = safe_float(data.get('loanAmnt'), default=50000, min_val=1000, max_val=100000000)
    term = safe_float(data.get('term'), default=5, min_val=0.25, max_val=30)
    employment_length = safe_int(data.get('employmentLength'), default=5, min_val=0, max_val=50)
    home_ownership = safe_int(data.get('homeOwnership'), default=0, min_val=0, max_val=3)
    credit_limit = safe_float(data.get('creditLimit'), default=50000, min_val=0, max_val=10000000)
    credit_used = safe_float(data.get('creditUsed'), default=15000, min_val=0, max_val=10000000)
    monthly_debt = safe_float(data.get('monthlyDebt'), default=3000, min_val=0, max_val=1000000)
    total_acc = safe_int(data.get('totalAcc'), default=15, min_val=1, max_val=100)
    open_acc = safe_int(data.get('openAcc'), default=8, min_val=1, max_val=50)
    delinquency = safe_int(data.get('delinquency'), default=0, min_val=0, max_val=10)
    pub_rec = safe_int(data.get('pubRec'), default=0, min_val=0, max_val=10)
    pub_rec_bankruptcies = safe_int(data.get('pubRecBankruptcies'), default=0, min_val=0, max_val=1)
    purpose = safe_int(data.get('purpose'), default=0, min_val=0, max_val=10)

    monthly_income = annual_income / 12
    dti = min((monthly_debt / monthly_income * 100), 200) if monthly_income > 0 else 100
    revol_util = min((credit_used / credit_limit * 100), 100) if credit_limit > 0 else 0

    fico_score = calculate_fico_score(
        annual_income, employment_length, home_ownership,
        revol_util, delinquency, pub_rec, pub_rec_bankruptcies,
        total_acc, open_acc
    )
    grade = calculate_grade(fico_score, dti, delinquency, pub_rec, pub_rec_bankruptcies)
    interest_rate = calculate_interest_rate(grade)
    installment = calculate_installment(loan_amnt, term, interest_rate)
    loan_to_income = min((loan_amnt / annual_income * 100), 500) if annual_income > 0 else 100
    risk_score = dti * 0.3 + interest_rate * 0.3 + (1 - fico_score / 850) * 0.4

    return {
        'annualIncome': annual_income,
        'loanAmnt': loan_amnt,
        'term': term,
        'employmentLength': employment_length,
        'homeOwnership': home_ownership,
        'purpose': purpose,
        'dti': round(dti, 2),
        'revolUtil': round(revol_util, 2),
        'revolBal': credit_used,
        'totalAcc': total_acc,
        'openAcc': open_acc,
        'delinquency_2years': delinquency,
        'pubRec': pub_rec,
        'pubRecBankruptcies': pub_rec_bankruptcies,
        'ficoRangeLow': fico_score - 2,
        'ficoRangeHigh': fico_score + 2,
        'ficoScore': fico_score,
        'grade': grade,
        'interestRate': interest_rate,
        'installment': installment,
        'loanToIncome': round(loan_to_income, 2),
        'riskScore': round(risk_score, 2)
    }

def calculate_fico_score(income, emp_len, home_own, revol_util, delinq, pub_rec, bankruptcies, total_acc, open_acc):
    score = 700
    if income >= 300000: score += 20
    elif income >= 200000: score += 10
    elif income >= 100000: score += 5
    elif income < 50000: score -= 10
    if emp_len >= 10: score += 15
    elif emp_len >= 5: score += 10
    elif emp_len >= 3: score += 5
    elif emp_len < 1: score -= 15
    if home_own == 2: score += 15
    elif home_own == 1: score += 8
    elif home_own == 0: score -= 5
    if revol_util <= 10: score += 15
    elif revol_util <= 30: score += 8
    elif revol_util <= 50: score += 0
    elif revol_util <= 70: score -= 15
    else: score -= 30
    if delinq >= 4: score -= 120
    elif delinq >= 3: score -= 90
    elif delinq >= 2: score -= 60
    elif delinq >= 1: score -= 35
    if pub_rec >= 3: score -= 100
    elif pub_rec >= 2: score -= 70
    elif pub_rec >= 1: score -= 45
    if bankruptcies >= 1: score -= 80
    if total_acc >= 10 and open_acc >= 5: score += 5
    elif total_acc < 3: score -= 10
    return max(350, min(850, int(score)))

def calculate_grade(fico, dti, delinq, pub_rec, bankruptcies):
    if bankruptcies >= 1 or pub_rec >= 3 or delinq >= 4: return 'G'
    if pub_rec >= 2 or delinq >= 3: return 'F'
    if pub_rec >= 1 or delinq >= 2: return 'E'
    score = 0
    if fico >= 780: score += 5
    elif fico >= 740: score += 4
    elif fico >= 700: score += 3
    elif fico >= 660: score += 2
    elif fico >= 620: score += 1
    if dti <= 20: score += 2
    elif dti <= 35: score += 1
    elif dti > 50: score -= 1
    if delinq == 0 and pub_rec == 0: score += 1
    if score >= 7: return 'A'
    if score >= 5: return 'B'
    if score >= 3: return 'C'
    if score >= 1: return 'D'
    return 'E'

def calculate_interest_rate(grade):
    grade_rates = {'A': 6.5, 'B': 8.5, 'C': 11.0, 'D': 14.5, 'E': 18.0, 'F': 22.0, 'G': 26.0}
    return grade_rates.get(grade, 12.0)

def calculate_installment(loan_amnt, term, rate):
    monthly_rate = rate / 100 / 12
    n_payments = term * 12
    if monthly_rate > 0 and n_payments > 0:
        installment = loan_amnt * monthly_rate * (1 + monthly_rate) ** n_payments / ((1 + monthly_rate) ** n_payments - 1)
    else:
        installment = loan_amnt / max(n_payments, 1)
    return round(installment, 2)

def build_model_features(derived_data):
    feats = model_info.get('features', [])
    direct_cols = ['loanAmnt', 'term', 'interestRate', 'installment', 'annualIncome',
                   'dti', 'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'revolUtil',
                   'delinquency_2years', 'pubRec', 'pubRecBankruptcies', 'revolBal', 'totalAcc']
    result = {col: derived_data.get(col, 0) for col in direct_cols}
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    result['grade'] = grade_map.get(derived_data.get('grade', 'C'), 2)

    loan_amnt = derived_data.get('loanAmnt', 50000)
    annual_income = derived_data.get('annualIncome', 100000)
    term = derived_data.get('term', 5)
    interest_rate = derived_data.get('interestRate', 12)
    dti = derived_data.get('dti', 20)
    installment = derived_data.get('installment', 1000)
    fico_low = derived_data.get('ficoRangeLow', 700)
    fico_high = derived_data.get('ficoRangeHigh', 704)
    revol_util = derived_data.get('revolUtil', 45)
    revol_bal = derived_data.get('revolBal', 30000)
    open_acc = derived_data.get('openAcc', 10)
    total_acc = derived_data.get('totalAcc', 20)
    emp_len = derived_data.get('employmentLength', 5)
    delinq = derived_data.get('delinquency_2years', 0)
    pub_rec = derived_data.get('pubRec', 0)
    fico_mean = (fico_low + fico_high) / 2

    result['loanAmnt_to_income'] = loan_amnt / (annual_income + 1)
    result['installment_to_income'] = installment / (annual_income / 12 + 1)
    result['loanAmnt_to_installment'] = loan_amnt / (installment + 1)
    result['term_loanAmnt'] = term * loan_amnt
    result['term_interestRate'] = term * interest_rate
    result['fico_mean'] = fico_mean
    result['fico_range'] = fico_high - fico_low
    result['fico_x_interest'] = fico_mean * interest_rate
    result['fico_to_income'] = fico_mean / (annual_income / 12 + 1)
    result['fico_dti_loan'] = fico_mean * dti * loan_amnt
    result['revolBal_to_income'] = revol_bal / (annual_income + 1)
    result['revolBal_to_loanAmnt'] = revol_bal / (loan_amnt + 1)
    result['openAcc_to_totalAcc'] = open_acc / (total_acc + 1)
    result['revolUtil_x_loanAmnt'] = revol_util * loan_amnt
    result['dti_loanAmnt'] = dti * loan_amnt
    result['dti_to_income'] = dti / (annual_income / 12 + 1)
    result['dti_x_interestRate'] = dti * interest_rate
    result['employmentLength'] = emp_len
    result['income_stability'] = annual_income / (emp_len + 1)
    result['credit_util_score'] = revol_util * open_acc / (total_acc + 1)
    result['risk_score'] = dti * 0.3 + interest_rate * 0.3 + (1 - fico_mean / 850) * 0.4
    result['repayment_capacity'] = (annual_income / 12 - installment) / (annual_income / 12 + 1)
    result['interestRate_sq'] = interest_rate ** 2
    result['interest_x_loan'] = interest_rate * loan_amnt
    result['interest_x_term'] = interest_rate * term
    result['delinquency_flag'] = 1 if delinq > 0 else 0
    result['pubRec_flag'] = 1 if pub_rec > 0 else 0

    for k, v in result.items():
        if np.isnan(v) or np.isinf(v):
            result[k] = 0
    for col in feats:
        if col not in result:
            result[col] = 0
    df_result = pd.DataFrame([result])
    for col in feats:
        if col not in df_result.columns:
            df_result[col] = 0
    if feats:
        df_result = df_result[feats]
    return df_result

# 原版多模型融合预测逻辑 完全不变
def predict_with_models(df):
    predictions = {}
    if 'lgb' in models:
        predictions['lgb'] = models['lgb'].predict(df)
    if 'xgb' in models:
        import xgboost as xgb
        dmat = xgb.DMatrix(df)
        predictions['xgb'] = models['xgb'].predict(dmat)
    if 'cat' in models:
        predictions['cat'] = models['cat'].predict_proba(df)[:, 1]

    if 'stacking' in models and len(predictions) >= 2:
        stack_input = np.column_stack([
            predictions.get('lgb', np.array([0.5])),
            predictions.get('xgb', np.array([0.5])),
            predictions.get('cat', np.array([0.5]))
        ])
        final_pred = models['stacking'].predict_proba(stack_input)[:, 1]
    elif predictions:
        final_pred = np.mean(list(predictions.values()), axis=0)
    else:
        final_pred = np.array([0.5])

    if 'calibrator' in models:
        final_pred = models['calibrator'].predict(final_pred)
    return float(final_pred[0])

def rule_based_predict(derived_data):
    fico = derived_data.get('ficoScore', 700)
    dti = derived_data.get('dti', 20)
    delinq = derived_data.get('delinquency_2years', 0)
    pub_rec = derived_data.get('pubRec', 0)
    bankruptcies = derived_data.get('pubRecBankruptcies', 0)
    interest_rate = derived_data.get('interestRate', 12)
    loan_to_income = derived_data.get('loanToIncome', 50)
    emp_len = derived_data.get('employmentLength', 5)
    revol_util = derived_data.get('revolUtil', 45)
    score = 0
    score += max(0, (700 - fico)) * 0.1
    score += max(0, dti - 20) * 0.5
    score += (interest_rate - 8) * 0.5
    score += max(0, loan_to_income - 30) * 0.3
    score += delinq * 8
    score += pub_rec * 10
    score += bankruptcies * 15
    if emp_len >= 10: score -= 5
    elif emp_len >= 5: score -= 2
    elif emp_len < 2: score += 5
    if revol_util > 70: score += 5
    elif revol_util > 50: score += 3
    probability = 1 / (1 + np.exp(-(score - 15) / 10))
    return max(0.01, min(0.99, float(probability)))

def get_approval_suggestion(probability, derived_data):
    delinq = derived_data.get('delinquency_2years', 0)
    pub_rec = derived_data.get('pubRec', 0)
    bankruptcies = derived_data.get('pubRecBankruptcies', 0)
    fico = derived_data.get('ficoScore', 700)
    dti = derived_data.get('dti', 20)
    loan_to_income = derived_data.get('loanToIncome', 50)
    rule_risk = 0
    if bankruptcies >= 1: rule_risk += 0.40
    if pub_rec >= 3: rule_risk += 0.35
    elif pub_rec >= 2: rule_risk += 0.25
    elif pub_rec >= 1: rule_risk += 0.15
    if delinq >= 4: rule_risk += 0.30
    elif delinq >= 3: rule_risk += 0.20
    elif delinq >= 2: rule_risk += 0.12
    elif delinq >= 1: rule_risk += 0.06
    if fico < 550: rule_risk += 0.15
    elif fico < 620: rule_risk += 0.08
    if dti > 100: rule_risk += 0.10
    elif dti > 50: rule_risk += 0.05
    if loan_to_income > 150: rule_risk += 0.08
    elif loan_to_income > 100: rule_risk += 0.04
    combined_prob = min(0.99, probability + rule_risk)
    if bankruptcies >= 1:
        return (combined_prob, '建议拒绝（有破产记录）')
    if pub_rec >= 3:
        return (combined_prob, '建议拒绝（公共负面记录过多）')
    if delinq >= 4:
        return (combined_prob, '建议拒绝（逾期记录过多）')
    if combined_prob < 0.15:
        return (combined_prob, '自动通过')
    elif combined_prob < 0.30:
        return (combined_prob, '优先通过')
    elif combined_prob < 0.50:
        return (combined_prob, '人工复核')
    elif combined_prob < 0.70:
        return (combined_prob, '谨慎审批')
    else:
        return (combined_prob, '建议拒绝')

# 启动仅加载记录，不加载模型
load_records()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    global model_loaded
    try:
        # 首次请求才加载全套模型，后续直接复用
        if not model_loaded:
            load_models()

        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400

        derived_data = calculate_derived_features(data)
        # 真实模型预测，逻辑完全没变
        if model_loaded and model_info.get('features'):
            df = build_model_features(derived_data)
            probability = predict_with_models(df)
        else:
            probability = rule_based_predict(derived_data)

        combined_prob, suggestion = get_approval_suggestion(probability, derived_data)
        record = {
            'id': len(application_records) + 1,
            'applicantName': data.get('applicantName', ''),
            'applicantPhone': data.get('applicantPhone', ''),
            'annualIncome': data.get('annualIncome', 0),
            'employmentLength': data.get('employmentLength', 0),
            'homeOwnership': data.get('homeOwnership', 0),
            'loanAmnt': data.get('loanAmnt', 0),
            'purpose': data.get('purpose', 0),
            'term': data.get('term', 5),
            'creditLimit': data.get('creditLimit', 0),
            'creditUsed': data.get('creditUsed', 0),
            'monthlyDebt': data.get('monthlyDebt', 0),
            'totalAcc': data.get('totalAcc', 0),
            'openAcc': data.get('openAcc', 0),
            'delinquency': data.get('delinquency', 0),
            'pubRec': data.get('pubRec', 0),
            'pubRecBankruptcies': data.get('pubRecBankruptcies', 0),
            'probability': combined_prob,
            'ficoScore': derived_data['ficoScore'],
            'dti': derived_data['dti'],
            'revolUtil': derived_data['revolUtil'],
            'loanToIncome': derived_data['loanToIncome'],
            'riskScore': derived_data['riskScore'],
            'interestRate': derived_data['interestRate'],
            'grade': derived_data['grade'],
            'installment': derived_data['installment'],
            'suggestion': suggestion,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        application_records.append(record)
        save_records()

        return jsonify({
            'probability': combined_prob,
            'details': {
                'ficoScore': derived_data['ficoScore'],
                'dti': derived_data['dti'],
                'revolUtil': derived_data['revolUtil'],
                'loanToIncome': derived_data['loanToIncome'],
                'riskScore': derived_data['riskScore'],
                'interestRate': derived_data['interestRate'],
                'grade': derived_data['grade'],
                'installment': derived_data['installment'],
                'suggestion': suggestion
            }
        })
    except Exception as e:
        print(f"预测错误: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/records', methods=['GET'])
def get_records():
    risk_level = request.args.get('riskLevel', '')
    suggestion = request.args.get('suggestion', '')
    name = request.args.get('name', '')
    phone = request.args.get('phone', '')
    filtered_records = application_records.copy()
    if risk_level == 'low':
        filtered_records = [r for r in filtered_records if r['probability'] < 0.15]
    elif risk_level == 'medium':
        filtered_records = [r for r in filtered_records if 0.15 <= r['probability'] < 0.50]
    elif risk_level == 'high':
        filtered_records = [r for r in filtered_records if r['probability'] >= 0.50]
    if suggestion == '通过':
        filtered_records = [r for r in filtered_records if '通过' in r['suggestion']]
    elif suggestion == '复核':
        filtered_records = [r for r in filtered_records if '复核' in r['suggestion']]
    elif suggestion == '拒绝':
        filtered_records = [r for r in filtered_records if '拒绝' in r['suggestion']]
    if name:
        filtered_records = [r for r in filtered_records if name.lower() in r.get('applicantName', '').lower()]
    if phone:
        filtered_records = [r for r in filtered_records if phone in r.get('applicantPhone', '')]
    total = len(filtered_records)
    approved = len([r for r in filtered_records if '通过' in r['suggestion']])
    rejected = len([r for r in filtered_records if '拒绝' in r['suggestion']])
    avg_prob = sum(r['probability'] for r in filtered_records) / total if total > 0 else 0
    return jsonify({
        'records': filtered_records,
        'stats': {
            'total': total,
            'approved': approved,
            'rejected': rejected,
            'avgProbability': round(avg_prob, 4)
        }
    })

@app.route('/api/records/<int:record_id>', methods=['GET'])
def get_record_detail(record_id):
    record = next((r for r in application_records if r['id'] == record_id), None)
    if record:
        return jsonify(record)
    return jsonify({'error': '记录不存在'}), 404

@app.route('/api/records/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    global application_records
    application_records = [r for r in application_records if r['id'] != record_id]
    save_records()
    return jsonify({'success': True})

@app.route('/api/model/status', methods=['GET'])
def model_status():
    return jsonify({
        'loaded': model_loaded,
        'models': list(models.keys()),
        'featureCount': len(model_info.get('features', []))
    })

if __name__ == '__main__':
    # 关闭调试，读取平台端口，稳定低内存运行
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
    