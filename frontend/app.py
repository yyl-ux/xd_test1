# -*- coding: utf-8 -*-
"""
信贷风险预测系统 - 后端API服务
前端只负责展示，所有计算、特征衍生、业务逻辑都在后端处理
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# 模型存储
models = {}
model_info = {}


def load_models():
    """加载训练好的模型"""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'output_v35')

    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return False

    try:
        print(f"正在从 {model_dir} 加载模型...")

        # 加载模型信息
        info_path = os.path.join(model_dir, 'model_info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                global model_info
                model_info = pickle.load(f)
            print(f"加载模型信息: {len(model_info.get('features', []))} 个特征")

        # 加载LightGBM模型
        lgb_path = os.path.join(model_dir, 'lgb_model.txt')
        if os.path.exists(lgb_path):
            import lightgbm as lgb
            models['lgb'] = lgb.Booster(model_file=lgb_path)
            print("LightGBM模型加载成功")

        # 加载XGBoost模型
        xgb_path = os.path.join(model_dir, 'xgb_model.json')
        if os.path.exists(xgb_path):
            import xgboost as xgb
            models['xgb'] = xgb.Booster()
            models['xgb'].load_model(xgb_path)
            print("XGBoost模型加载成功")

        # 加载CatBoost模型
        cat_path = os.path.join(model_dir, 'cat_model.cbm')
        if os.path.exists(cat_path):
            from catboost import CatBoostClassifier
            models['cat'] = CatBoostClassifier()
            models['cat'].load_model(cat_path)
            print("CatBoost模型加载成功")

        # 加载Stacking模型
        stack_path = os.path.join(model_dir, 'stacking_model.pkl')
        if os.path.exists(stack_path):
            with open(stack_path, 'rb') as f:
                models['stacking'] = pickle.load(f)
            print("Stacking模型加载成功")

        # 加载校准器
        calib_path = os.path.join(model_dir, 'calibrator.pkl')
        if os.path.exists(calib_path):
            with open(calib_path, 'rb') as f:
                models['calibrator'] = pickle.load(f)
            print("校准器加载成功")

        if models:
            print(f"成功加载 {len(models)} 个模型组件")
            return True
        return False
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def safe_float(value, default=0, min_val=None, max_val=None):
    """安全转换浮点数，防止无穷大和异常值"""
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
    """安全转换整数"""
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
    """
    后端计算所有衍生特征
    用户只输入基础数据，所有计算都在这里完成
    """
    # 安全提取基础数据，设置合理的默认值和边界
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

    # ====== 计算衍生指标 ======

    # 1. 月收入
    monthly_income = annual_income / 12

    # 2. DTI (债务收入比) - 后端计算，限制在合理范围
    if monthly_income > 0:
        dti = min((monthly_debt / monthly_income * 100), 200)  # 限制最大200%
    else:
        dti = 100  # 无收入时DTI设为100%

    # 3. 额度使用率 - 后端计算，限制在合理范围
    if credit_limit > 0:
        revol_util = min((credit_used / credit_limit * 100), 100)  # 限制最大100%
    else:
        revol_util = 0

    # 4. FICO评分 - 后端根据用户信息计算
    fico_score = calculate_fico_score(
        annual_income, employment_length, home_ownership,
        revol_util, delinquency, pub_rec, pub_rec_bankruptcies,
        total_acc, open_acc
    )

    # 5. 信用等级 - 后端计算
    grade = calculate_grade(fico_score, dti, delinquency, pub_rec, pub_rec_bankruptcies)

    # 6. 利率 - 后端根据等级计算
    interest_rate = calculate_interest_rate(grade)

    # 7. 月供金额 - 后端计算
    installment = calculate_installment(loan_amnt, term, interest_rate)

    # 8. 贷款收入比 - 限制在合理范围
    if annual_income > 0:
        loan_to_income = min((loan_amnt / annual_income * 100), 500)  # 限制最大500%
    else:
        loan_to_income = 100

    # 9. 风险评分
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
    """后端计算FICO评分"""
    score = 700

    # 收入影响
    if income >= 300000: score += 20
    elif income >= 200000: score += 10
    elif income >= 100000: score += 5
    elif income < 50000: score -= 10

    # 就业稳定性
    if emp_len >= 10: score += 15
    elif emp_len >= 5: score += 10
    elif emp_len >= 3: score += 5
    elif emp_len < 1: score -= 15

    # 房产状态
    if home_own == 2: score += 15
    elif home_own == 1: score += 8
    elif home_own == 0: score -= 5

    # 额度使用率
    if revol_util <= 10: score += 15
    elif revol_util <= 30: score += 8
    elif revol_util <= 50: score += 0
    elif revol_util <= 70: score -= 15
    else: score -= 30

    # 逾期记录（影响大大增加）
    if delinq >= 4: score -= 120
    elif delinq >= 3: score -= 90
    elif delinq >= 2: score -= 60
    elif delinq >= 1: score -= 35

    # 公共负面记录（影响大大增加）
    if pub_rec >= 3: score -= 100
    elif pub_rec >= 2: score -= 70
    elif pub_rec >= 1: score -= 45

    # 破产记录（影响最大）
    if bankruptcies >= 1: score -= 80

    # 账户历史
    if total_acc >= 10 and open_acc >= 5: score += 5
    elif total_acc < 3: score -= 10

    return max(350, min(850, int(score)))


def calculate_grade(fico, dti, delinq, pub_rec, bankruptcies):
    """后端计算信用等级"""
    # 有重大负面记录直接降级
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
    """后端根据等级计算利率"""
    grade_rates = {'A': 6.5, 'B': 8.5, 'C': 11.0, 'D': 14.5, 'E': 18.0, 'F': 22.0, 'G': 26.0}
    return grade_rates.get(grade, 12.0)


def calculate_installment(loan_amnt, term, rate):
    """后端计算月供"""
    monthly_rate = rate / 100 / 12
    n_payments = term * 12
    if monthly_rate > 0 and n_payments > 0:
        installment = loan_amnt * monthly_rate * (1 + monthly_rate) ** n_payments / ((1 + monthly_rate) ** n_payments - 1)
    else:
        installment = loan_amnt / max(n_payments, 1)
    return round(installment, 2)


def build_model_features(derived_data):
    """构建模型需要的所有特征（包括衍生特征）"""
    feats = model_info.get('features', [])

    # 基础特征
    result = {}
    direct_cols = ['loanAmnt', 'term', 'interestRate', 'installment', 'annualIncome',
                   'dti', 'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'revolUtil',
                   'delinquency_2years', 'pubRec', 'pubRecBankruptcies', 'revolBal', 'totalAcc']

    for col in direct_cols:
        result[col] = derived_data.get(col, 0)

    # grade编码
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    result['grade'] = grade_map.get(derived_data.get('grade', 'C'), 2)

    # ====== 后端计算所有衍生特征 ======
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

    # 贷款金额相关比例（防止除零和无穷大）
    result['loanAmnt_to_income'] = loan_amnt / (annual_income + 1)
    result['installment_to_income'] = installment / (annual_income / 12 + 1)
    result['loanAmnt_to_installment'] = loan_amnt / (installment + 1)
    result['term_loanAmnt'] = term * loan_amnt
    result['term_interestRate'] = term * interest_rate

    # FICO相关
    result['fico_mean'] = fico_mean
    result['fico_range'] = fico_high - fico_low
    result['fico_x_interest'] = fico_mean * interest_rate
    result['fico_to_income'] = fico_mean / (annual_income / 12 + 1)
    result['fico_dti_loan'] = fico_mean * dti * loan_amnt

    # 循环额度相关
    result['revolBal_to_income'] = revol_bal / (annual_income + 1)
    result['revolBal_to_loanAmnt'] = revol_bal / (loan_amnt + 1)
    result['openAcc_to_totalAcc'] = open_acc / (total_acc + 1)
    result['revolUtil_x_loanAmnt'] = revol_util * loan_amnt

    # DTI相关
    result['dti_loanAmnt'] = dti * loan_amnt
    result['dti_to_income'] = dti / (annual_income / 12 + 1)
    result['dti_x_interestRate'] = dti * interest_rate

    # 就业稳定性
    result['employmentLength'] = emp_len
    result['income_stability'] = annual_income / (emp_len + 1)

    # 信用利用率
    result['credit_util_score'] = revol_util * open_acc / (total_acc + 1)

    # 风险评分
    result['risk_score'] = dti * 0.3 + interest_rate * 0.3 + (1 - fico_mean / 850) * 0.4

    # 还款能力
    result['repayment_capacity'] = (annual_income / 12 - installment) / (annual_income / 12 + 1)

    # 其他衍生特征
    result['interestRate_sq'] = interest_rate ** 2
    result['interest_x_loan'] = interest_rate * loan_amnt
    result['interest_x_term'] = interest_rate * term
    result['delinquency_flag'] = 1 if delinq > 0 else 0
    result['pubRec_flag'] = 1 if pub_rec > 0 else 0

    # 处理无穷大和NaN
    for k, v in result.items():
        if np.isnan(v) or np.isinf(v):
            result[k] = 0

    # 填充默认值
    for col in feats:
        if col not in result:
            result[col] = 0

    # 构建DataFrame
    df_result = pd.DataFrame([result])
    for col in feats:
        if col not in df_result.columns:
            df_result[col] = 0
    if feats:
        df_result = df_result[feats]

    return df_result


def predict_with_models(df):
    """使用模型预测"""
    predictions = {}

    if 'lgb' in models:
        predictions['lgb'] = models['lgb'].predict(df)

    if 'xgb' in models:
        import xgboost as xgb
        dmat = xgb.DMatrix(df)
        predictions['xgb'] = models['xgb'].predict(dmat)

    if 'cat' in models:
        predictions['cat'] = models['cat'].predict_proba(df)[:, 1]

    # 融合预测
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

    # 校准
    if 'calibrator' in models:
        final_pred = models['calibrator'].predict(final_pred)

    return float(final_pred[0])


def rule_based_predict(derived_data):
    """
    规则引擎预测（模型未加载时的备用方案）
    根据实际数据计算风险，而不是返回固定值
    """
    fico = derived_data.get('ficoScore', 700)
    dti = derived_data.get('dti', 20)
    delinq = derived_data.get('delinquency_2years', 0)
    pub_rec = derived_data.get('pubRec', 0)
    bankruptcies = derived_data.get('pubRecBankruptcies', 0)
    interest_rate = derived_data.get('interestRate', 12)
    loan_to_income = derived_data.get('loanToIncome', 50)
    emp_len = derived_data.get('employmentLength', 5)
    revol_util = derived_data.get('revolUtil', 45)

    # 基础风险分
    score = 0

    # FICO评分影响（FICO越低风险越高）
    score += max(0, (700 - fico)) * 0.1

    # DTI影响
    score += max(0, dti - 20) * 0.5

    # 利率影响
    score += (interest_rate - 8) * 0.5

    # 贷款收入比影响
    score += max(0, loan_to_income - 30) * 0.3

    # 逾期记录（重大影响）
    score += delinq * 8

    # 公共负面记录（重大影响）
    score += pub_rec * 10

    # 破产记录（最大影响）
    score += bankruptcies * 15

    # 就业年限（稳定就业降低风险）
    if emp_len >= 10:
        score -= 5
    elif emp_len >= 5:
        score -= 2
    elif emp_len < 2:
        score += 5

    # 额度使用率影响
    if revol_util > 70:
        score += 5
    elif revol_util > 50:
        score += 3

    # 转换为概率（使用sigmoid函数）
    probability = 1 / (1 + np.exp(-(score - 15) / 10))

    return max(0.01, min(0.99, float(probability)))


def get_approval_suggestion(probability, derived_data):
    """
    后端计算审批建议 - 综合考虑概率和负面记录
    返回: (综合概率, 审批建议)
    """
    delinq = derived_data.get('delinquency_2years', 0)
    pub_rec = derived_data.get('pubRec', 0)
    bankruptcies = derived_data.get('pubRecBankruptcies', 0)
    fico = derived_data.get('ficoScore', 700)
    dti = derived_data.get('dti', 20)
    loan_to_income = derived_data.get('loanToIncome', 50)

    # 计算综合风险分（结合模型概率和规则风险）
    rule_risk = 0

    # 破产记录（最严重）
    if bankruptcies >= 1:
        rule_risk += 0.40

    # 公共负面记录
    if pub_rec >= 3:
        rule_risk += 0.35
    elif pub_rec >= 2:
        rule_risk += 0.25
    elif pub_rec >= 1:
        rule_risk += 0.15

    # 逾期记录
    if delinq >= 4:
        rule_risk += 0.30
    elif delinq >= 3:
        rule_risk += 0.20
    elif delinq >= 2:
        rule_risk += 0.12
    elif delinq >= 1:
        rule_risk += 0.06

    # FICO评分过低
    if fico < 550:
        rule_risk += 0.15
    elif fico < 620:
        rule_risk += 0.08

    # DTI过高
    if dti > 100:
        rule_risk += 0.10
    elif dti > 50:
        rule_risk += 0.05

    # 贷款收入比过高
    if loan_to_income > 150:
        rule_risk += 0.08
    elif loan_to_income > 100:
        rule_risk += 0.04

    # 综合概率 = 模型概率 + 规则风险（上限0.99）
    combined_prob = min(0.99, probability + rule_risk)

    # 根据综合概率给出建议
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


# 加载模型
model_loaded = load_models()


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    预测接口
    前端只发送用户输入的基础数据
    后端负责：计算衍生特征、计算利率、计算评分、调用模型、返回结果
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400

        print(f"[DEBUG] 收到请求数据: {data}")

        # 后端计算所有衍生特征
        derived_data = calculate_derived_features(data)
        print(f"[DEBUG] 衍生特征: FICO={derived_data['ficoScore']}, DTI={derived_data['dti']}%, Grade={derived_data['grade']}, 利率={derived_data['interestRate']}%")

        # 预测
        if model_loaded and model_info.get('features'):
            df = build_model_features(derived_data)
            probability = predict_with_models(df)
            print(f"[DEBUG] 模型预测概率: {probability}")
        else:
            # 模型未加载时使用规则引擎
            probability = rule_based_predict(derived_data)
            print(f"[DEBUG] 规则引擎预测概率: {probability}")

        # 后端计算审批建议（返回综合概率和建议）
        combined_prob, suggestion = get_approval_suggestion(probability, derived_data)
        print(f"[DEBUG] 综合概率: {combined_prob}, 建议: {suggestion}")

        # 返回结果给前端（前端只负责展示）
        result = {
            'probability': combined_prob,  # 使用综合概率
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
        }
        print(f"[DEBUG] 返回结果: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"预测错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/status', methods=['GET'])
def model_status():
    return jsonify({
        'loaded': model_loaded,
        'models': list(models.keys()),
        'featureCount': len(model_info.get('features', []))
    })


if __name__ == '__main__':
    print("=" * 50)
    print("信贷风险预测系统 - 后端服务")
    print("=" * 50)
    print(f"模型加载状态: {'成功' if model_loaded else '失败，使用规则引擎'}")
    print("服务地址: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
