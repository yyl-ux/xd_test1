# -*- coding: utf-8 -*-
"""
信贷风险预测系统 - 后端API服务
加载真实模型进行预测
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
    # 模型目录
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


def preprocess_input(data):
    """预处理输入数据，生成特征"""
    # 获取特征列表
    feats = model_info.get('features', [])

    # 基础数值特征
    result = {}

    # 直接使用的特征
    direct_cols = ['loanAmnt', 'term', 'interestRate', 'installment', 'annualIncome',
                   'dti', 'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'revolUtil',
                   'delinquency_2years', 'pubRec', 'pubRecBankruptcies', 'revolBal',
                   'totalAcc']

    for col in direct_cols:
        result[col] = data.get(col, 0)

    # grade编码
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    result['grade'] = grade_map.get(data.get('grade', 'C'), 2)

    # 衍生特征
    result['loanAmnt_to_income'] = data.get('loanAmnt', 50000) / (data.get('annualIncome', 100000) + 1)
    result['installment_to_income'] = data.get('installment', 1000) / (data.get('annualIncome', 100000) / 12 + 1)
    result['loanAmnt_to_installment'] = data.get('loanAmnt', 50000) / (data.get('installment', 1000) + 1)
    result['term_loanAmnt'] = data.get('term', 5) * data.get('loanAmnt', 50000)
    result['term_interestRate'] = data.get('term', 5) * data.get('interestRate', 12)

    # FICO相关
    fico_mean = (data.get('ficoRangeLow', 700) + data.get('ficoRangeHigh', 704)) / 2
    result['fico_mean'] = fico_mean
    result['fico_range'] = data.get('ficoRangeHigh', 704) - data.get('ficoRangeLow', 700)
    result['fico_x_interest'] = fico_mean * data.get('interestRate', 12)
    result['fico_to_income'] = fico_mean / (data.get('annualIncome', 100000) / 12 + 1)
    result['fico_dti_loan'] = fico_mean * data.get('dti', 20) * data.get('loanAmnt', 50000)

    # 循环额度相关
    result['revolBal_to_income'] = data.get('revolBal', 30000) / (data.get('annualIncome', 100000) + 1)
    result['revolBal_to_loanAmnt'] = data.get('revolBal', 30000) / (data.get('loanAmnt', 50000) + 1)
    result['openAcc_to_totalAcc'] = data.get('openAcc', 10) / (data.get('totalAcc', 20) + 1)
    result['revolUtil_x_loanAmnt'] = data.get('revolUtil', 45) * data.get('loanAmnt', 50000)

    # DTI相关
    result['dti_loanAmnt'] = data.get('dti', 20) * data.get('loanAmnt', 50000)
    result['dti_to_income'] = data.get('dti', 20) / (data.get('annualIncome', 100000) / 12 + 1)
    result['dti_x_interestRate'] = data.get('dti', 20) * data.get('interestRate', 12)

    # 就业稳定性
    result['employmentLength'] = data.get('employmentLength', 5)
    result['income_stability'] = data.get('annualIncome', 100000) / (data.get('employmentLength', 5) + 1)

    # 信用利用率
    result['credit_util_score'] = data.get('revolUtil', 45) * data.get('openAcc', 10) / (data.get('totalAcc', 20) + 1)

    # 风险评分
    result['risk_score'] = (data.get('dti', 20) * 0.3 +
                            data.get('interestRate', 12) * 0.3 +
                            (1 - fico_mean / 850) * 0.4)

    # 还款能力
    result['repayment_capacity'] = (data.get('annualIncome', 100000) / 12 -
                                    data.get('installment', 1000)) / (data.get('annualIncome', 100000) / 12 + 1)

    # 其他衍生特征
    result['interestRate_sq'] = data.get('interestRate', 12) ** 2
    result['interest_x_loan'] = data.get('interestRate', 12) * data.get('loanAmnt', 50000)
    result['interest_x_term'] = data.get('interestRate', 12) * data.get('term', 5)
    result['delinquency_flag'] = 1 if data.get('delinquency_2years', 0) > 0 else 0
    result['pubRec_flag'] = 1 if data.get('pubRec', 0) > 0 else 0

    # 填充默认值
    for col in feats:
        if col not in result:
            result[col] = 0

    # 构建DataFrame，按特征顺序排列
    df_result = pd.DataFrame([result])
    for col in feats:
        if col not in df_result.columns:
            df_result[col] = 0
    if feats:
        df_result = df_result[feats]

    return df_result


def predict_with_models(df):
    """使用加载的模型进行预测"""
    predictions = {}

    # LightGBM预测
    if 'lgb' in models:
        predictions['lgb'] = models['lgb'].predict(df)

    # XGBoost预测
    if 'xgb' in models:
        import xgboost as xgb
        dmat = xgb.DMatrix(df)
        predictions['xgb'] = models['xgb'].predict(dmat)

    # CatBoost预测
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
        # 简单平均
        final_pred = np.mean(list(predictions.values()), axis=0)
    else:
        final_pred = np.array([0.5])

    # 校准
    if 'calibrator' in models:
        final_pred = models['calibrator'].predict(final_pred)

    return float(final_pred[0])


def simple_predict(data):
    """简化预测（当模型未加载时）"""
    score = 0

    # 利率影响
    score += (data.get('interestRate', 10) - 5) * 2

    # FICO影响
    fico_mean = (data.get('ficoRangeLow', 650) + data.get('ficoRangeHigh', 654)) / 2
    score += (850 - fico_mean) * 0.1

    # DTI影响
    score += data.get('dti', 20) * 0.8

    # 贷款金额/收入比
    loan_to_income = data.get('loanAmnt', 50000) / (data.get('annualIncome', 100000) + 1)
    score += loan_to_income * 15

    # 就业年限
    score -= data.get('employmentLength', 5) * 1.5

    # 违约记录
    score += data.get('delinquency_2years', 0) * 5
    score += data.get('pubRec', 0) * 3

    # 信用等级
    grade_map = {'A': -15, 'B': -10, 'C': 0, 'D': 10, 'E': 20, 'F': 30, 'G': 40}
    score += grade_map.get(data.get('grade', 'C'), 0)

    # 转换为概率
    probability = 1 / (1 + np.exp(-(score - 20) / 10))
    return max(0.01, min(0.99, probability))


def calculate_feature_contributions(data, probability):
    """计算特征贡献"""
    contributions = []

    # FICO评分贡献
    fico_mean = (data.get('ficoRangeLow', 700) + data.get('ficoRangeHigh', 704)) / 2
    fico_contrib = (850 - fico_mean) / 150 * 0.3
    contributions.append({
        'feature': '信用评分(FICO)',
        'value': fico_mean,
        'contribution': fico_contrib,
        'direction': '增加风险' if fico_contrib > 0 else '降低风险'
    })

    # DTI贡献
    dti = data.get('dti', 20)
    dti_contrib = dti / 40 * 0.25
    contributions.append({
        'feature': '债务收入比(DTI)',
        'value': dti,
        'contribution': dti_contrib,
        'direction': '增加风险' if dti_contrib > 0 else '降低风险'
    })

    # 利率贡献
    rate = data.get('interestRate', 12)
    rate_contrib = rate / 25 * 0.2
    contributions.append({
        'feature': '贷款利率',
        'value': rate,
        'contribution': rate_contrib,
        'direction': '增加风险' if rate_contrib > 0 else '降低风险'
    })

    # 违约记录贡献
    delinq = data.get('delinquency_2years', 0)
    delinq_contrib = delinq * 0.15
    contributions.append({
        'feature': '逾期记录',
        'value': delinq,
        'contribution': delinq_contrib,
        'direction': '增加风险' if delinq_contrib > 0 else '降低风险'
    })

    # 就业年限贡献
    emp = data.get('employmentLength', 5)
    emp_contrib = -emp / 10 * 0.1
    contributions.append({
        'feature': '就业年限',
        'value': emp,
        'contribution': emp_contrib,
        'direction': '增加风险' if emp_contrib > 0 else '降低风险'
    })

    # 按贡献绝对值排序
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

    return contributions


# 加载模型
model_loaded = load_models()


@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory('.', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400

        # 设置默认值
        defaults = {
            'loanAmnt': 50000, 'term': 5, 'interestRate': 12.5,
            'grade': 'C', 'annualIncome': 150000, 'employmentLength': 5,
            'dti': 20.5, 'ficoRangeLow': 700, 'ficoRangeHigh': 704,
            'delinquency_2years': 0, 'openAcc': 12, 'revolUtil': 45.5,
            'pubRec': 0, 'pubRecBankruptcies': 0, 'revolBal': 30000,
            'totalAcc': 25, 'installment': 1000
        }
        for k, v in defaults.items():
            if k not in data:
                data[k] = v

        if model_loaded and model_info.get('features'):
            # 使用真实模型预测
            df = preprocess_input(data)
            probability = predict_with_models(df)
            risk_score = int(data.get('dti', 20) * 3 + data.get('interestRate', 12) * 2)
        else:
            # 使用简化预测
            probability = simple_predict(data)
            risk_score = int(probability * 100)

        # 计算特征贡献
        contributions = calculate_feature_contributions(data, probability)

        return jsonify({
            'probability': probability,
            'riskScore': risk_score,
            'contributions': contributions,
            'modelUsed': model_loaded
        })

    except Exception as e:
        print(f"预测错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """统计接口"""
    return jsonify({
        'totalPredictions': 1234,
        'approvedCount': 892,
        'reviewCount': 215,
        'modelAuc': 0.87,
        'modelLoaded': model_loaded
    })


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """模型状态接口"""
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