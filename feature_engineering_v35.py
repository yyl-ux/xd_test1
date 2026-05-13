# -*- coding: utf-8 -*-
"""
信贷评分卡 V35.0 - 融合优化版

结合两份代码的优点并优化：
1. 完整的特征工程（贷款金额比例、FICO相关、利率交互等）
2. 五折目标编码（防泄漏）
3. 缺失值指示器
4. 三模型融合（LightGBM + XGBoost + CatBoost）
5. Stacking + 加权平均双融合策略
6. 模型校准（Isotonic Regression）
7. 更精细的参数调优
"""

import os
import sys
import gc
import warnings
import logging
import pickle
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')


class Config:
    """配置类"""
    DATA_DIR = './data'
    OUTPUT_DIR = './output_v35'

    N_SPLITS = 5
    RANDOM_SEED = 2024

    # LightGBM参数 - 优化版
    LGB_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'max_depth': -1,
        'min_split_gain': 0.01,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 2024,
    }

    # XGBoost参数 - 优化版
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'max_depth': 8,
        'eta': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 30,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'gamma': 0.1,
        'scale_pos_weight': 4,
        'nthread': -1,
        'seed': 2024,
    }

    # CatBoost参数 - 优化版
    CAT_PARAMS = {
        'iterations': 5000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 5,
        'bagging_temperature': 1.0,
        'random_seed': 2024,
        'eval_metric': 'AUC',
        'verbose': 500,
        'thread_count': -1,
        'auto_class_weights': 'Balanced',
    }

    # 目标编码的类别特征
    TARGET_ENC_COLS = [
        'grade', 'subGrade', 'employmentLength', 'homeOwnership',
        'verificationStatus', 'purpose', 'regionCode',
        'initialListStatus', 'applicationType', 'term'
    ]


def setup_logging():
    """设置日志"""
    logger = logging.getLogger('v35')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(h)
    return logger


logger = setup_logging()


def calc_ks(y, p):
    """计算KS值"""
    fpr, tpr, _ = roc_curve(y, p)
    return max(abs(tpr - fpr))


def calc_psi(t, v, bins=10):
    """计算PSI值"""
    eps = 1e-10
    bp = np.percentile(np.sort(t), np.linspace(0, 100, bins + 1))
    bp[0], bp[-1] = -np.inf, np.inf
    tc = np.histogram(t, bins=bp)[0] + 1
    vc = np.histogram(v, bins=bp)[0] + 1
    tp, vp = tc / tc.sum(), vc / vc.sum()
    return np.sum((tp - vp) * np.log((tp + eps) / (vp + eps)))


def load_data():
    """加载数据"""
    logger.info('=' * 60)
    logger.info('[1/6] 加载数据')
    train = pd.read_csv(os.path.join(Config.DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(Config.DATA_DIR, 'testA.csv'))

    logger.info('训练集: %s' % str(train.shape))
    logger.info('测试集: %s' % str(test.shape))
    logger.info('目标分布:\n%s' % str(train['isDefault'].value_counts(normalize=True)))

    return train, test, train['isDefault'].copy(), train['id'].values, test['id'].values


def preprocess(train, test):
    """预处理"""
    logger.info('[2/6] 预处理')

    for df in [train, test]:
        # employmentLength处理
        emp_len_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years': 8, '9 years': 9, '10+ years': 10
        }
        df['employmentLength'] = df['employmentLength'].map(emp_len_map)

        # issueDate处理
        df['issueDate'] = pd.to_datetime(df['issueDate'], errors='coerce')
        min_date = df['issueDate'].min()
        df['issueDate_dt'] = (df['issueDate'] - min_date).dt.days
        df.drop('issueDate', axis=1, inplace=True)

        # earliesCreditLine处理
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        df['earliesCreditLine_month'] = df['earliesCreditLine'].str[:3].map(month_map)
        df['earliesCreditLine_year'] = df['earliesCreditLine'].str[-4:].astype(float)
        df['earliesCreditLine_ts'] = df['earliesCreditLine_year'] * 12 + df['earliesCreditLine_month']
        df.drop('earliesCreditLine', axis=1, inplace=True)

        # grade/subGrade标签编码
        for col in ['grade', 'subGrade']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    return train, test


def create_features(train, test, y):
    """特征工程 - 完整版"""
    logger.info('[3/6] 特征工程')

    # 合并处理避免编码不一致
    test['isDefault'] = -1
    all_data = pd.concat([train, test], axis=0, ignore_index=True)

    # ---- 缺失值指示器 ----
    missing_cols = [
        'employmentLength', 'dti', 'revolUtil', 'pubRecBankruptcies',
        'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9',
        'n10', 'n11', 'n12', 'n13', 'n14'
    ]
    for col in missing_cols:
        if col in all_data.columns:
            all_data[f'{col}_miss'] = all_data[col].isnull().astype(int)

    # ---- 填充缺失值 ----
    all_data['employmentLength'].fillna(-1, inplace=True)
    all_data['dti'].fillna(all_data['dti'].median(), inplace=True)
    all_data['revolUtil'].fillna(all_data['revolUtil'].median(), inplace=True)
    all_data['pubRecBankruptcies'].fillna(0, inplace=True)
    all_data['employmentTitle'].fillna(-1, inplace=True)
    all_data['postCode'].fillna(-1, inplace=True)
    all_data['title'].fillna(-1, inplace=True)

    # n系列用0填充
    n_cols = [f'n{i}' for i in range(15)]
    for c in n_cols:
        if c in all_data.columns:
            all_data[c].fillna(0, inplace=True)

    # ---- 衍生特征 ----
    # 贷款金额相关比例
    all_data['loanAmnt_to_income'] = all_data['loanAmnt'] / (all_data['annualIncome'] + 1)
    all_data['installment_to_income'] = all_data['installment'] / (all_data['annualIncome'] / 12 + 1)
    all_data['loanAmnt_to_installment'] = all_data['loanAmnt'] / (all_data['installment'] + 1)
    all_data['term_loanAmnt'] = all_data['term'] * all_data['loanAmnt']
    all_data['term_interestRate'] = all_data['term'] * all_data['interestRate']

    # FICO 相关
    all_data['fico_mean'] = (all_data['ficoRangeLow'] + all_data['ficoRangeHigh']) / 2
    all_data['fico_range'] = all_data['ficoRangeHigh'] - all_data['ficoRangeLow']
    all_data['fico_x_interest'] = all_data['fico_mean'] * all_data['interestRate']
    all_data['fico_to_income'] = all_data['fico_mean'] / (all_data['annualIncome'] / 12 + 1)
    all_data['fico_dti_loan'] = all_data['fico_mean'] * all_data['dti'] * all_data['loanAmnt']

    # 循环额度相关
    all_data['revolBal_to_income'] = all_data['revolBal'] / (all_data['annualIncome'] + 1)
    all_data['revolBal_to_loanAmnt'] = all_data['revolBal'] / (all_data['loanAmnt'] + 1)
    all_data['openAcc_to_totalAcc'] = all_data['openAcc'] / (all_data['totalAcc'] + 1)
    all_data['revolUtil_x_loanAmnt'] = all_data['revolUtil'] * all_data['loanAmnt']

    # DTI 相关
    all_data['dti_loanAmnt'] = all_data['dti'] * all_data['loanAmnt']
    all_data['dti_to_income'] = all_data['dti'] / (all_data['annualIncome'] / 12 + 1)
    all_data['dti_x_interestRate'] = all_data['dti'] * all_data['interestRate']

    # n 系列统计
    existing_n = [c for c in n_cols if c in all_data.columns]
    if existing_n:
        all_data['n_sum'] = all_data[existing_n].sum(axis=1)
        all_data['n_mean'] = all_data[existing_n].mean(axis=1)
        all_data['n_std'] = all_data[existing_n].std(axis=1)
        all_data['n_max'] = all_data[existing_n].max(axis=1)
        all_data['n_min'] = all_data[existing_n].min(axis=1)
        all_data['n_kurt'] = all_data[existing_n].kurtosis(axis=1)
        all_data['n_range'] = all_data['n_max'] - all_data['n_min']
        all_data['n_missing_cnt'] = all_data[
            [f'{c}_miss' for c in existing_n if f'{c}_miss' in all_data.columns]
        ].sum(axis=1)

    # 利率相关
    all_data['interestRate_sq'] = all_data['interestRate'] ** 2
    all_data['interest_x_loan'] = all_data['interestRate'] * all_data['loanAmnt']
    all_data['interest_x_term'] = all_data['interestRate'] * all_data['term']

    # 信用历史时长
    all_data['credit_history_len'] = all_data['issueDate_dt'] - all_data['earliesCreditLine_ts']
    all_data['credit_history_len'] = all_data['credit_history_len'].clip(lower=0)

    # 分期金额占贷款比
    all_data['installment_ratio'] = all_data['installment'] / (all_data['loanAmnt'] + 1)

    # 年收入与贷款期数
    all_data['income_term'] = all_data['annualIncome'] / (all_data['term'] + 1)

    # 违约记录标记
    all_data['delinquency_flag'] = (all_data['delinquency_2years'] > 0).astype(int)
    all_data['pubRec_flag'] = (all_data['pubRec'] > 0).astype(int)

    # 多特征交叉
    if 'grade' in all_data.columns:
        all_data['grade_interest'] = all_data['grade'] * all_data['interestRate']
    all_data['homeOwn_income'] = all_data['homeOwnership'] * all_data['annualIncome']
    all_data['purpose_interest'] = all_data['purpose'] * all_data['interestRate']
    all_data['verification_income'] = all_data['verificationStatus'] * all_data['annualIncome']

    # 借贷比
    all_data['total_loan_ratio'] = all_data['openAcc'] / (all_data['totalAcc'] + 1) * all_data['revolUtil']

    # ---- 新增优化特征 ----
    # 收入稳定性指标
    all_data['income_stability'] = all_data['annualIncome'] / (all_data['employmentLength'] + 1)

    # 信用利用率综合指标
    all_data['credit_util_score'] = all_data['revolUtil'] * all_data['openAcc'] / (all_data['totalAcc'] + 1)

    # 风险综合评分
    all_data['risk_score'] = (
        all_data['dti'] * 0.3 +
        all_data['interestRate'] * 0.3 +
        (1 - all_data['fico_mean'] / 850) * 0.4
    )

    # 还款能力指数
    all_data['repayment_capacity'] = (
        all_data['annualIncome'] / 12 - all_data['installment']
    ) / (all_data['annualIncome'] / 12 + 1)

    # 无穷值处理
    for c in all_data.select_dtypes(include=[np.number]).columns:
        if c not in ['id', 'isDefault']:
            all_data[c] = all_data[c].replace([np.inf, -np.inf], np.nan)
            if all_data[c].isna().sum() > 0:
                m = all_data[c].median()
                all_data[c] = all_data[c].fillna(m)

    # 分离训练和测试
    train = all_data[all_data['isDefault'] != -1].copy()
    test = all_data[all_data['isDefault'] == -1].copy()
    test.drop('isDefault', axis=1, inplace=True)

    logger.info('特征工程后: Train %s, Test %s' % (train.shape, test.shape))

    return train, test


def target_encoding(train, test, y):
    """五折目标编码 - 防泄漏"""
    logger.info('[4/6] 目标编码')

    target_enc_cols = [c for c in Config.TARGET_ENC_COLS if c in train.columns]
    logger.info('目标编码特征数: %d' % len(target_enc_cols))

    X = train.copy()
    X_test = test.copy()
    ya = y.values

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)

    for col in target_enc_cols:
        X[f'{col}_target_enc'] = np.nan

        for trn_idx, val_idx in kf.split(X, ya):
            tmp = pd.DataFrame({col: X.iloc[trn_idx][col].values, 'target': ya[trn_idx]})
            mean_enc = tmp.groupby(col)['target'].mean()
            X.iloc[val_idx, X.columns.get_loc(f'{col}_target_enc')] = X.iloc[val_idx][col].map(mean_enc)

        # 测试集用全量数据
        tmp_full = pd.DataFrame({col: X[col].values, 'target': ya})
        global_mean = tmp_full.groupby(col)['target'].mean()
        X_test[f'{col}_target_enc'] = X_test[col].map(global_mean)

    # 填充目标编码的NaN
    global_target_mean = ya.mean()
    for col in target_enc_cols:
        enc_col = f'{col}_target_enc'
        X[enc_col].fillna(global_target_mean, inplace=True)
        X_test[enc_col].fillna(global_target_mean, inplace=True)

    return X, X_test, target_enc_cols


def train_models(train, test, y, tid, teid):
    """模型训练 - 三模型融合"""
    logger.info('[5/6] 模型训练')

    drop_cols = ['id', 'isDefault']
    feats = [c for c in train.columns if c not in drop_cols]

    X, Xt = train[feats], test[feats]
    ya = y.values

    # 类别特征
    cat_feats = ['homeOwnership', 'verificationStatus', 'initialListStatus', 'purpose', 'applicationType']
    cat_feats = [c for c in cat_feats if c in feats]
    cat_idx = [feats.index(f) for f in cat_feats]

    # LightGBM/XGBoost需要数值编码
    X_encoded = X.copy()
    Xt_encoded = Xt.copy()

    for c in cat_feats:
        if c in X.columns:
            le = LabelEncoder()
            combined = pd.concat([X[c], Xt[c]], axis=0).astype(str)
            le.fit(combined)
            X_encoded[c] = le.transform(X[c].astype(str))
            Xt_encoded[c] = le.transform(Xt[c].astype(str))

    logger.info('特征数: %d, 类别特征数: %d' % (len(feats), len(cat_idx)))

    skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_SEED)

    # 存储各模型OOF和测试预测
    lgb_oof = np.zeros(len(X))
    lgb_pred = np.zeros(len(Xt))

    xgb_oof = np.zeros(len(X))
    xgb_pred = np.zeros(len(Xt))

    cat_oof = np.zeros(len(X))
    cat_pred = np.zeros(len(Xt))

    # 保存每折的模型
    lgb_models = []
    xgb_models = []
    cat_models = []

    # ========== LightGBM训练 ==========
    logger.info('\n===== LightGBM =====')
    for fold, (ti, vi) in enumerate(skf.split(X_encoded, ya)):
        logger.info('  Fold %d/%d' % (fold + 1, Config.N_SPLITS))
        Xtr, ytr = X_encoded.iloc[ti], ya[ti]
        Xv, yv = X_encoded.iloc[vi], ya[vi]

        dtrain = lgb.Dataset(Xtr, label=ytr)
        dvalid = lgb.Dataset(Xv, label=yv, reference=dtrain)

        model = lgb.train(
            Config.LGB_PARAMS,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(500)
            ]
        )

        lgb_oof[vi] = model.predict(Xv)
        lgb_pred += model.predict(Xt_encoded) / Config.N_SPLITS
        lgb_models.append(model)

        auc = roc_auc_score(yv, lgb_oof[vi])
        logger.info('    AUC=%.5f, best_iter=%d' % (auc, model.best_iteration))

        del model, dtrain, dvalid
        gc.collect()

    lgb_auc = roc_auc_score(ya, lgb_oof)
    logger.info('LightGBM OOF AUC: %.5f' % lgb_auc)

    # ========== XGBoost训练 ==========
    import xgboost as xgb

    logger.info('\n===== XGBoost =====')
    for fold, (ti, vi) in enumerate(skf.split(X_encoded, ya)):
        logger.info('  Fold %d/%d' % (fold + 1, Config.N_SPLITS))
        Xtr, ytr = X_encoded.iloc[ti], ya[ti]
        Xv, yv = X_encoded.iloc[vi], ya[vi]

        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dvalid = xgb.DMatrix(Xv, label=yv)

        model = xgb.train(
            Config.XGB_PARAMS,
            dtrain,
            num_boost_round=5000,
            evals=[(dvalid, 'val')],
            early_stopping_rounds=200,
            verbose_eval=500
        )

        xgb_oof[vi] = model.predict(dvalid)
        dtest = xgb.DMatrix(Xt_encoded)
        xgb_pred += model.predict(dtest) / Config.N_SPLITS
        xgb_models.append(model)

        auc = roc_auc_score(yv, xgb_oof[vi])
        logger.info('    AUC=%.5f, best_iter=%d' % (auc, model.best_iteration))

        del model, dtrain, dvalid, dtest
        gc.collect()

    xgb_auc = roc_auc_score(ya, xgb_oof)
    logger.info('XGBoost OOF AUC: %.5f' % xgb_auc)

    # ========== CatBoost训练 ==========
    logger.info('\n===== CatBoost =====')
    for fold, (ti, vi) in enumerate(skf.split(X, ya)):
        logger.info('  Fold %d/%d' % (fold + 1, Config.N_SPLITS))
        Xtr, ytr = X.iloc[ti], ya[ti]
        Xv, yv = X.iloc[vi], ya[vi]

        train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
        valid_pool = Pool(Xv, yv, cat_features=cat_idx)

        model = CatBoostClassifier(**Config.CAT_PARAMS)
        model.fit(train_pool, eval_set=valid_pool)

        bi = model.get_best_iteration()

        cat_oof[vi] = model.predict_proba(Xv)[:, 1]
        cat_pred += model.predict_proba(Xt)[:, 1] / Config.N_SPLITS
        cat_models.append(model)

        auc = roc_auc_score(yv, cat_oof[vi])
        logger.info('    AUC=%.5f, best_iter=%d' % (auc, bi))

        del model, train_pool, valid_pool
        gc.collect()

    cat_auc = roc_auc_score(ya, cat_oof)
    logger.info('CatBoost OOF AUC: %.5f' % cat_auc)

    # ========== 模型融合 ==========
    logger.info('\n===== 模型融合 =====')

    # 方法1: 按AUC加权（softmax）
    aucs = np.array([lgb_auc, xgb_auc, cat_auc])
    temperature = 10
    weights = np.exp(aucs * temperature) / np.exp(aucs * temperature).sum()
    logger.info('Softmax权重: LGB=%.4f, XGB=%.4f, CAT=%.4f' % (weights[0], weights[1], weights[2]))

    oof_weighted = lgb_oof * weights[0] + xgb_oof * weights[1] + cat_oof * weights[2]
    pred_weighted = lgb_pred * weights[0] + xgb_pred * weights[1] + cat_pred * weights[2]
    weighted_auc = roc_auc_score(ya, oof_weighted)
    logger.info('加权平均 AUC: %.5f' % weighted_auc)

    # 方法2: Stacking
    stack_train = np.column_stack([lgb_oof, xgb_oof, cat_oof])
    stack_test = np.column_stack([lgb_pred, xgb_pred, cat_pred])

    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(stack_train, ya)
    oof_stack = lr.predict_proba(stack_train)[:, 1]
    pred_stack = lr.predict_proba(stack_test)[:, 1]
    stack_auc = roc_auc_score(ya, oof_stack)
    logger.info('Stacking AUC: %.5f' % stack_auc)

    # 选择最优融合方法
    if stack_auc > weighted_auc:
        best_oof = oof_stack
        best_pred = pred_stack
        best_auc = stack_auc
        best_method = 'stacking'
    else:
        best_oof = oof_weighted
        best_pred = pred_weighted
        best_auc = weighted_auc
        best_method = 'weighted'

    logger.info('最优融合方法: %s, AUC=%.5f' % (best_method, best_auc))

    # 校准
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(best_oof, ya)
    oof_cal = iso.predict(best_oof)
    pred_cal = iso.predict(best_pred)

    oa_cal = roc_auc_score(ya, oof_cal)
    ok = calc_ks(ya, oof_cal)
    op = calc_psi(oof_cal, pred_cal)

    logger.info('=' * 50)
    logger.info('最终结果')
    logger.info('=' * 50)
    logger.info('OOF AUC:       %.5f' % oa_cal)
    logger.info('OOF KS:        %.5f' % ok)
    logger.info('PSI:           %.5f' % op)

    # 保存结果
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    pred_final = np.clip(pred_cal, 0.001, 0.999)
    pd.DataFrame({'id': teid.astype(int), 'isDefault': pred_final}).to_csv(
        os.path.join(Config.OUTPUT_DIR, 'submission.csv'), index=False)

    # 保存各模型预测
    pd.DataFrame({
        'id': teid.astype(int),
        'lgb': lgb_pred,
        'xgb': xgb_pred,
        'cat': cat_pred,
        'final': pred_final
    }).to_csv(os.path.join(Config.OUTPUT_DIR, 'all_predictions.csv'), index=False)

    # 保存OOF
    pd.DataFrame({
        'id': tid,
        'true': ya,
        'lgb': lgb_oof,
        'xgb': xgb_oof,
        'cat': cat_oof,
        'final': oof_cal
    }).to_csv(os.path.join(Config.OUTPUT_DIR, 'oof.csv'), index=False)

    # ==================== 保存模型文件 ====================
    logger.info('\n[6/6] 保存模型文件')

    # 保存LightGBM模型
    lgb_model_path = os.path.join(Config.OUTPUT_DIR, 'lgb_model.txt')
    lgb_models[0].save_model(lgb_model_path)
    logger.info('LightGBM模型已保存: %s' % lgb_model_path)

    # 保存XGBoost模型
    xgb_model_path = os.path.join(Config.OUTPUT_DIR, 'xgb_model.json')
    xgb_models[0].save_model(xgb_model_path)
    logger.info('XGBoost模型已保存: %s' % xgb_model_path)

    # 保存CatBoost模型
    cat_model_path = os.path.join(Config.OUTPUT_DIR, 'cat_model.cbm')
    cat_models[0].save_model(cat_model_path)
    logger.info('CatBoost模型已保存: %s' % cat_model_path)

    # 保存Stacking模型
    stacking_model_path = os.path.join(Config.OUTPUT_DIR, 'stacking_model.pkl')
    with open(stacking_model_path, 'wb') as f:
        pickle.dump(lr, f)
    logger.info('Stacking模型已保存: %s' % stacking_model_path)

    # 保存校准器
    calibrator_path = os.path.join(Config.OUTPUT_DIR, 'calibrator.pkl')
    with open(calibrator_path, 'wb') as f:
        pickle.dump(iso, f)
    logger.info('校准器已保存: %s' % calibrator_path)

    # 保存特征列表和其他必要信息
    model_info = {
        'features': feats,
        'cat_features': cat_feats,
        'cat_idx': cat_idx,
        'best_method': best_method,
        'weights': weights.tolist() if 'weights' in dir() else None,
        'lgb_auc': lgb_auc,
        'xgb_auc': xgb_auc,
        'cat_auc': cat_auc,
        'final_auc': best_auc,
        'oof_ks': ok,
        'psi': op
    }
    model_info_path = os.path.join(Config.OUTPUT_DIR, 'model_info.pkl')
    with open(model_info_path, 'wb') as f:
        pickle.dump(model_info, f)
    logger.info('模型信息已保存: %s' % model_info_path)

    logger.info('\n所有模型文件保存完成!')

    return oa_cal, ok, op, feats


def main():
    """主函数"""
    logger.info('=' * 60)
    logger.info('V35.0 - 融合优化版')
    logger.info('=' * 60)
    logger.info('优化点:')
    logger.info('  1. 完整特征工程（贷款比例、FICO、利率交互等）')
    logger.info('  2. 五折目标编码（防泄漏）')
    logger.info('  3. 缺失值指示器')
    logger.info('  4. 三模型融合（LGB+XGB+CAT）')
    logger.info('  5. Stacking + 加权平均双融合')
    logger.info('  6. Isotonic校准')
    logger.info('  7. 新增优化特征（收入稳定性、风险评分等）')
    logger.info('=' * 60)

    try:
        train, test, y, tid, teid = load_data()
        train, test = preprocess(train, test)
        train, test = create_features(train, test, y)
        train, test, target_enc_cols = target_encoding(train, test, y)
        oa, ok, op, feats = train_models(train, test, y, tid, teid)

        logger.info('')
        logger.info('=' * 60)
        logger.info('完成! OOF AUC: %.5f' % oa)
        logger.info('=' * 60)

    except Exception as e:
        logger.error('错误: %s' % str(e))
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
