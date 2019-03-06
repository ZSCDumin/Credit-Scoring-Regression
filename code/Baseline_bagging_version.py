import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 输入数据
data_path = '../input/'
train_data = pd.read_csv(data_path + 'train_dataset.csv')
test_data = pd.read_csv(data_path + 'test_dataset.csv')
sample_sub = pd.read_csv(data_path + 'submit_example.csv')

# 预处理
train_data.head(1)
print(train_data.columns)

train_data.columns = ['uid', 'true_name_flag', 'age', 'uni_student_flag', 'blk_list_flag',
                      '4g_unhealth_flag', 'net_age_till_now', 'top_up_month_diff', 'top_up_amount',
                      'recent_6month_avg_use', 'total_account_fee', 'curr_month_balance',
                      'curr_overdue_flag', 'cost_sensitivity', 'connect_num', 'freq_shopping_flag',
                      'recent_3month_shopping_count', 'wanda_flag', 'sam_flag', 'movie_flag',
                      'tour_flag', 'sport_flag', 'online_shopping_count', 'express_count',
                      'finance_app_count', 'video_app_count', 'flight_count', 'train_count',
                      'tour_app_count', 'score']
test_data.columns = train_data.columns[:-1]


# 特征提取
# age and net_age_in_month ---> 入网时的年龄 --- useless
# 先前余额，当前余额 + 当月话费 - 上次缴费 --- useless
# 充值金额/余额 --- useless
# 当月话费/最近充值金额 --- useless
# 六个月均值/充值金额 --- useless

# top up amount, 充值金额是整数，和小数，应该对应不同的充值途径？

def produce_offline_feat(train_data):
	train_data['top_up_amount_offline'] = 0
	train_data['top_up_amount_offline'][(train_data['top_up_amount'] % 10 == 0) & train_data['top_up_amount'] != 0] = 1
	return train_data


train_data = produce_offline_feat(train_data)
test_data = produce_offline_feat(test_data)


def produce_fee_rate(train_data):
	# 看importance，当月话费 和最近半年平均话费都很高，算一下当月/半年 -->稳定性
	train_data['current_fee_stability'] = train_data['total_account_fee'] / (train_data['recent_6month_avg_use'] + 1)
	
	# 当月话费/当月账户余额
	train_data['use_left_rate'] = train_data['total_account_fee'] / (train_data['curr_month_balance'] + 1)
	return train_data


train_data = produce_fee_rate(train_data)
test_data = produce_fee_rate(test_data)


# 训练
def display_importances(feature_importance_df_):
	cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
	best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
	plt.figure(figsize=(8, 10))
	sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
	plt.title('LightGBM Features (avg over folds)')
	plt.tight_layout()
	plt.show()


# para
params = {
	'learning_rate': 0.01,
	'boosting_type': 'gbdt',
	'objective': 'regression_l1',
	'metric': 'mae',
	'feature_fraction': 0.6,
	'bagging_fraction': 0.8,
	'bagging_freq': 2,
	'num_leaves': 31,
	'verbose': -1,
	'max_depth': 5,
	'lambda_l2': 5,
	'lambda_l1': 0,
	'nthread': 8
}

# para
params2 = {
	'learning_rate': 0.01,
	'boosting_type': 'gbdt',
	'objective': 'regression_l2',
	'metric': 'mae',
	'feature_fraction': 0.6,
	'bagging_fraction': 0.8,
	'bagging_freq': 2,
	'num_leaves': 31,
	'verbose': -1,
	'max_depth': 5,
	'lambda_l2': 5,
	'lambda_l1': 0,
	'nthread': 8,
	'seed': 89
}

cv_pred_all = 0
en_amount = 3
for seed in range(en_amount):
	NFOLDS = 5
	train_label = train_data['score']
	kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
	kf = kfold.split(train_data, train_label)
	
	train_data_use = train_data.drop(['uid', 'score', 'blk_list_flag'], axis=1)
	test_data_use = test_data.drop(['uid', 'blk_list_flag'], axis=1)
	
	cv_pred = np.zeros(test_data.shape[0])
	valid_best_l2_all = 0
	
	feature_importance_df = pd.DataFrame()
	count = 0
	for i, (train_fold, validate) in enumerate(kf):
		print('fold: ', i, ' training')
		X_train, X_validate, label_train, label_validate = train_data_use.iloc[train_fold, :], train_data_use.iloc[validate, :], train_label[train_fold], train_label[validate]
		dtrain = lgb.Dataset(X_train, label_train)
		dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
		bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1, early_stopping_rounds=50)
		cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
		valid_best_l2_all += bst.best_score['valid_0']['l1']
		
		fold_importance_df = pd.DataFrame()
		fold_importance_df["feature"] = list(X_train.columns)
		fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
		fold_importance_df["fold"] = count + 1
		feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
		count += 1

cv_pred /= NFOLDS
valid_best_l2_all /= NFOLDS

cv_pred_all += cv_pred
cv_pred_all /= en_amount
print('cv score for valid is: ', 1 / (1 + valid_best_l2_all))

cv_pred_all2 = 0
en_amount = 3
for seed in range(en_amount):
	NFOLDS = 5
	train_label = train_data['score']
	kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=(seed + 2019))
	kf = kfold.split(train_data, train_label)
	
	train_data_use = train_data.drop(['uid', 'score', 'blk_list_flag'], axis=1)
	test_data_use = test_data.drop(['uid', 'blk_list_flag'], axis=1)
	
	cv_pred = np.zeros(test_data.shape[0])
	valid_best_l2_all = 0
	
	feature_importance_df = pd.DataFrame()
	count = 0
	for i, (train_fold, validate) in enumerate(kf):
		print('fold: ', i, ' training')
		X_train, X_validate, label_train, label_validate = train_data_use.iloc[train_fold, :], train_data_use.iloc[validate, :], train_label[train_fold], train_label[validate]
		dtrain = lgb.Dataset(X_train, label_train)
		dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
		bst = lgb.train(params2, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1, early_stopping_rounds=50)
		cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
		valid_best_l2_all += bst.best_score['valid_0']['l1']
	
		fold_importance_df = pd.DataFrame()
		fold_importance_df["feature"] = list(X_train.columns)
		fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
		fold_importance_df["fold"] = count + 1
		feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
		count += 1

cv_pred /= NFOLDS
valid_best_l2_all /= NFOLDS

cv_pred_all2 += cv_pred

cv_pred_all2 /= en_amount
print('cv score for valid is: ', 1/(1+valid_best_l2_all))
display_importances(feature_importance_df)

# 提交
test_data_sub = test_data[['uid']]
test_data_sub['score'] = (cv_pred_all2 + cv_pred_all) / 2
test_data_sub.columns = ['id', 'score']
test_data_sub['score1'] = cv_pred_all
test_data_sub['score2'] = cv_pred_all2
test_data_sub['score'] = test_data_sub['score'].apply(lambda x: int(np.round(x)))
test_data_sub[['id', 'score']].to_csv('../output/result_bagging.csv', index=False)
