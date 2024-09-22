# test_code  
# Testing the performance optimization effect of SGDA method on logistic regression models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from my_utils.base import model_val, GradientDistributionLikelihood

# Number of intervals
bins = 20

edges = [float(x) / bins for x in range(bins+1)]
edges[-1] += 1e-5

src_data_set = pd.read_csv('../data_sets/Creditcard_fraud/creditcard.csv')
# Data preprocessing
mm_scaler = MinMaxScaler()
src_data_set['Amount'] = mm_scaler.fit_transform(src_data_set.Amount.values.reshape(-1, 1))
src_data_set['Time'] = mm_scaler.fit_transform(src_data_set.Time.values.reshape(-1, 1))

X = src_data_set.drop(columns=['Class'])
y = src_data_set['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Get prediction probability of predicted samples by LR 
src_lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=200)
src_lr.fit(X_train, y_train)

# --GDL--LR
gdl_src_set = GradientDistributionLikelihood(bins, X_train, y_train, src_lr)
# print(sorted(gdl_src_set.maj_bins_info.items(), key=lambda x: x[0]))
# print(sorted(gdl_src_set.min_bins_info.items(), key=lambda x: x[0]))
# print(gdl_src_set.generate_info)
# print(gdl_src_set.get_to_generate_num_for_every_bins())

X_resampled_gdl, y_resampled_gdl = gdl_src_set.fit_resample(flag=0)
gdl_resampled_lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=200)
gdl_resampled_lr.fit(X_resampled_gdl, y_resampled_gdl)
print("--GDL--LR--")
model_val(X_test, y_test, gdl_resampled_lr)

