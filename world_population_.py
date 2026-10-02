# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import seaborn as sns

data=pd.read_csv('world-population-by-country-2020.csv')
data.info()
data['Yearly Change'] = data['Yearly Change'].str.replace('%', '')
data['Urban Pop %'] = data['Urban Pop %'].str.replace('%', '')
data['World Share'] = data['World Share'].str.replace('%', '')
data['Density  (P/Km²)'] = data['Density  (P/Km²)'].str.replace(',', '')
data['Population 2020'] = data['Population 2020'].str.replace(',', '')
data['Net Change'] = data['Net Change'].str.replace(',', '')
data['Land Area (Km²)'] = data['Land Area (Km²)'].str.replace(',', '')
data['Migrants (net)'] = data['Migrants (net)'].str.replace(',', '')
data.isnull().sum()
data["Migrants (net)"] = data["Migrants (net)"].apply(pd.to_numeric)
mean_value=data['Migrants (net)'].mean()
data['Migrants (net)'].fillna(value=mean_value, inplace=True)
data["Population 2020"] = data["Population 2020"].apply(pd.to_numeric)
data["Yearly Change"] = data["Yearly Change"].apply(pd.to_numeric)
data["Net Change"] = data["Net Change"].apply(pd.to_numeric)
data["Density  (P/Km²)"] = data["Density  (P/Km²)"].apply(pd.to_numeric)
data["Land Area (Km²)"] = data["Land Area (Km²)"].apply(pd.to_numeric)
data["Fert. Rate"] = data["Fert. Rate"].replace("N.A.", "0", regex=True)
data["Fert. Rate"] = data["Fert. Rate"].apply(pd.to_numeric)
mean_fert=data["Fert. Rate"].mean()
mean_fert
data['Fert. Rate'] = data['Fert. Rate'].replace(['0'], ['2.3'])
data["Med. Age"] = data["Med. Age"].replace("N.A.", "0", regex=True)
data["Med. Age"] = data["Med. Age"].apply(pd.to_numeric)
mean_med_age=data["Med. Age"].mean()
mean_med_age
data['Med. Age'] = data['Med. Age'].replace(['0'], ['26.19'])
data["Urban Pop %"] = data["Urban Pop %"].replace("N.A.", "0", regex=True)
data["Urban Pop %"] = data["Urban Pop %"].apply(pd.to_numeric)
mean_urban_pop=data["Urban Pop %"].mean()
mean_urban_pop
data['Urban Pop %'] = data['Urban Pop %'].replace(['0'], ['56.15'])
data["World Share"] = data["World Share"].apply(pd.to_numeric)

X = data.drop(['Yearly Change','no','Country (or dependency)'],axis=1)
y = data['Yearly Change'].to_numpy()
from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.3, random_state=101)

from sklearn.model_selection import KFold, cross_val_score, train_test_split
cv = KFold(n_splits=10, shuffle=True, random_state=42)

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

lasso_reg = LassoCV().fit(X_train, y_train)
lasso_score_train = -1 * cross_val_score(lasso_reg, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error').mean()
lasso_score_test = mean_squared_error(y_test, lasso_reg.predict(X_test), squared=False)

lin_reg = LinearRegression().fit(X_train, y_train)
lr_score_train = -1 * cross_val_score(lin_reg, X_train,y_train, cv=cv, scoring='neg_root_mean_squared_error').mean()
lr_score_test = mean_squared_error(y_test, lin_reg.predict(X_test), squared=False)

ridge_reg = RidgeCV().fit(X_train, y_train)
ridge_score_train = -1 * cross_val_score(ridge_reg, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error').mean()
ridge_score_test = mean_squared_error(y_test, ridge_reg.predict(X_test), squared=False)

regressor_svr = SVR().fit(X_train, y_train)
svr_score_train = -1 * cross_val_score(regressor_svr, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error').mean()
svr_score_test = mean_squared_error(y_test, regressor_svr.predict(X_test), squared=False)

from sklearn.decomposition import PCA
pca = PCA() # Default n_components = min(n_samples, n_features)
X_train_pc = pca.fit_transform(X_train)

pd.DataFrame(pca.components_.T).loc[:4,:]

import matplotlib.pyplot as plt
# Initialize linear regression instance
lin_reg = LinearRegression()

# Create empty list to store RMSE for each iteration
rmse_list = []

# Loop through different count of principal components for linear regression
for i in range(1, X_train_pc.shape[1]+1):
    rmse_score = -1 * cross_val_score(lin_reg, 
                                      X_train_pc[:,:i], # Use first k principal components
                                      y_train, 
                                      cv=cv, 
                                      scoring='neg_root_mean_squared_error').mean()
    rmse_list.append(rmse_score)
    
# Visual analysis - plot RMSE vs count of principal components used
plt.plot(rmse_list, '-o')
plt.xlabel('Number of principal components in regression')
plt.ylabel('RMSE')
plt.title('Quality')
plt.xlim(xmin=-1);
plt.xticks(np.arange(X_train_pc.shape[1]), np.arange(1, X_train_pc.shape[1]+1))
plt.axhline(y=lr_score_train, color='g', linestyle='-');

# Visually determine optimal number of principal components
best_pc_num = 9
# Train model with first 9 principal components
lin_reg_pc = LinearRegression().fit(X_train_pc[:,:best_pc_num], y_train)

# Get cross-validation RMSE (train set)
pcr_score_train = -1 * cross_val_score(lin_reg_pc, 
                                       X_train_pc[:,:best_pc_num], 
                                       y_train, 
                                       cv=cv, 
                                       scoring='neg_root_mean_squared_error').mean()

# Train model on training set
lin_reg_pc = LinearRegression().fit(X_train_pc[:,:best_pc_num], y_train)

# Get first 9 principal components of test set
X_test_pc = pca.transform(X_test)[:,:best_pc_num]

# Predict on test data
preds = lin_reg_pc.predict(X_test_pc)
pcr_score_test = mean_squared_error(y_test, preds, squared=False)

train_metrics = np.array([round(lr_score_train,3), 
                          round(lasso_score_train,3), 
                          round(ridge_score_train,3), 
                          round(pcr_score_train,3), 
                          round(svr_score_train,3)])
train_metrics = pd.DataFrame(train_metrics, columns=['RMSE (Train Set)'])
train_metrics.index = ['Linear Regression', 
                       'Lasso Regression', 
                       'Ridge Regression', 
                       f'PCR ({best_pc_num} components)',
                       'Support Vector Regression']
train_metrics

test_metrics = np.array([round(lr_score_test,3), 
                         round(lasso_score_test,3), 
                         round(ridge_score_test,3), 
                         round(pcr_score_test,3), 
                         round(svr_score_test,3)])
test_metrics = pd.DataFrame(test_metrics, columns=['RMSE (Test Set)'])
test_metrics.index = ['Linear Regression', 
                      'Lasso Regression', 
                      'Ridge Regression', 
                      f'PCR ({best_pc_num} components)',
                      'Support Vector Regression']
test_metrics