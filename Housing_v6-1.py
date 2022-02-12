# -------------------------------------
# Project 5 - Statistical Analysis
# V 6
# -------------------------------------

############## Libraries ##############
from operator import index
from urllib.request import DataHandler
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.formula.api as smf

# Statistical modeling
import sklearn
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
import sklearn.metrics as metrics
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


############## Dataset ##############
data = pd.read_csv('/Users/patriciac/Documents/IRON/04_Proyectos/House pricing/train - train.csv')
data.head()
data.info()
data.shape()
data.dtypes

############## Data cleaning ##############
missing_val = data.isnull().sum()
(missing_val[missing_val > 0]/data.shape[0]*100).sort_values(ascending=False)

## Removed features
# Because Alley, Fence and MiscFeature have more than 80% of null values, and compared to other features these ones are not so relevant to the price of a house, 
# they were dropped. 
data = data.drop(['Alley','Fence','MiscFeature'], axis =1 )

## Fill NA
# It was observed the NaN values what actually mean is the absence of that feature, for example if there is no garage, then there is no garage type. Then, if there 
# is no feature, the null values in its characteristics were replaced by '0'. 

# In 'PoolQC' all NA values, which mean 'no pool', where replaced by '0'
data['PoolQC'] = data['PoolQC'].fillna(0)

# Additionally, FireplaceQu (Fireplace quality) has more than 47% of null values. After comparing the NaN values in FireplaceQu and the values in Firelaces, it was
# decided to fill the NaN with '0'. 
data[data['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']].head()
data['FireplaceQu']=data['FireplaceQu'].fillna(0)

# The features related to garage were filled with '0' when there is no garage. 
data['GarageType'] = data['GarageType'].fillna(0)
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
data['GarageFinish'] = data['GarageFinish'].fillna(0)
data['GarageQual'] = data['GarageQual'].fillna(0)
data['GarageCond'] = data['GarageCond'].fillna(0)

# Where there is no basement or second basement, the features related to them were replaced by '0'
data['BsmtFinType2'] = data['BsmtFinType2'].fillna(0)
data['BsmtQual'] = data['BsmtQual'].fillna(0)
data['BsmtFinType1'] = data['BsmtFinType1'].fillna(0)
data['BsmtExposure'] = data['BsmtExposure'].fillna(0)
data['BsmtCond'] = data['BsmtCond'].fillna(0)
data['BsmtFinType1'] = data['BsmtFinType1'].fillna(0)

# Electrical system has only 0.6% null value, which were also replaced by '0'
data['Electrical'] = data['Electrical'].fillna(0)

# For the feature 'MasVnrType', masonry veneer type, all the rows with 'none' were replaced by 0
data['MasVnrType'] = data['MasVnrType'].fillna(0)
# The area of the masonry veneer where actually this feature does not exit is also '0'
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

# Feature 'LotFrontage' was filled with '0' instead of NaN values 
data['LotFrontage'] = data['LotFrontage'].fillna(0) 

## New variables 
# To reduce the number of variables, all baths and half baths were summarized in one single columns
data['Total_baths'] = data.BsmtFullBath + data.BsmtHalfBath + data.FullBath + data.HalfBath
data.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis= 1, inplace= True)

# Variables GarageCars and GarageArea both measure garage with different units, then GarageCars was dropped
data.drop(['GarageCars'], axis= 1, inplace= True)

# 'YearBuilt' was replaced by 'Age', which si the total years since the age building till 2022
data['Age'] = 2022 - data.YearBuilt
data.drop(['YearBuilt'], axis = 1, inplace = True)

# 'Id' variable does provides much information, so it was dropped. 
data.drop(['Id'], axis= 1, inplace= True)

# Data was later classified by type, ir order to be used in different models according to ther category
numerical_data = data._get_numeric_data()
numeric_cols = list(numerical_data.columns)

categorical_data = data.drop(numeric_cols, axis= 1)
categorical_data.drop(categorical_data, axis = 1, inplace = True)
categorical_cols = list(categorical_data.columns)
# Categorical then are transformed into numerical through dummies
data = pd.get_dummies(data = data, columns= categorical_cols, drop_first=True)

data_clean = data




############## Exploratory Analysis ##############
## Sale Price distribution 
fig,axes = plt.subplots(figsize = (16,8))
ax = sns.distplot(data.SalePrice, rug=False, kde_kws={"color": "darksalmon", "lw": 3, "label": "KDE"}, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "cornflowerblue"})
plt.show()
data.describe()
# Observation: 
# The Sale Price distribution is positively skewed, which means an increase in the prices. 

# Transformation of sale price
data['SalePrice_log'] = np.log(data['SalePrice'])
data.drop(['SalePrice'], axis = 1, inplace= True)
fig,axes = plt.subplots(figsize = (16,8))
ax = sns.distplot(data.SalePrice_log, rug=False, kde_kws={"color": "darksalmon", "lw": 3, "label": "KDE"}, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "cornflowerblue"})
plt.show()
data.describe()
# Observation: 
# After a transformation to log, the sale price distribution follows a near normal distribution.  

## Correlation 
# Heatmap 
plt.figure(figsize=(16,8))
# Create a mask to only show the bottom triangle 
mask = np.triu(np.ones_like(data.corr(), dtype=bool))
sns.heatmap(data.corr(), annot=False, mask=mask, vmin=-1, vmax=1)
plt.title('Predictors correlation coefficient')
plt.show()

cor_data =data.corr()["SalePrice_log"].abs().sort_values(ascending = False).reset_index()
cor_05 = cor_data[cor_data.SalePrice_log > 0.5]

print('The variables stronger correlated to SalePrice are:', cor_05.loc[1:, ])

# Based on the correlation matrix, 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'Total_baths', 
# '1stFlrSF', 'TotRmsAbvGrd', 'Age', 'YearRemodAdd' and 'TotRmsAbvGrd' will be the features for the linear regression model. 
# How these features vary with SalePrice?
plt.figure(figsize= 25, 10))
features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'Total_baths', '1stFlrSF', 'TotRmsAbvGrd', 'Age','YearRemodAdd']
target = data['SalePrice']
for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = data[col]
    y = target
    plt.scatter(x, y, marker = 'o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Sale Price')
    plt.show()
# Observations: 
# The prices increases as the OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'Total_baths', '1stFlrSF' and 'TotRmsAbvGrd' increase. 
# There are few outliers, most of them in  'GrLivArea', 'GarageArea' and 'TotalBsmtSF'
# As th Age increases, the price decreases. 

## Hypothesis testing
# H0: all the features impact on the sale price
# HA: some features might impact, while other will not affect it at all. 

hyp_test = data.copy()

def p_value(data,ic):
    y = data['SalePrice_log']
    p_value = []
    for column in data.columns:
        x = data[column]
        slope, intercept, rvalue, pvalue, stderr = linregress(x, y)
        p_value.append((column,pvalue))  
    pvalue_d = pd.DataFrame(p_value,columns=['Variable','P_Value'])
    pvalue_d['H0'] = pvalue_d['P_Value'] <= ic     
    pvalue_d['H0'] = pvalue_d['H0'].astype('str')
    pvalue_d['H0'].replace({'True':'Statistically significant','False':'Statistically no significant'},inplace=True)
    
    return pvalue_d

# Features statistically significant 
variables_df = p_value(hyp_test, 0.05)
variables_pval = variables_df[(variables_df.H0 == 'Statistically significant')]
var_pval_list = list(variables_pval.Variable)


### Lineal regression
## In order to perform a good lineal regression there are two key assumptions:
# Multivariate Normality: Multiple regression assumes that the residuals are normally distributed.
# No Multicollinearity: Multiple regression assumes that the independent variables are not highly correlated with each other. This assumption is tested using Variance Inflation Factor (VIF) values.

## VIF 
## Computing VIF (Variange Inflation Factor), which gauges how much a feature’s inclusion contributes to the overall variance of the coefficients of the features in the model
# VIF = 1, there is no multicollinearity
# If VIF < 1, but VIF < 5, good multicollinearity

# Compute the vif for all numeric features:
def compute_vif(considered_features):
    X = data[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1  # the calculation of variance inflation requires a constant
    # create dataframe to store vif values
    vif = pd.DataFrame() # create a dataframe to store vif values
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif

# Applying compute_vif function to the variables highly correlated with other variables 
considered_features = numeric_cols
compute_vif(considered_features).sort_values('VIF', ascending=False)

# There are 9 features with a result of "inf", which means  there is perfect collinearity. 
# These represent completely redundant variables. 

# Looking at the description of the variables, these 9 variables were transformed: 
# 'BsmtFinSF1', 'BsmtFinSF2', and 'BsmtUnfSF' represent  'TotalBsmtSF', so the first three variables were removed
data.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis= 1, inplace= True)
# '1stFlrSF' and '2ndFlrSF' both sum the 'GrLivArea', and both are also evaluated as one feature in 'LowQualFinSF'. 
# Therefore, '1stFlrSF' and '2ndFlrSF' were removed
data.drop(['1stFlrSF','2ndFlrSF'], axis= 1, inplace= True)

# VIF computation without features evaluated as 'inf'
compute_vif(considered_features).sort_values('VIF', ascending=False)

# Following the rules, 'GrLivArea' was removed
# Highest VIF: GrLivArea = 5.265502
data_2 = data.drop(['GrLivArea'], axis =1 )

### Lineal regression 
variables_lm = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual',
       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinType2', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', 'LowQualFinSF', 'GrLivArea', 'BedroomAbvGr',
       'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional',
       'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscVal', 'MoSold', 'YrSold',
       'SaleType', 'SaleCondition', 'Total_baths', 'Age']


data_lm = data[variables_lm]

# Check how many features are  no numeric
numerical_data = data_lm._get_numeric_data()
numeric_cols = list(numerical_data.columns)

categorical_data = data.drop(numeric_cols, axis= 1)
categorical_data = categorical_data.drop(['SalePrice_log'], axis = 1)
# categorical_data.drop(categorical_data, axis = 1, inplace = True)
categorical_cols = list(categorical_data.columns)


data_lm = pd.get_dummies(data = data_lm, columns= categorical_cols, drop_first=True)

## Linear regression 
# Split the dataset into random train and test subset
# X = general information about the houses
# y = sale price
y = data['SalePrice_log']
train_test_split(data_lm, y)

X_train, X_test, y_train, y_test = train_test_split(data_lm, y, random_state = 10, train_size=0.8) #test_size = 0.2

## Linear Regression model using training data to fit and test data to predict
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg.coef_

data_train = pd.DataFrame(linreg.coef_, data_lm.columns)
linreg.intercept_

# R2: percentage of the variance in the dependent variable that the independent variables explain collectively 
linreg.score(X_train, y_train)
linreg.score(X_test, y_test)

predicted = linreg.predict(X_test)

df_compared = pd.DataFrame({'Real': y_test, 'Prediction': predicted})
df_compared

print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_test, predicted)}')
print(f'Mean Squared Error: {metrics.mean_squared_error(y_test, predicted)}')
print(f'Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, predicted))}')

## OLS model using the training data to fit and predict
X = sm.add_constant(data_lm)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22, test_size=0.2)
model = sm.OLS(y_train, X_train)
results = model.fit()
results.summary()

# Observation:
# After the data transformation of the data, and the VIF test the model improved, and now with the data analyzed the model can predict around 95% of the data. 
# Because the pvalue is less than 0.05, we can confirm the features analyzed are statistically significant. 
# However the model still can be improved, if more cleaning is performed. 


