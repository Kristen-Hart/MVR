# libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler

# loading
data = pd.read_csv('Agony - Health And Zillow Population Fixed.csv')

# filtering to 2017-2021
data = data[(data['Year'] >= 2017) & (data['Year'] <= 2021)]

# duplicate removal
data = data.drop_duplicates()

# Drop rows with NaN values
data = data.dropna()


# predictors
X = data.drop(columns=['Zillow_Value', 'FIPS', 'State', 'Year', 'County'])
y = data['Zillow_Value']

# training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# fitting regression model
X_train_sm = sm.add_constant(X_train)  # intercept
model = sm.OLS(y_train, X_train_sm).fit()

# model summary
print(model.summary())

# scikit learn model
lr = LinearRegression()
lr.fit(X_train, y_train)

# coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_})
print(coefficients.sort_values(by='Coefficient', ascending=False))

# VIF calculation
X_train_vif = sm.add_constant(X_train)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_train.columns
vif_data['VIF'] = [variance_inflation_factor(X_train_vif.values, i + 1) for i in range(len(X_train.columns))]

print("Variance Inflation Factor (VIF):")
print(vif_data.sort_values(by='VIF', ascending=False))

# lasso and ridge to improve performance of model
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

# cross validate the data
ridge_cv_score = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
lasso_cv_score = cross_val_score(lasso, X_train, y_train, cv=5, scoring='r2')

print("\nRidge Regression Cross-Validation R^2 Scores:", ridge_cv_score)
print("Mean Ridge Regression R^2 Score:", ridge_cv_score.mean())

print("\nLasso Regression Cross-Validation R^2 Scores:", lasso_cv_score)
print("Mean Lasso Regression R^2 Score:", lasso_cv_score.mean())

# fitting to lasso and ridge
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

ridge_coefficients = pd.DataFrame({'Feature': X.columns, 'Ridge Coefficient': ridge.coef_})
lasso_coefficients = pd.DataFrame({'Feature': X.columns, 'Lasso Coefficient': lasso.coef_})

print("\nRidge Coefficients:")
print(ridge_coefficients.sort_values(by='Ridge Coefficient', ascending=False))

print("\nLasso Coefficients:")
print(lasso_coefficients.sort_values(by='Lasso Coefficient', ascending=False))

# remove high VIF features to address multicollinearity
X_train_reduced = X_train.drop(columns=['Some_College_Perc_25_44', 'Population'])  # Drop based on high VIF values

# fit and check for improvement
X_train_reduced_sm = sm.add_constant(X_train_reduced)
model_reduced = sm.OLS(y_train, X_train_reduced_sm).fit()
print(model_reduced.summary())

# cross validate lasso and ridge regrsesions
alphas = np.logspace(-4, 4, 50) 

ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
lasso_cv = LassoCV(alphas=alphas, cv=5).fit(X_train, y_train)

print("Optimal alpha for Ridge:", ridge_cv.alpha_)
print("Optimal alpha for Lasso:", lasso_cv.alpha_)

# coefficients after tuning
ridge_coefficients_cv = pd.DataFrame({'Feature': X.columns, 'Ridge Coefficient': ridge_cv.coef_})
lasso_coefficients_cv = pd.DataFrame({'Feature': X.columns, 'Lasso Coefficient': lasso_cv.coef_})

print("\nRidge Coefficients after tuning:")
print(ridge_coefficients_cv.sort_values(by='Ridge Coefficient', ascending=False))

print("\nLasso Coefficients after tuning:")
print(lasso_coefficients_cv.sort_values(by='Lasso Coefficient', ascending=False))

# siginificant features from p values
significant_features = [
    'Ment_Unhealth_Days', 
    'Smoke_Perc', 
    'Obese_Perc', 
    'Severe_Housing_Perc', 
    'Excess_Drink_Perc', 
    'Some_College_Perc_25_44', 
    'Some_Association_Rate', 
    'Insufficent_Sleep_Perc'
]

# training set only include significant features
X_train_significant = X_train[significant_features]

# new model with only signifcant features
X_train_significant_sm = sm.add_constant(X_train_significant)
model_significant = sm.OLS(y_train, X_train_significant_sm).fit()
print(model_significant.summary())

# standardizing the significant features
scaler = StandardScaler()
X_train_significant_scaled = scaler.fit_transform(X_train_significant)

# further tuning on lasso and ridge
alphas = np.logspace(-4, 4, 50)

ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train_significant_scaled, y_train)
lasso_cv = LassoCV(alphas=alphas, cv=5).fit(X_train_significant_scaled, y_train)

print("Optimal alpha for Ridge (refined features):", ridge_cv.alpha_)
print("Optimal alpha for Lasso (refined features):", lasso_cv.alpha_)

# coefficients after further tuning
ridge_coefficients_cv = pd.DataFrame({'Feature': significant_features, 'Ridge Coefficient': ridge_cv.coef_})
lasso_coefficients_cv = pd.DataFrame({'Feature': significant_features, 'Lasso Coefficient': lasso_cv.coef_})

print("\nRidge Coefficients after tuning (refined):")
print(ridge_coefficients_cv.sort_values(by='Ridge Coefficient', ascending=False))

print("\nLasso Coefficients after tuning (refined):")
print(lasso_coefficients_cv.sort_values(by='Lasso Coefficient', ascending=False))