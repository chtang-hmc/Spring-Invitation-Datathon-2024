import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

def test_linear(model):
    print(f'The p-value for the F-test is: {model.f_pvalue}.')
    if model.f_pvalue < 0.05:
        print('The model is statistically significant.')
        return True
    else:
        print('The model is not statistically significant.')
        return False
    
def plot_residuals(model):
    plt.hist(model.resid, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    # fit a normal curve to the histogram of residuals
    mu, std = stats.norm.fit(model.resid)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p * len(model.resid) * 10, 'k', linewidth=2)
    plt.show()
    
def check_normal_residuals(model):
    print('The null hypothesis for the Jarque-Bera test is normality.')
    name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
    test = sm.stats.stattools.jarque_bera(model.resid)
    print(dict(zip(name, test)))
    plot_residuals(model)
    if test[1] < 0.05:
        print('The residuals are not normally distributed.')
        return False
    else:
        print('The residuals are normally distributed.')
        return True
    
def test_homoscedasticity(model):
    print('The null hypothesis for the Breusch-Pagan test is homoscedasticity.')
    name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    test = sm.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
    print(dict(zip(name, test)))
    
    plt.scatter(model.fittedvalues, model.resid)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted values')
    plt.show()
    
    if test[1] < 0.05:
        print('The residuals are heteroscedastic.')
        return False
    else:
        print('The residuals are homoscedastic.')
        return True
    
def check_autocorrelation(model):
    print('The null hypothesis for the Durbin-Watson test is no autocorrelation.')
    name = ['Durbin-Watson', 'p-value']

    # Sample data
    residuals = model.resid

    # Calculate Durbin-Watson statistic
    durbin_watson_statistic = sm.stats.stattools.durbin_watson(model.resid)

    # Get the number of observations and number of predictors
    n_obs = len(residuals)
    n_predictors = len(model.params) - 1

    # Calculate approximate p-value
    p_value = 2 * (1 - stats.norm.cdf(durbin_watson_statistic))
    test = [durbin_watson_statistic, p_value]
    print(dict(zip(name, test)))
    
    sm.graphics.tsa.plot_acf(model.resid, lags=10)
    plt.title('Autocorrelation Plot of Residuals')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()
    
    if p_value < 0.05:
        print('The residuals are autocorrelated.')
        return False
    else:
        print('The residuals are not autocorrelated.')
        return True
    
def draw_qq_plot(model):
    sm.qqplot(model.resid, line='r')
    plt.title('QQ Plot')
    plt.show()
    
def check_multi_collinearity(X):
    corr = X.corr()
    print('The correlation matrix is:')
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    X = sm.add_constant(X)
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)
    if vif['VIF'].max() > 10:
        print('There is multicollinearity in the model.')
        return False
    else:
        print('There is no multicollinearity in the model.')
        return True
    
def test_linear_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    linear = test_linear(model)
    if linear:
        draw_qq_plot(model)
        normal_residuals = check_normal_residuals(model)
        homoscedasticity = test_homoscedasticity(model)
        autocorrelation = check_autocorrelation(model)
        multi_collinearity = check_multi_collinearity(X)
        if normal_residuals and homoscedasticity and not autocorrelation and multi_collinearity:
            print('The model passes all the assumptions.')
            print(model.summary())
    return model