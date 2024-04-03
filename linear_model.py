import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

def test_linear(model):
    print(f'The p-value for the F-test is: {model.f_pvalue}.')
    if model.f_pvalue < 0.05:
        print('The model is statistically significant.')
        return True
    else:
        print('The model is not statistically significant.')
        return False
    
def plot_residuals(model):
    residuals = model.resid
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=20, edgecolor='black')
    ax.set_title('Histogram of Residuals')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    plt.show()
    
def check_normal_residuals(model):
    print('The null hypothesis for the Jarque-Bera test is normality.')
    name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
    test = sm.stats.stattools.jarque_bera(model.resid)
    print(dict(zip(name, test)))
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
    if test[1] < 0.05:
        print('The residuals are heteroscedastic.')
        return False
    else:
        print('The residuals are homoscedastic.')
        return True
    
def check_autocorrelation(model):
    print('The null hypothesis for the Durbin-Watson test is no autocorrelation.')
    name = ['Durbin-Watson', 'p-value']
    test = sm.stats.stattools.durbin_watson(model.resid)
    print(dict(zip(name, test)))
    if test[1] < 0.05:
        print('The residuals are autocorrelated.')
        return False
    else:
        print('The residuals are not autocorrelated.')
        return True
    
def check_multi_collinearity(X):
    corr = X.corr()
    print('The correlation matrix is:')
    print(corr)
    if (np.abs(corr) > 0.9).sum().sum() > corr.shape[0]:
        print('The features are highly correlated.')
        return False
    else:
        print('The features are not highly correlated.')
        return True
    
def test_linear_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    linear = test_linear(model)
    if linear:
        plot_residuals(model)
        normal_residuals = check_normal_residuals(model)
        homoscedasticity = test_homoscedasticity(model)
        autocorrelation = check_autocorrelation(model)
        multi_collinearity = check_multi_collinearity(X)
        if normal_residuals and homoscedasticity and not autocorrelation and multi_collinearity:
            print('The model passes all the assumptions.')
    return model