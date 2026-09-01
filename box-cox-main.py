import pandas as pd
from verification import plots
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import stats

def verification(data):
    print('Shapiro-Wilk p = ', scipy.stats.shapiro(data)[1])
    print('Jarque-Bera p = ', scipy.stats.jarque_bera(data)[1])
    print('ACF p-value for Ljung-Box test = ', stats.acorr_ljungbox(data, lags = [5, 10])['lb_pvalue'].values)
    print('Same for absolute values = ', stats.acorr_ljungbox(abs(data), lags = [5, 10])['lb_pvalue'].values)
   
def BoxCox(data, label):
    BC = scipy.stats.boxcox(data)
    print(label)
    print('order = ', BC[1])
    new = BC[0]
    return new
    
def AR(data, label):
    AR = OLS(data[1:], pd.DataFrame({'const' : 1,' lag' : data[:-1]})).fit()
    print(label)
    print(AR.summary())
    verification(AR.resid)
    
DF = pd.read_excel('data.xlsx', sheet_name = None)
dfPrice = DF['main']
vol = dfPrice['Volatility'].values[1:]
N = len(vol)
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values[1:]
baa = dfPrice['BAA'].values
spread = dfPrice['Long'].values - dfPrice['Short'].values

verification(BoxCox(baa, 'BAA'))
AR(BoxCox(baa, 'BAA'), 'BAA')
verification(BoxCox(np.exp(spread), 'exp-spread'))
AR(BoxCox(np.exp(spread), 'exp-spread'), 'exp-spread')

dfEarnings = DF['earnings']
earnings = dfEarnings['Earnings'].values
gearn = earnings[1:]/earnings[:-1]
verification(BoxCox(gearn, 'earn-growth'))

total = np.array([np.log(price[k+1] + dividend[k]) - np.log(price[k]) for k in range(N)])
verification(BoxCox(np.exp(total), 'USA'))

world = DF['world'] 
intlReturns = world['International'].values # international returns
IntlRet = np.log(1 + intlReturns)
verification(BoxCox(np.exp(IntlRet), 'intl'))

bonds = DF['bonds']
wealthBond = bonds['Bond Wealth'].values
verification(BoxCox(wealthBond[1:]/wealthBond[:-1], 'bonds'))