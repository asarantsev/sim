import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

def BoxCox(data, label):
    BC = stats.boxcox(data)
    print(label)
    print('order = ', BC[1])
    new = BC[0]
    print('Shapiro-Wilk p = ', stats.shapiro(new)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(new)[1])
    
DF = pd.read_excel('data.xlsx', sheet_name = None)
dfPrice = DF['main']
vol = dfPrice['Volatility'].values[1:]
N = len(vol)
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values[1:]
baa = dfPrice['BAA'].values
spread = dfPrice['Long'].values - dfPrice['Short'].values
BoxCox(baa, 'BAA')
BoxCox(np.exp(spread), 'exp-spread')
dfEarnings = DF['earnings']
earnings = dfEarnings['Earnings'].values
gearn = earnings[1:]/earnings[:-1]
BoxCox(gearn, 'earn-growth')
total = np.array([np.log(price[k+1] + dividend[k]) - np.log(price[k]) for k in range(N)])
BoxCox(np.exp(total), 'USA')
world = DF['world'] 
intlReturns = world['International'].values # international returns
IntlRet = np.log(1 + intlReturns)
BoxCox(np.exp(IntlRet), 'intl')
bonds = DF['bonds']
wealthBond = bonds['Bond Wealth'].values
BoxCox(wealthBond[1:]/wealthBond[:-1], 'bonds')