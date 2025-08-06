import pandas as pd
import numpy as np
from statsmodels.api import OLS
from scipy import stats
from statsmodels.tsa.stattools import acf

# To print matrices and tables in full
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# This is a technical command needed to write data in pandas data frames
pd.options.mode.copy_on_write = True 

R2All = []
skewAll = []
kurtAll = []
SWp = []
JBp = []
L1O = []
L1A = []

def analysis(data):
    skewAll.append(round(stats.skew(data), 2))
    kurtAll.append(round(stats.kurtosis(data), 2))
    SWp.append(round(stats.shapiro(data)[1], 3))
    JBp.append(round(stats.jarque_bera(data)[1], 3))
    L1O.append(round(sum(abs(acf(data, nlags = 5)[1:])), 3))
    L1A.append(round(sum(abs(acf(abs(data), nlags = 5)[1:])), 3))

dfPrice = pd.read_excel('data.xlsx', sheet_name = 'main')
vol = dfPrice['Volatility'].values[1:]
N = len(vol)
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values[1:]
dfEarnings = pd.read_excel('data.xlsx', sheet_name = 'earnings')
earnings = dfEarnings['Earnings'].values
L = 9

def fit(window, inflMode, normMode):
    VolFactors = pd.DataFrame({'const' : 1/vol, 'vol' : 1})
    if inflMode == 'N':
        index = price
        div = dividend
        total = np.array([np.log(index[k+1] + dividend[k]) - np.log(index[k]) for k in range(N)])
        earn = earnings
    if inflMode == 'R':
        cpi = dfEarnings['CPI'].values
        index = cpi[-1]*price/cpi[L:]
        div = cpi[-1]*dividend/cpi[L+1:]
        total = np.array([np.log(index[k+1] + div[k]) - np.log(index[k]) for k in range(N)])
        earn = cpi[-1]*earnings/cpi
    Ntotal = total/vol  
    cumearn = np.array([np.mean(earn[k-window:k]) for k in range(L + 1, L + N + 2)])
    IDY = total - np.diff(np.log(cumearn))
    cumIDY = np.append(np.array([0]), np.cumsum(IDY))
    
    if normMode == 'N':
        AllFactors = pd.DataFrame({'const' : 1, 'trend' : range(N), 'Bubble' : -cumIDY[:-1]})
        modelValuation = OLS(IDY, AllFactors).fit()
    if normMode == 'Y':
        AllFactors = pd.DataFrame({'const' : 1/vol, 'trend' : range(N)/vol, 'Bubble' : -cumIDY[:-1]/vol})
        modelValuation = OLS(IDY/vol, AllFactors).fit()
    
    R2All.append(round(modelValuation.rsquared, 3))
    resValuation = modelValuation.resid
    analysis(resValuation)

for window in range(1, 11):
    fit(window, 'N', 'N')
    # change the second 'N' to 'Y' to include volatility
    
statDF = pd.DataFrame({'R^2' : R2All, 'skew': skewAll, 'kurt' : kurtAll, 'SW' : SWp, 'JB' : JBp, 'L1O': L1O, 'L1A' : L1A})
print(statDF)