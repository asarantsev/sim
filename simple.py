import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# To print matrices and tables in full
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# This is a technical command needed to write data in pandas data frames
pd.options.mode.copy_on_write = True 

stdAll = []
skewAll = []
kurtAll = []
SWp = []
JBp = []
L1O = []
L1A = []

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data):
    stdAll.append(round(np.std(data), 4))
    skewAll.append(round(stats.skew(data), 3))
    kurtAll.append(round(stats.kurtosis(data), 3))
    SWp.append(round(stats.shapiro(data)[1], 3))
    JBp.append(round(stats.jarque_bera(data)[1], 3))
    L1O.append(round(sum(abs(acf(data, nlags = 5)[1:])), 3))
    L1A.append(round(sum(abs(acf(abs(data), nlags = 5)[1:])), 3))
        
DF = pd.read_excel('data.xlsx', sheet_name = None)
dfPrice = DF['main']
vol = dfPrice['Volatility'].values[1:]
N = len(vol)
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values[1:]
baa = dfPrice['BAA'].values

lvol = np.log(vol)
total = np.array([np.log(price[k+1] + dividend[k]) - np.log(price[k]) for k in range(N)])
nUSAret = total/vol
plots(total, 'USA-returns')
plots(nUSAret, 'norm-USA-returns')
analysis(total)
analysis(nUSAret)

RegVol = OLS(lvol[1:], pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})).fit()
RegBAA = OLS(np.diff(np.log(baa)), pd.DataFrame({'const' : 1, 'baa' : np.log(baa)[:-1]})).fit()

world = DF['world'] 
intlReturns = world['International'].values # international returns
NINTL = len(intlReturns)
IntlRet = np.log(1 + intlReturns)
nIntlRet = IntlRet/vol[-NINTL:] # normalized intl returns in %
plots(IntlRet, 'INTL-returns')
plots(nIntlRet, 'norm-INTL-returns')
analysis(IntlRet)
analysis(nIntlRet)

allRets = ['USA', 'nUSA', 'Intl', 'nIntl']
statDF = pd.DataFrame({'reg' : allRets, 'stdev' : stdAll, 'skew': skewAll, 'kurt' : kurtAll, 'SW' : SWp, 'JB' : JBp, 'L1O': L1O, 'L1A' : L1A})
print(statDF)


bonds = DF['bonds']
wealthBond = bonds['Bond Wealth'].values
NBOND = len(wealthBond) - 1
bondRet = np.diff(np.log(wealthBond)) # bond returns in %

dfBAA = pd.DataFrame({'const' : 1, 'dur' : np.diff(baa)})
RegBond = OLS(bondRet - 0.01 * baa[-NBOND-1:-1], dfBAA.iloc[-NBOND:]).fit()

regDF = pd.DataFrame({'const' : 1/vol, 'duration' : np.diff(baa)/vol, 'vol' : 1}) 
RegUSA = OLS(nUSAret, regDF).fit()
RegIntl = OLS(nIntlRet, regDF.iloc[-NINTL:]).fit()

allRegs = [RegVol, RegBAA, RegUSA, RegIntl, RegBond]
allNames = ['vol', 'baa', 'usa-ret', 'intl-ret', 'bond-ret']
allResiduals = pd.DataFrame(columns = allNames)
DIM = 5

stdAll = []
skewAll = []
kurtAll = []
SWp = []
JBp = []
L1O = []
L1A = []
lengths = []

for k in range(DIM):
    print(allNames[k], '\n') # name of regression
    regression = allRegs[k] # regression itself
    print(regression.summary()) # print regression summary
    print('coefficients')
    print(regression.params) # print regression parameters
    resids = regression.resid.values # residuals of this regression
    lengths.append(len(resids))
    allResiduals[allNames[k]] = np.pad(resids[::-1], (0, N - lengths[k]), constant_values = np.nan)
    plots(resids, allNames[k]) # normality and autocorrelation function plots
    analysis(resids) # are these residuals normal white noise?

corrMatrix = allResiduals.corr()
print(corrMatrix)

statDF = pd.DataFrame({'reg' : allNames, 'length' : lengths, 'stdev' : stdAll, 'skew': skewAll, 'kurt' : kurtAll, 'SW' : SWp, 'JB' : JBp, 'L1O': L1O, 'L1A' : L1A})
print(statDF)
