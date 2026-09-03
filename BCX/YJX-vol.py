import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

# reading the data file
DF = pd.read_excel('data2025.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]

# fit autoregression for logarithmic volatility
RegVol = OLS(np.diff(np.log(vol)), pd.DataFrame({'const' : 1, 'lag' : np.log(vol)[:-1]})).fit()
print(RegVol.summary())

# and apply Yeo-Johnson transform to exponentiated residuals
volresid = RegVol.resid
nresid = scipy.stats.yeojohnson(volresid)[0]
la = scipy.stats.yeojohnson(volresid)[1]
print('lambda = ', la)

# test for Gaussianity
qqplot(nresid, line = 's')
plt.show()
print('Shapiro-Wilk test p = ', scipy.stats.shapiro(nresid)[1])
print('Jarque-Bera test p = ', scipy.stats.jarque_bera(nresid)[1])

# now apply the Yeo-Johnson transform to the volatility data
nvol = scipy.stats.yeojohnson(vol)[0]
la = scipy.stats.yeojohnson(vol)[1]
print('lambda = ', la)

# fit autoregression of order 1 to logarithms of transformed volatility
RegNVol = OLS(np.diff(np.log(nvol)), pd.DataFrame({'const' : 1, 'lag' : np.log(nvol)[:-1]})).fit()
print(RegNVol.summary())
nvolresid = RegNVol.resid

# analyze residuals for Gaussianity
qqplot(nvolresid, line = 's')
plt.show()
print('Shapiro-Wilk test p = ', scipy.stats.shapiro(nvolresid)[1])
print('Jarque-Bera test p = ', scipy.stats.jarque_bera(nvolresid)[1])

# and for IID
plot_acf(nvolresid)
plt.show()
plot_acf(abs(nvolresid))
plt.show()