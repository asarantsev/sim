import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
import scipy
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from YJX import YJinv

# reading the data file
DF = pd.read_excel('data2025.xlsx', sheet_name = 'data')
vol = DF['Volatility'].values[1:]

# fit autoregression for logarithmic volatility
RegVol = OLS(np.diff(np.log(vol)), pd.DataFrame({'const' : 1, 'lag' : np.log(vol)[:-1]})).fit()
print(RegVol.summary())

# and apply Yeo-Johnson transform to residuals
volresid = RegVol.resid
nresid = scipy.stats.yeojohnson(volresid)[0]
la = scipy.stats.yeojohnson(volresid)[1]
print('lambda = ', la)

# apply the inverse transform to test this function
fresid = YJinv(nresid, la)
plt.plot(fresid, volresid, 'o')
plt.show()

# test for Gaussianity
qqplot(nresid, line = 's')
plt.show()
print('Shapiro-Wilk test p = ', scipy.stats.shapiro(nresid)[1])
print('Jarque-Bera test p = ', scipy.stats.jarque_bera(nresid)[1])
