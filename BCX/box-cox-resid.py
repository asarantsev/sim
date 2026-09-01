import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

DF = pd.read_excel('innovations.xlsx')
for k in range(2, 6):
    print(list(DF.keys())[k])
    resid = DF.values[:, k]
    nres = stats.boxcox(np.exp(resid))[0]
    qqplot(nres, line = 's')
    plt.show()
    print('Shapiro-Wilk p = ', stats.shapiro(nres)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(nres)[1])
