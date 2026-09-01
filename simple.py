import pandas as pd
import numpy as np
from statsmodels.api import OLS
from scipy import stats
from matplotlib import pyplot as plt

DF = pd.read_excel('data.xlsx', sheet_name = None)
dfPrice = DF['main']
vol = dfPrice['Volatility'].values[1:]
N = len(vol)
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values[1:]
lvol = np.log(vol)
total = np.array([np.log(price[k+1] + dividend[k]) - np.log(price[k]) for k in range(N)])
nUSAret = total/vol
RegVol = OLS(lvol[1:], pd.DataFrame({'const' : 1, 'lag' : lvol[:-1]})).fit()
RegUSA = OLS(nUSAret, pd.DataFrame({'const' : 1/vol, 'vol' : 1})).fit()
print(RegVol.params)
print(np.std(RegVol.resid))
print(RegUSA.params)
print(np.std(RegUSA.resid))
print(N - 1)
covVol = np.linalg.inv(np.array([[N - 1, np.sum(lvol[:-1])], [np.sum(lvol[:-1]), np.sum(np.square(lvol[:-1]))]]))
covUSA = np.linalg.inv(np.array([[N, np.sum(1/vol)], [np.sum(1/vol), np.sum(np.square(1/vol))]])) 
print(covVol)
print(covUSA)
NSIMS = 10000

def sim(initVol, T):
    noiseUSA = np.random.normal(0, 0.0162389, (T, NSIMS))
    noiseVol = np.random.normal(0, 0.364353, (T, NSIMS))
    simRetUSA = np.zeros((T, NSIMS))
    simLVol = np.zeros((T+1, NSIMS))
    simLVol[0] = np.log(initVol) * np.ones(NSIMS)
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = 0.84785 * np.ones(NSIMS) + 0.620146 * simLVol[t] + noiseVol[t]
    
    # take exponents to get volatility
    simVol = np.exp(simLVol)
    for t in range(T):
        simRetUSA[t] = 0.227336 * np.ones(NSIMS) - 0.012476 * simVol[t + 1]  + simVol[t + 1] * noiseUSA[t]
        
    return simRetUSA

def bayesCoeffSim(initVol, T):
    noiseUSA = np.random.normal(0, 0.0162389, (T, NSIMS))
    noiseVol = np.random.normal(0, 0.364353, (T, NSIMS))
    simRetUSA = np.zeros((T, NSIMS))
    simLVol = np.zeros((T+1, NSIMS))
    simLVol[0] = np.log(initVol) * np.ones(NSIMS)
    
    simCoeffVol = np.random.multivariate_normal([0.84785, 0.620146], covVol * 0.364353**2, NSIMS)
    simCoeffUSA = np.random.multivariate_normal([-0.012476, 0.227336], covUSA * 0.0162389**2, NSIMS)
    
    alpha = simCoeffVol[:, 0]
    beta = simCoeffVol[:, 1]
    theta = simCoeffUSA[:, 0]
    gamma = simCoeffUSA[:, 1]
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = alpha * np.ones(NSIMS) + beta * simLVol[t] + noiseVol[t]
    
    # take exponents to get volatility
    simVol = np.exp(simLVol)
    for t in range(T):
        simRetUSA[t] = gamma * np.ones(NSIMS) + theta * simVol[t + 1]  + simVol[t + 1] * noiseUSA[t]
        
    return simRetUSA

def bayesAllSim(initVol, T):
    noiseUSA = np.random.normal(0, 0.0162389, (T, NSIMS))
    noiseVol = np.random.normal(0, 0.364353, (T, NSIMS))
    simRetUSA = np.zeros((T, NSIMS))
    simLVol = np.zeros((T+1, NSIMS))
    simLVol[0] = np.log(initVol) * np.ones(NSIMS)
    
    simPrecVol = np.random.gamma((N - 1)/2, 2*0.364353**(-2)/(N - 1), NSIMS)
    simStdVol = np.power(simPrecVol, -0.5)
    simCoeffVol = np.tile([0.84785, 0.620146], (NSIMS, 1)) + np.random.multivariate_normal([0, 0], covVol, NSIMS) * np.transpose(np.tile(simStdVol, (2, 1)))
    simPrecUSA = np.random.gamma(N/2, 2*0.0162389**(-2)/N, NSIMS)
    simStdUSA = np.power(simPrecUSA, -0.5)
    simCoeffUSA = np.tile([-0.012476, 0.227336], (NSIMS, 1)) + np.random.multivariate_normal([0, 0], covUSA, NSIMS) * np.transpose(np.tile(simStdUSA, (2, 1)))
    
    alpha = simCoeffVol[:, 0]
    beta = simCoeffVol[:, 1]
    theta = simCoeffUSA[:, 0]
    gamma = simCoeffUSA[:, 1]
    
    # now comes the simulation itself!
    # simulate logarithms of volatility as autoregression
    for t in range(T):
        simLVol[t + 1] = alpha * np.ones(NSIMS) + beta * simLVol[t] + noiseVol[t]
    
    # take exponents to get volatility
    simVol = np.exp(simLVol)
    for t in range(T):
        simRetUSA[t] = gamma * np.ones(NSIMS) + theta * simVol[t + 1]  + simVol[t + 1] * noiseUSA[t]
        
    return simRetUSA

T = 30
initVol = 20

model0 = sim(initVol, T)
model1 = bayesCoeffSim(initVol, T)
model2 = bayesAllSim(initVol, T)

for model in [model0, model1, model2]:
    avgModel = np.mean(model, axis = 0)
    print('mean = ', np.mean(avgModel))
    print('std = ', np.std(avgModel))
    print('median = ', np.median(avgModel))
    for percent in [10, 30, 70, 90]:
        print(str(percent) + '% = ', np.percentile(avgModel, percent))
    # Now test the withdrawal rule
    wealth = np.zeros((T+1, NSIMS))
    wealth[0] = np.ones(NSIMS)
    for withdrawal in [0.03, 0.04, 0.05]:
        for t in range(T):
            wealth[t+1] = wealth[t] * np.exp(model[t]) - withdrawal * (1.04**t) * np.ones(NSIMS)
        print('withdrawal rate ', withdrawal)
        print(np.sum(wealth[T] > 0)/NSIMS)
