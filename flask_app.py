# Flask is framework used to connect Python and HTML
from flask import Flask, render_template, request, send_file

# operating system library to get current folder
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FuncFormatter

# introductory commands
app = Flask(__name__)
app.config["DEBUG"] = True
np.random.seed(0)
current_dir = os.path.abspath(os.path.dirname(__file__))
dataFile = os.path.join(current_dir, 'static', 'filled.xlsx')
outputPNG = os.path.join(current_dir, 'static', 'wealth.png')
outputPDF = os.path.join(current_dir, 'static', 'wealth.pdf')

# reading the innovations file with filled values
DFinnovations = pd.read_excel(dataFile)
innovations = DFinnovations.values

# Current factor values in the market
# Volatility, long-short spread, BAA rate, the bubble measure

# Volatility scaled to VIX: VIX daily July 2024 -  2025 close average
currVol = (10.37/19.4645) * 19.13

# BAA rate, June 2025 daily close average
currRate = 6.15

# Earnings (nominal) over the years 2015 - 2024
currEarn = [0.99, 0.93, 0.80, 0.29, 0.69, 0.98, 0.93, 1.25, 1.24, 1.11]

# June daliy average difference between 10-year and 3-month Treasury rates
currSpread = 4.38 - 4.23

# The bubble measure computed using December 31, 2024, bubble measure,
# plus total nominal return of the S&P 500, Dec 31, 2024 -- June 30, 2025
# We did not have earnings for even Q1 2025, unfortunately
currBubble = -0.193141 + np.log(6204.95 + 19.37 + 19.48) - np.log(5881.63)

# White noise data: independent identically distribution innovations/residuals
DIM = 7 # number of dimensions
NDATA = 97 # number of data points

NSIMS = 1600 # number of Monte Carlo simulations

# displayed wealth tiem series graphs in Monte Carlo simulations
NDISPLAYS = 5 # number of graphs
# chosen percentages for final wealth ranking of paths
selectedPercentages = [0.1, 0.3, 0.5, 0.7, 0.9]

# this is simulating all necessary values from kernel density estimation
# data is the original data, written so that data[k] has dimension dim
# N is the number of data points
# bandwidth is the array of chosen variances
# bandwidth[k] is the variance for kth component
# we assume that the Gaussian noise has independent components
# nYears is, as usual, the number of years in simulation
def simKDE(data, N, dim, nYears):

    # simulate randomly chosen index from 0 to N - 1
    # Need this since the function choice works only with 1D arrays
    index = np.random.choice(range(N), size = (NSIMS, nYears), replace = True)
    pick = data[index] # Result is two-dimensional array of vectors in R^dim

    # Silverman's rule of thumb: Common factor for all 'dim'
    silvermanFactor = (4/(dim + 2))**(1/(dim + 4))*N**(-1/(dim + 4))
    noise = [] # Here we will write 2D simulated arrays for each of 'dim' components

    for k in range(dim):

        # actual bandwidth for the kth component
        bandwidth = silvermanFactor * min(np.std(data[:, k]), stats.iqr(data[:, k]/1.34))

        # simulate kth component
        # we can simulate them independently since the covariance matrix is diagonal
        component = np.random.normal(0, bandwidth, size = (NSIMS, nYears))
        noise.append(component) # write the 2D simulated array for the current component
    noise = np.transpose(np.array(noise), (1, 2, 0)) # need to swap coordinates to sum with 'pick'
    return pick + np.array(noise)

def summation(data, length, window):
    output = np.zeros((NSIMS, length - window + 1))
    for k in range(window):
        output = output + data[:, k:k + length - window + 1]
    return output

# simulate portfolio returns
# first two arguments are initial volatility and rate
# nYears is the number of years
# last two arguments are shares of bonds in portfolio
# and international among stocks
def simReturns(initVol, initBubble, initRate, initSpread, initEarn, nYears, bondPstart, bondPend, intlP):

    # initial earnings for the last 10 years
    # initializing array simulations
    simLVol = np.zeros((NSIMS, nYears + 1))
    simLVol[:, 0] = np.log(initVol) * np.ones(NSIMS)
    simLRate = np.zeros((NSIMS, nYears + 1))
    simLRate[:, 0] = np.log(initRate) * np.ones(NSIMS)
    simSpread = np.zeros((NSIMS, nYears + 1))
    simSpread[:, 0] = initSpread * np.ones(NSIMS)
    simBubble = np.zeros((NSIMS, nYears + 1))
    simBubble[:, 0] = initBubble * np.ones(NSIMS)
    simGeomUS = np.zeros((NSIMS, nYears))
    simGeomIntl = np.zeros((NSIMS, nYears))
    simGeomBond = np.zeros((NSIMS, nYears))

    # simulate series of residuals using the given 'innovations' series
    # We use multivariate kernel density estimation with bandwidth
    # given by Silverman's rule of thumb
    # need this to simulate these random terms fast
    simResiduals = simKDE(innovations, NDATA, DIM, nYears)

    # we attach name to each component for clarity
    noiseVol = simResiduals[:, :, 0] # noise for autoregression for volatility
    noiseSpread = simResiduals[:, :, 1] # noise for vector autoregression for long-short spread
    noiseLRate = simResiduals[:, :, 2] # noise for vector autoregression for log rate
    noiseEarnGr = simResiduals[:, :, 3] # noise for earnings growth
    noiseUS = simResiduals[:, :, 4] # noise for US returns
    noiseIntl = simResiduals[:, :, 5] # noise for international returns
    noiseBond = simResiduals[:, :, 6] # noise for bond returns

    # simulate log volatility and log rate as autoregression
    for t in range(nYears):
        simLVol[:, t + 1] = 0.847850 * np.ones(NSIMS) + 0.620146 * simLVol[:, t] + noiseVol[:, t]
        simSpread[:, t + 1] = 0.643694 * np.ones(NSIMS) + 0.539518 * simSpread[:, t] + noiseSpread[:, t]
        simLRate[:, t + 1] = 0.107208 * np.ones(NSIMS) + 0.942062 * simLRate[:, t] + noiseLRate[:, t]


    simRate = np.exp(simLRate) # switch from log rate to rate
    simD = simRate[:, 1:] - simRate[:, :-1] # one-year change in simulated rate
    simVol = np.exp(simLVol) # from log volatility to volatility

    # simulate log earnings growth using linear regression
    simEarnGr = 0.077565 * np.ones((NSIMS, nYears)) - 0.007842 * simVol[:, 1:] + 0.047863 * simSpread[:, :-1] + 0.03721 * simD + simVol[:, 1:] * noiseEarnGr

    # to do: simulate cumulative earnings
    simLEarn = np.zeros((NSIMS, nYears + 10)) # initialize
    simLEarn[:, :10] = np.tile(np.log(initEarn), (NSIMS, 1)) # start from given ones

    # now add earnings growth simulated terms to previous log earnings terms to get new log earnings terms
    for t in range(nYears):
        simLEarn[:, t + 10] = simLEarn[:, t + 9] + simEarnGr[:, t]

    # from log earnings to earnings
    simEarn = np.exp(simLEarn)

    # and compute cumulative earnings from annual earnings using 10-year window
    simCumEarn = summation(simEarn, nYears + 10, 10)
    simCumEarnGr = np.log(simCumEarn[:, 1:]) - np.log(simCumEarn[:, :-1]) # log growth

    # simulate US S&P stock, international stock, and corporate bond returns
    # as linear regressions vs simulated factors
    for t in range(nYears):

        # simulate geometric returns of USA stocks
        simGeomUS[:, t] = 0.26851 * np.ones(NSIMS) - 0.013568 * simVol[:, t + 1] - 0.078238 * simD[:, t] - 0.164398 * simBubble[:, t] - 0.03412 * simSpread[:, t] + simVol[:, t + 1] * noiseUS[:, t]

        # compute the next step of the bubble measure
        simBubble[:, t + 1] = simGeomUS[:, t] - simCumEarnGr[:, t] - 0.0452742 * np.ones(NSIMS)

    # simulate geometric returns of international stocks
    simGeomIntl = 0.268868 * np.ones((NSIMS, nYears)) - 0.018790 * simVol[:, 1:] - 0.051445 * simD - 0.094098 * simBubble[:, :-1] + noiseIntl * simVol[:, 1:]

    # simulate geometric returns of bonds
    simGeomBond = 0.01 * simRate[:, :-1] - 0.016611 * np.ones((NSIMS, nYears)) - 0.055884 * simD + noiseBond

    # convert geometric returns into arithmetic
    simUS = np.exp(simGeomUS) - np.ones((NSIMS, nYears))
    simIntl = np.exp(simGeomIntl) - np.ones((NSIMS, nYears))
    simBond = np.exp(simGeomBond) - np.ones((NSIMS, nYears))

    simStock = simIntl * intlP + simUS * (1 - intlP) # simulate stock portfolio
    # changing decomposition of stock/bond split
    bondP = np.linspace(bondPstart, bondPend, nYears)
    stockP = np.ones(nYears) - bondP
    simOverall = simBond * bondP + simStock * stockP # simulate combined stock and bond portfolio
    return simOverall

# simulate wealth process given initial wealth and contributions/withdrawals
# some arguments are the same as in the previous function
# others are: initialW = initialWealth and
# initialFlow (signed value) = first year flow: contribution (+) or withdrawal (-)
# growthFlow (signed value) = annual growth (+) or decline (-) of flow
def simWealth(initVol, initBubble, initRate, initSpread, initEarn, initialW, initialFlow, growthFlow, nYears, bondShare0, bondShare1, intlShare):

    #simulate returns of this portfolio
    simRet = simReturns(initVol, initBubble, initRate, initSpread, initEarn, nYears, bondShare0, bondShare1, intlShare)
    pathData = np.mean(simRet, axis = 1) # average arithmetic returns over each simulation
    wealth = np.zeros((NSIMS, nYears + 1)) # create an array for wealth simulation
    wealth[:, 0] = np.ones(NSIMS) * initialW # initial wealth year 0 initialize

    # create (deterministic) array for flow (contributions and withdrawals)
    # for each year in nYears, exponentially growing or decreasing
    flow = initialFlow * np.exp(np.array(range(nYears)) * np.log(1 +  growthFlow))

    # this is the main function connecting wealth to returns and flow
    if initialFlow < 0:
        for sim in range(NSIMS):
            for t in range(nYears):
                # main equation connecting returns, flow, wealth
                wealth[sim, t + 1] = wealth[sim, t] * (1 + simRet[sim, t]) + flow[t]
                if wealth[sim, t + 1] <= 0:
                    pathData[sim] = t + 1 # record ruin year in place of average returns
                    wealth[sim, t + 1:] = 0 # all future wealth is zero
                    break # and stop this particular simulation

    else: # if no withdrawals then we do not need to check for bankruptcy
        for t in range(nYears):
            # main equation connecting returns, flow, wealth
            wealth[:, t+1] = wealth[:, t] * (1 + simRet[:, t]) + flow[t] * np.ones(NSIMS)

    # timeAvgRet = average total portfolio return array over each path
    # wealth = paths of wealth
    return pathData, wealth

# Percentage format for probability 'x' rounded to 2 decimal points
# for text in output picture legend say 45.33%
def percent(x):
    return str(round(100*x, 2)) + '%'

# Wealth amount format with K, M, B and one decimal point
# to simplify output and make legend less cluttered
# K = 1,000, M = 1,000,000, B = 1,000,000,000
def form(x):
    if x < 10**3:
        return f"{x:.1f}"
    if 10**3 <= x < 10**6: # 1.2K not 1236
        return f"{10**(-3)*x:.1f}K"
    if 10**6 <= x < 10**9: # 15.2M not 15192124
        return f"{10**(-6)*x:.1f}M"
    if 10**9 <= x: # 24.7B not 24694M
        return f"{10**(-9)*x:.1f}B"

# This function is necessary to make the same K, M, B for y-axis formatting
# for the plot of wealth evolution
def tickFormat(x, pos):
    return form(x)

# Vertical lines on the graph of simulations
def allTicks(horizon):
    if horizon < 10:
        return range(horizon + 1) # if less than 10 years make all lines visible
    else: # make a line visible every 5 years, including the start
        step = int(horizon/5) # how many lines with 5-year intervals
        if horizon - 5 * step > 2: # horizon = 14, then lines = 0, 5, 10, 14
            return np.append(np.array(range(6))*step, [horizon])
        else: # horizon = 12, then lines = 0, 5, 12
            return np.append(np.array(range(5))*step, [horizon])

# text for legend: setup part, where we explain in words the inputs
# output will be created after simulation in the next function
# need to print this in the legend to the right of the main picture
# to remind the investor about their inputs
# the arguments are the same as for 'simWealth' except initial volatility
def setupText(initialWealth, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare):

    # This part is text description of flow (contributions or withdrawals)
    # Initial value for year 1 and rate of annual increase/decrease
    if initialFlow == 0:
        initialFlowText = 'No regular contributions or withdrawals'
        growthText = ''
    # case when contributions
    if initialFlow > 0:
        initFlow = form(initialFlow)
        if growthFlow == 0: # no change in contributions from year to year
            initialFlowText = 'Constant contributions ' + initFlow
            growthText = ''
        else:
            initialFlowText = 'Initial contributions ' + initFlow
            if growthFlow > 0:
                growthText = 'annual increase in contributions ' + percent(growthFlow)
            if growthFlow < 0:
                growthText = 'annual decrease in contributions ' + percent(abs(growthFlow))

    # case when withdrawals
    if initialFlow < 0:
        initFlow = form(abs(initialFlow))
        if growthFlow == 0: # no change in withdrawals from year to year
            initialFlowText = 'Constant withdrawals ' + initFlow
            growthText = ''
        else:
            initialFlowText = 'Initial withdrawals ' + initFlow
            if growthFlow > 0:
                growthText = 'annual increase in withdrawals ' + percent(growthFlow)
            if growthFlow < 0:
                growthText = 'annual decrease in withdrawals ' + percent(abs(growthFlow))

    # text output explaining portfolio weights
    # for example 33% American: S&P 500 and 67% International: MSCI EAFE
    # At the start, 60% stocks and 40% bonds, at the end, 90% stocks and 10% bonds
    usText = 'Stocks: ' + percent(1 - intlShare) + ' American: S&P 500'
    intlText = 'and ' + percent(intlShare) + ' International: MSCI EAFE'
    initText = 'Portfolio: Stocks and USA corporate bonds'
    startText = 'At the start: ' + percent(1 - bondShare0) + ' Stocks ' + percent(bondShare0) + ' Bonds'
    endText = 'At the end: ' + percent(1 - bondShare1) + ' Stocks ' + percent(bondShare1) + ' Bonds'

    # number of simulations, convert to string
    simText = str(NSIMS) + ' Monte Carlo simulations'

    # number of years in time horizon
    timeText = 'Time Horizon: ' + str(timeHorizon) + ' years'

    # initial wealth
    initWealthText = 'Initial Wealth ' + form(initialWealth)

    # return all these texts combined
    texts = [simText, usText, intlText, initText, startText, endText, timeText + ' ' + initWealthText, initialFlowText + ' ' + growthText]

    # combine all these texts and return combined text
    return 'SETUP: ' + '\n'.join(texts)

# setup text for advanced simulator
def advancedText(initVol, initBubble, initRate, initSpread):
    textMain = 'INITIAL CONDITIONS FOR MODEL FACTORS:'
    textVol = 'Daily average over last year S&P 500 Volatility Index ' + str(round(initVol, 2))
    textBubble = 'Last year S&P 500 Bubble Valuation Measure ' + str(round(initBubble, 2))
    textRate = 'Daily average over last month Moody BAA Bond Rate ' + str(round(initRate, 2))
    textSpread = 'Daily average over last month 10 Year - 3 Month Spread ' + str(round(initSpread, 2))
    texts = [textMain, textVol, textBubble, textRate, textSpread]
    return '\n'.join(texts)

# Function creating the graph when click Compute
# Perform simulations and draw them on a picture
# Select NDISPLAYS paths ranked by final wealth
# This includes paths which end in zero wealth (ruin, bankruptcy)
# Write a legend for each path, and the overall legend
# Including setup in the above function and the results
# The arguments are the same as for the function 'simWealth'
# advanced = True if we are in complete simulator page
# advanced = False if we are in the main simulator page

def output(initVol, initBubble, initRate, initSpread, initialW, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare, advanced):

    # case when we are in advanced simulator in complete page
    if advanced:
        # simulate wealth process of a portfolio
        pathData, paths = simWealth(initVol, initBubble, initRate, initSpread, currEarn, initialW, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare)

    # case when we are in the main simulator in landing page
    if not advanced:
        # simulate wealth process of a portfolio
        pathData, paths = simWealth(currVol, currBubble, currRate, currSpread, currEarn, initialW, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare)

    # take average total portfolio return over each path
    # and pick paths which do not end in bankruptcy (ruin)
    # average these averaged return over such paths
    allAvgReturns = [pathData[sim] for sim in range(NSIMS) if paths[sim, -1] > 0]
    # pick paths which do end in ruin and average ruin time over them
    allRuinTimes = [pathData[sim] for sim in range(NSIMS) if paths[sim, -1] == 0]

    if len(allAvgReturns) > 0:
        avgRet = np.mean(allAvgReturns)
        AvgRetText = 'average over paths without ruin time averaged returns: ' + percent(avgRet)
    if len(allAvgReturns) == 0:
        AvgRetText = 'all paths end in bankruptcy'
    if len(allRuinTimes) > 0:
        avgTime = np.mean(allRuinTimes)
        AvgRuinText = 'average ruin time for paths in ruin: ' + str(round(avgTime, 2))
    if len(allRuinTimes) == 0:
        AvgRuinText = 'no paths end in bankruptcy'

    wealthMean = np.mean(paths[:, -1]) # average final wealth over paths

    # share of paths which end in bankruptcy (ruin) = zero wealth
    ruinProb = np.mean([paths[sim, -1] == 0 for sim in range(NSIMS)])

    # sort all paths by final wealth from bottom to top
    sortedIndices = np.argsort(paths[:, -1])

    # indices for selected paths ranked by final wealth
    # NDISPLAYS = number of displayed paths on the main image
    # equidistant by ranks of final wealth
    selectedIndices = [sortedIndices[int(NSIMS*item)] for item in selectedPercentages]

    # all time points: 0, 1, ..., timeHorizon
    times = range(timeHorizon + 1)
    RuinProbText = 'Ruin Probability ' + percent(ruinProb)
    MeanText = 'average final wealth ' + form(wealthMean)
    ResultText = 'RESULTS: ' + RuinProbText + '\n' + AvgRetText + '\n' + MeanText + '\n' + AvgRuinText

    # text for setup which is in the main legend for the plot
    # so that user sees output image in a different page than inputs
    # and does not forget these inputs
    SetupText = setupText(initialW, initialFlow, growthFlow, timeHorizon, bondShare0, bondShare1, intlShare)

    # if we are on complete simulator page
    if advanced:
        # initial condition text
        initialText = advancedText(initVol, initBubble, initRate, initSpread)

        # add this initial condition text to the setup text
        SetupText = SetupText + '\n\n' + initialText

    # this plot of only one point in white color is necessary for big legend
    # because it serves as its anchor
    plt.plot([0], [initialW], color = 'w', label = SetupText + '\n\n' + ResultText)

    # next show plots of wealth paths
    for display in range(NDISPLAYS):
        index = selectedIndices[display]

        # text shows final wealth and its % rank
        rankText = ' final wealth, ranked ' + percent(selectedPercentages[display])
        endWealth = paths[index, -1]

        if (endWealth == 0): # this path ended with zero wealth
            pathLabel = '0' + rankText + '; bankruptcy in ' + str(int(pathData[index])) + ' years'
        else: # this path ends with positive wealth
            pathLabel = form(endWealth) + rankText + '; time averaged returns: ' + percent(pathData[index])
        plt.plot(times, paths[index], label = pathLabel) # plot this given path

    plt.gca().set_facecolor('ivory') # background plot color
    plt.xlabel('Years') # label of the X-axis, for time

    ticks = allTicks(timeHorizon) # make vertical lines selected years
    plt.xticks(ticks) # this uses the x-axis for these vertical lines
    plt.gca().set_ylabel('Wealth')

    # and for the y-axis, we format it according to the K, M, B format
    # using the function 'form' and 'tickFormat' above
    plt.gca().yaxis.set_major_formatter(FuncFormatter(tickFormat))

    plt.title('Wealth Plot') # title of the entire figure

    # Choose font size for the legend, depending on the mode: main or advanced
    if advanced:
        # For the advanced version of the simulator, we have more setup data
        # since we choose initial conditions ourselves
        # so font needs to be smaller
        legendSize = 11
    else:
        # For the main version of the simulator, we have less setup data
        # since we do not choose initial conditions
        # so font is larger
        legendSize = 12

    # properties of legend: location relative to the anchor above
    # font size and background color
    plt.legend(bbox_to_anchor = (1, 1.1), loc = 'upper left', prop = {'size': legendSize}, facecolor = 'azure')
    plt.grid(True) # make vertical and horizontal grid

    # save to folder 'static' to present in output page below
    plt.savefig(outputPNG, bbox_inches = 'tight')
    plt.savefig(outputPDF, bbox_inches = 'tight', format = 'pdf')
    plt.close()

def readInput():

    # initial and terminal share of bonds in portfolio, converted from %
    bond0 = float(request.form['bond0'])*0.01 # initial
    bond1 = float(request.form['bond1'])*0.01 # terminal

    # share of international among stocks, converted from %
    intl = float(request.form['intl'])*0.01

    # number of years and initial wealth for simulation
    nYears = int(request.form['years'])
    initialWealth = float(request.form['initWealth'])

    # Do you withdraw = '-1' or contribute = '+1' annually?
    action = int(request.form.get('action'))

    # Do you annually increase = '+1' or decrease = '-1' these amounts
    # of withdrawals and contributions?
    change = int(request.form.get('change'))

    # First year contributions or withdrawals
    initialFlow = float(request.form['initialFlow'])*action

    # Annual change amount for withdrawals or contributions converted from %
    growthFlow = float(request.form['growthFlow'])*0.01*change

    return bond0, bond1, intl, nYears, initialWealth, initialFlow, growthFlow

# main landing page for the main version of the simulator
@app.route('/')
def mainPage():
    return render_template("main_page.html")

# landing page for the complete version of the simulator
@app.route('/complete')
def completePage():
    return render_template("complete_page.html")

# function which downloads the results in a PDF file when clicking on a link
@app.route('/download')
def downloadFile():
    return send_file(outputPDF, as_attachment=True)

# main function executing when click Submit
# differs from the main landing page by /compute
@app.route('/compute', methods=["POST"])
def mainComputation():

    # read the input data on portfolio and actions
    bond0, bond1, intl, nYears, initialWealth, initialFlow, growthFlow = readInput()

    # Draw the PNG picture with simulation results and graphs
    output(currVol, currBubble, currRate, currSpread, initialWealth, initialFlow, growthFlow, nYears, bond0, bond1, intl, False)

    # the response page after clicking Submit, with this PNG picture
    return render_template('response_page.html')

# when lick Compute on advanced simulator HTML page
@app.route('/advancedCompute', methods=["POST"])
def advancedComputation():

    # initial conditions for model factors
    initVol = float(request.form.get('initVol')) * (10.37/19.4645) # volatility
    initBubble = float(request.form.get('initBubble')) # bubble measure
    initRate = float(request.form.get('initRate')) # BAA bond rate
    initSpread = float(request.form.get('initSpread')) # long-short spread

    # read input for portfolio and actions
    bond0, bond1, intl, nYears, initialWealth, initialFlow, growthFlow = readInput()

    # Draw the PNG picture with simulation results and graphs
    output(initVol, initBubble, initRate, initSpread, initialWealth, initialFlow, growthFlow, nYears, bond0, bond1, intl, True)

    # the response page after clicking Submit, with this PNG picture
    return render_template('response_page.html')