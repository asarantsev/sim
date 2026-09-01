Code and data for the article Valuation Measure of the Stock Market using Stochastic Volatility and Stock Earnings https://arxiv.org/abs/2508.06010 

data.xlsx: Main data file

innovations.xlsx: Residuals of 7 regressions of the main model with missing data, except the valuation measure regression, after main.py

filled.xlsx: Residuals of 7 regressions of the main model with filled missing data file, after applying innovations.py

corrMatrix.xlsx: Correlation matrix for 8 residuals of the main model, after main.py

MonteCarlo.py: Do Monte Carlo simulation to find critical values for skewness, kurtosis, and L1 for ACF, Section 3

simple.py: Fit the simple model, Section 4

bubble.py: Fit the valuation measure, Section 5

window.py: Choose the averaging window for earnings, Section 5

main.py: Fit the main model, Section 6

innovations.py: Filling missing data for innovations to turn innovations.xlsx into filled.xlsx, Section 7

flask_app.py: Python/Flask back end of the simulator, Section 7

main_page.html: Landing page front end for the simulator

complete_page.html: Front end: version of the simulator with an option to change initial factors

response_page.html: Front end: page after Submit

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

UPDATE Sep 1, 2026: We apply the Box-Cox transforms to one-dimensional stationary data series to see which one can be made Gaussian. 

box-cox-vol.py discusses volatility and application of Box-Cox transforms to it. We succeed in making it Gaussian. 

box-cox-resid.py applies Box-Cox transforms to each other non-Gaussian series of residuals. We succeeded for some and failed for spread and earnings growth. Thus we cannot hope to normalize residuals by simply applying these transforms. 

box-cox-main.py applies the Box-Cox transform to the following series (with index indicated): BAA rate (successful, -0.2425); exponentiated long-short term spread (successful, 0.1353); exponentiated earnings growth (failed); exp(log US returns) (successful, 1.8403); exp(log intl returns) (successful, 0.8599); exp(log bond returns) (successful, -0.2881). 
