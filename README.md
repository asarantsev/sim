data.xlsx: Main data file
innovations.xlsx: Residuals of 7 regressions of the main model with missing data, except the valuation measure regression, after main.py
filled.xlsx: Residuals of 7 regressions of the main model with filled missing data file, after applying innovations.py
corrMatrix.xlsx: Correlation matrix for 8 residuals of the main model
flask_app.py: Python/Flask back end
main.py: Fitting the main model, Section 6
simple.py: Fitting the simple model, Section 4
bubble.py: Fitting the valuation measure, Section 5
innovations.py: Filling missing data for innovations
main_page.html: Landing page front end for the simulator
complete_page.html: Front end: version of the simulator with an option to change initial factors
response_page.html: Front end: page after Submit
