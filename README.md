#WP4 > T4.2 > Photovoltaic production prediction

###MAIN GOAL :
Predict a PV installation future production, knowing:
- previous data on production
- previous weather condition

We are using real data from PVOutput (https://pvoutput.org/) to build PV prediction models.

This repository gather tools for 
- Data acquisition : extract detailed and aggregated production data from PVOutput, in DATA_COLLECTOR,
- Data analysis : analyse the aggregated time serie trend and seasonality, in SAISO_ANALYSE,
- Prediction : benchmark several time series forecasting methods, in MODEL_PREDICT_BENCHMARK.

Run generate_deliverable.sh to generate the zip file containing WP4 final deliverable 
(notebook, data, License, requirements and README) in the DELIVERABLE folder.

Requires Python >= 3.6