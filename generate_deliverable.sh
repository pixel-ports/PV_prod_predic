#!/bin/bash
version=1.0
arg_version=$1

#Create a zip archive for PV_production notebook
zip PV_Prediction_v"${arg_version:-$version}"_deliverable.zip LICENSE README.md requirements.txt DATA_COLLECTOR/AGGREGATION_PRODUCTION/aggreg_PRODUCTION_data.csv MODEL_PREDIC_BENCHMARK/PV_Prediction.ipynb