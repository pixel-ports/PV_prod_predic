#!/bin/bash
version=1.0
arg_version=$1

#Create a zip archive for PV_production notebook
zip -j DELIVERABLE/PV_Prediction_v"${arg_version:-$version}"_deliverable.zip LICENSE requirements.txt DELIVERABLE/README.md DATA_COLLECTOR/AGGREGATION_PRODUCTION/aggreg_PRODUCTION_data.csv MODEL_PREDIC_BENCHMARK/PV_Prediction.ipynb