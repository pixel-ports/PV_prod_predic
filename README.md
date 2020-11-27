## WP4 > T4.2 > Photovoltaic production prediction

#### MAIN GOAL :
Predict a PV installation future production, knowing:
- previous data on production,
- previous weather condition.

We are using real data from PVOutput (https://pvoutput.org/) to build PV prediction models.

This repository gather tools for: 
- Data acquisition : extract detailed and aggregated production data from PVOutput, in DATA_COLLECTOR,
- Data analysis : analyse the aggregated time serie trend and seasonality, in SAISO_ANALYSE,
- Prediction : benchmark several time series forecasting methods, in MODEL_PRED ICT_BENCHMARK.

Run `generate_deliverable.sh` to generate the zip file containing WP4 final deliverable 
(notebook, data, License, requirements and README) in the DELIVERABLE folder.

Requires Python >= 3.6

## Statistics for WP8 Product Quality Model

```bash
pip3 install psutil dask[complete] flask-restful
sudo apt-get install valgrind

cd ./WP8

# Monitoring RAM
rm ./massif.out.*
valgrind --tool=massif --time-unit=ms python3 training.py ./trained.scaler ./trained.model ./multiple_sequences.csv
python3 ./massif_analyser.py $(ls -1 -v ./massif.out.* | tail -n 1)
rm ./massif.out.*
valgrind --tool=massif --time-unit=ms python3 inference.py ./trained.scaler ./trained.model ./one_sequence.csv
python3 ./massif_analyser.py $(ls -1 -v ./massif.out.* | tail -n 1)

# Monitoring CPU
python3 ./monitor_cpu.py "python3 training.py ./trained.scaler ./trained.model ./multiple_sequences.csv"
python3 ./monitor_cpu.py "python3 inference.py ./trained.scaler ./trained.model ./one_sequence.csv"

# Monitoring simultaneous requests performance
python3 api.py
python3 simultaneous_requests.py --min_processes 1 --max_processes 51 --step_processes 5 "/"
python3 simultaneous_requests.py --min_processes 1 --max_processes 51 --step_processes 5 "/loaded_inference/"


# Throughput
time python3 training.py ./trained.scaler ./trained.model ./multiple_sequences.csv
time python3 inference.py ./trained.scaler ./trained.model ./one_sequence.csv
time python3 inference.py ./trained.scaler ./trained.model ./multiple_sequences.csv
```
