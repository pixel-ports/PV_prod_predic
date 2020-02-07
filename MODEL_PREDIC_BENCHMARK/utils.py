import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class Differencial_SMAPE(nn.Module):

	def __init__(self):
		super(Differencial_SMAPE, self).__init__()

	def forward(self, true, predicted):
		epsilon = 0.1
		summ = torch.clamp(torch.abs(true) + torch.abs(predicted) + epsilon, min=(0.5 + epsilon))
		smape = torch.abs(predicted - true) / summ * 2.0

		return torch.sum(smape)

def get_production_data():
	# open file and normalize data
	data = pd.read_pickle("aggreg_PRODUCTION_data.pkl")
	data.Date = pd.to_datetime(data['Date'])
	data = data.sort_values(by='Date')
	data = data[data.Date > '2012-12-31']
	data=data.set_index('Date')
	data.Energy_Generated=pd.to_numeric(data.Energy_Generated)

	train_data = data[data.index < '2018-12-31']
	test_data =data[data.index > '2018-12-31']

	daily_data_with_weather = train_data[['Energy_Generated','Condition']]
	encoder = LabelEncoder()
	daily_data_with_weather.iloc[:,1] = encoder.fit_transform(daily_data_with_weather.iloc[:,1])
	weekly_data_mean = train_data.Energy_Generated.resample('W').mean().dropna()
	monthly_data_mean = train_data.Energy_Generated.resample('M').mean().dropna()
	# normalize features
	monthly_data_mean = monthly_data_mean.values.reshape(-1, 1) 

	#print(df)

	for col in ['Energy_Generated']:
		for attention in range(1,28):
			daily_data_with_weather[str(col) + '_' + str(attention)] = daily_data_with_weather[col].shift(attention).fillna(0.0)
	print(daily_data_with_weather.columns)

	return daily_data_with_weather, 0