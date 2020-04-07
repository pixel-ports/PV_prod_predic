#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
sns.set(rc={'figure.figsize':(11, 4)})


# read data and keep the data from Dec,31 2012.

# In[29]:


data = pd.read_pickle("../DATA_COLLECTOR/AGGREGATION_PRODUCTION/aggreg_PRODUCTION_data.pkl")
data.Date = pd.to_datetime(data['Date'],format='%Y%m%d')
data = data.sort_values(by='Date')
data = data[data.Date > '2012-12-31']
data = data.set_index('Date')

print(data.tail())
print(len(data))


data.Energy_Generated = pd.to_numeric(data.Energy_Generated)
data.Energy_Generated.plot()


# Split data into train and test

# In[3]:


thresh_date = '2018-12-31' #'2019-02-07', '2019-08-31'
train_data = data[data.index < thresh_date]
test_data = data[data.index > thresh_date]
print(train_data.head(2))
print(test_data.head(2))

print("Train size %d, test size %d" % (len(train_data), len(test_data)))



# LSTM 

# In[30]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# In[31]:


conditions = data.Condition.values


# In[32]:


data = data.reset_index()
daily_data_with_weather = data[['Date','Condition','Energy_Generated']]
#daily_data_with_weather['Energy_Generated'] = np.log(daily_data_with_weather['Energy_Generated'])
conditions = set(daily_data_with_weather['Condition'])
print(daily_data_with_weather.tail())
encoder = LabelEncoder()
daily_data_with_weather.iloc[:,1] = encoder.fit_transform(daily_data_with_weather.iloc[:,1])
le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print(le_name_mapping)
#X_enc = pd.get_dummies(X_enc, columns=['Condition'])
#enc = OneHotEncoder(handle_unknown='ignore')
#enc.fit_transform(daily_data_with_weather.iloc[:,0])
#print(enc.categories_)

# ensure all data is float
daily_data_with_weather[['Condition','Energy_Generated']] = daily_data_with_weather[['Condition','Energy_Generated']].astype('float32')
daily_data_with_weather = daily_data_with_weather.dropna()

daily_data_with_weather = daily_data_with_weather[daily_data_with_weather.Condition != 3]
print(daily_data_with_weather.tail())


from sklearn.preprocessing import MinMaxScaler

def create_sliding_window(data, sequence_length, stride=1):
    X_list, y_list = [], []
    for i in range(len(data)):
      if (i + sequence_length) < len(data):
        X_list.append(data.iloc[i:i+sequence_length:stride, :].values)
        y_list.append(data.iloc[i+sequence_length, -1])
    return np.array(X_list), np.array(y_list)

train_split = 0.8
n_train = int(train_split * daily_data_with_weather.shape[0])
n_test = daily_data_with_weather.shape[0] - n_train

features = ['Condition','Energy_Generated']
features = ['Energy_Generated']
feature_array = daily_data_with_weather[features].values
print('feature_array',feature_array[0])
## Fit Scaler only on Training features
feature_scaler = MinMaxScaler(feature_range=(0, 1))
feature_scaler.fit(feature_array[:n_train])
## Fit Scaler only on Training target values
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(feature_array[:n_train, -1].reshape(-1, 1))

print(feature_array)
# Transfom on both Training and Test data
scaled_array = pd.DataFrame(feature_scaler.transform(feature_array),
                            columns=features)
sequence_length =365 
X, y = create_sliding_window(scaled_array, 
                             sequence_length)

X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]
print(X_train)

# LSTM Architecture
# Encoder-Decoder Stage:
# 
# 1 uni-directional LSTM layer with 128 hidden units acts as an encoding layer to construct a fixed-dimension embedding state
# 1 uni-directional LSTM layer with 32 hidden units acts as a decoding layer to produce predictions at future steps
# Dropout is applied at both training and inference for both LSTM layers
# 
# Predictor Stage:
# 1 fully-connected output layer with 1 output (for predicting the target value) to produce a single value for the target variable
# By allowing dropout at both training and testing time, the model simulates random sampling, thus allowing varying predictions that can be used to estimate the underlying distribution of the target value, enabling explicit model uncertainties.

# In[41]:


class LSTM(nn.Module):

    def __init__(self, n_features, output_length,seq_length):

        super(LSTM, self).__init__()

        self.hidden_size_1 =128 
        self.seq_length = seq_length
        self.hidden_size_2 =64 
        self.n_layers = 1 # number of (stacked) LSTM layers

        self.lstm1 = nn.LSTM(n_features, 
                             self.hidden_size_1, 
                             num_layers=1,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=1,
                             batch_first=True)
        
        self.dense = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=0.5, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=0.5, training=True)
        output = self.dense(state[0].squeeze(0))
        
        return output

    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        return hidden_state.cuda(), cell_state.cuda()
    
    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_2))
        return hidden_state.cuda(), cell_state.cuda()
    
    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).cuda()
        return self(X).view(-1).cpu().detach().numpy()




n_features = scaled_array.shape[-1]
output_length = 1

bayesian_lstm = LSTM(n_features=n_features,
                             output_length=output_length,seq_length=sequence_length)
bayesian_lstm.cuda()
optimizer = torch.optim.Adam(bayesian_lstm.parameters(), lr=0.01,weight_decay=0.001)
criterion = torch.nn.MSELoss().cuda() 
batch_size =128 
n_epochs = 400


print("begain training")
loss_values = []
val_loss_values = []
for e in range(1, n_epochs+1):
    bayesian_lstm.train()
    epoch_loss = 0
    for b in range(0,len(X_train),batch_size):
        features = X_train[b:b+batch_size,:,:]
        target = y_train[b:b+batch_size]    
        X_batch = torch.tensor(features,dtype=torch.float32).cuda()
        y_batch = torch.tensor(target,dtype=torch.float32).cuda()

        output = bayesian_lstm(X_batch) 
        loss = criterion(output.view(-1), y_batch)
        epoch_loss = epoch_loss + loss

        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
    loss_values.append(epoch_loss)
    
    ##add test
    bayesian_lstm.eval()
    X_test = torch.tensor(X_test,dtype=torch.float32).cuda()
    y_test =torch.tensor(y_test,dtype=torch.float32).cuda() 
    test_predict = bayesian_lstm(X_test)
    val_loss = criterion(test_predict.view(-1), y_test)  
    val_loss_values.append(val_loss.detach())
    if e % 10 == 0:
        print("%d, loss %f val_loss %f" % (e,loss_values[-1],val_loss)) 


# Evaluating Model Performance
# The Bayesian LSTM implemented is shown to produce reasonably accurate and sensible results on both the training and test sets, often comparable to other existing frequentist machine learning and deep learning methods.

# In[79]:


offset = sequence_length

def inverse_transform(data_predict):
    #data_predict = np.c_[data_predict, np.zeros(data_predict.shape[0]) ]
    data_predict = target_scaler.inverse_transform(data_predict.reshape(-1, 1))
    return data_predict

#data = data.reset_index()
training_df = pd.DataFrame()
training_df['date'] = daily_data_with_weather['Date'].iloc[offset:n_train + offset:1] 
training_predictions = bayesian_lstm.predict(X_train)
print(inverse_transform(training_predictions))
training_df['energy_generated'] = inverse_transform(training_predictions)
training_df['source'] = 'Training Prediction'
training_truth_df = pd.DataFrame()
training_truth_df['date'] = training_df['date']
training_truth_df['energy_generated'] = daily_data_with_weather['Energy_Generated'].iloc[offset:n_train + offset:1] 
training_truth_df['source'] = 'True Values'

testing_df = pd.DataFrame()
testing_df['date'] = daily_data_with_weather['Date'].iloc[n_train + offset::1] 
testing_predictions = bayesian_lstm.predict(X_test)
testing_df['energy_generated'] = inverse_transform(testing_predictions)
print(testing_df)
testing_df['source'] = 'Test Prediction'

testing_truth_df = pd.DataFrame()
testing_truth_df['date'] = testing_df['date']
testing_truth_df['energy_generated'] = daily_data_with_weather['Energy_Generated'].iloc[n_train + offset::1] 
testing_truth_df['source'] = 'True Values'

evaluation = pd.concat([training_df, 
                        testing_df,
                        training_truth_df,
                        testing_truth_df
                        ], axis=0)


# In[80]:


import plotly.express as px
fig = px.line(evaluation.loc[evaluation['date'].between('2019-04-14', '2019-04-23')],
                 x="date",
                 y="energy_generated",
                 color="source",
                 title="Energy production in Wh vs Time")
fig.show()


# Uncertainty Quantification
# The fact that stochastic dropouts are applied after each LSTM layer in the Bayesian LSTM enables users to interpret the model outputs as random samples from the posterior distribution of the target variable.
# 
# This implies that by running multiple experiments/predictions, can approximate parameters of the posterioir distribution, namely the mean and the variance, in order to create confidence intervals for each prediction.
# 
# We construct 99% confidence intervals that are three standard deviations away from the approximate mean of each prediction.

# In[81]:


n_experiments = 100

test_uncertainty_df = pd.DataFrame()
test_uncertainty_df['date'] = testing_df['date']

for i in range(n_experiments):
  experiment_predictions = bayesian_lstm.predict(X_test)
  test_uncertainty_df['energy_production_{}'.format(i)] = inverse_transform(experiment_predictions)

energy_production_df = test_uncertainty_df.filter(like='energy_production', axis=1)
print(energy_production_df)
test_uncertainty_df['energy_production_mean'] = energy_production_df.mean(axis=1)
test_uncertainty_df['energy_production_std'] = energy_production_df.std(axis=1)

test_uncertainty_df = test_uncertainty_df[['date', 'energy_production_mean', 'energy_production_std']]


# In[82]:


test_uncertainty_df['lower_bound'] = test_uncertainty_df['energy_production_mean'] - 3*test_uncertainty_df['energy_production_std']
test_uncertainty_df['upper_bound'] = test_uncertainty_df['energy_production_mean'] + 3*test_uncertainty_df['energy_production_std']
print(test_uncertainty_df)

# In[84]:


import plotly.graph_objects as go

test_uncertainty_plot_df = test_uncertainty_df.copy(deep=True)
test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2018-05-01', '2018-05-09')]
truth_uncertainty_plot_df = testing_truth_df.copy(deep=True)
truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2018-05-01', '2018-05-09')]

upper_trace = go.Scatter(
    x=test_uncertainty_plot_df['date'],
    y=test_uncertainty_plot_df['upper_bound'],
    mode='lines',
    fill=None,
    name='99% Upper Confidence Bound'
    )
lower_trace = go.Scatter(
    x=test_uncertainty_plot_df['date'],
    y=test_uncertainty_plot_df['lower_bound'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 211, 0, 0.5)',
    name='99% Lower Confidence Bound'
    )
real_trace = go.Scatter(
    x=truth_uncertainty_plot_df['date'],
    y=truth_uncertainty_plot_df['energy_generated'],
    mode='lines',
    fill=None,
    name='Real Values'
    )

data = [upper_trace, lower_trace, real_trace]

fig = go.Figure(data=data)
fig.update_layout(title='Uncertainty Quantification for Energy Consumption Test Data',
                   xaxis_title='Time',
                   yaxis_title='log_energy_consumption (log Wh)')

fig.show()


# Evaluating Uncertainty
# Using multiple experiments above, 99% confidence intervals have been constructed for each the prediction of the target variable (the logarithm of appliance power consumption). While we can visually observe that the model is generally capturing the behavior of the time-series, approximately only 50% of the real data points lie within a 99% confidence interval from the mean prediction value.
# 
# Despite the relatively low percentage of points within the confidence interval, it must be noted that Bayesian Neural Networks only seek to quantify the epistemic model uncertainty and does not account for aleatoric uncertainty (i.e. noise).

# In[85]:


bounds_df = pd.DataFrame()

# Using 99% confidence bounds
bounds_df['lower_bound'] = test_uncertainty_plot_df['lower_bound']
bounds_df['prediction'] = test_uncertainty_plot_df['energy_production_mean']
bounds_df['real_value'] = truth_uncertainty_plot_df['energy_generated']
bounds_df['upper_bound'] = test_uncertainty_plot_df['upper_bound']

print(bounds_df)
bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                          (bounds_df['real_value'] <= bounds_df['upper_bound']))

print("Proportion of points contained within 99% confidence interval:", 
      bounds_df['contained'].mean())





