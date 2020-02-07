import pandas as pd
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from utils import *

class HoltWinter:
        def __init__(self):
                self.seasonals = {}
                self.trend = 0.0
                self.serie_length = 0.0
                self.smooth = 0.0
                self.slen = 0.0

        def initial_trend(self, series, slen):
                sum = 0.0

                for i in range(slen):
                        sum += float(series[i+slen] - series[i]) / slen

                self.trend = sum / slen

        def initial_seasonal_components(self, series, slen):
                season_averages = []
                n_seasons = int(len(series)/slen)

                # compute season averages
                for j in range(n_seasons):
                        season_averages.append(sum(series[slen * j:slen * j + slen]) / float(slen))

                # compute initial values
                for i in range(slen):
                        sum_of_vals_over_avg = 0.0
                        for j in range(n_seasons):
                                sum_of_vals_over_avg += series[slen * j + i] - season_averages[j]
                                self.seasonals[i] = sum_of_vals_over_avg / n_seasons

        def fit(self, series, slen, alpha, beta, gamma):
                result = []
                self.slen = slen
                self.serie_length = len(series)
                seasonals = self.initial_seasonal_components(series, slen)

                for i in range(self.serie_length):
                        if i == 0: # initial values
                                self.smooth = series[0]
                                self.initial_trend(series, slen)
                                result.append(series[0])
                                continue

                        val = series[i]
                        last_smooth, self.smooth = self.smooth, alpha * (val - self.seasonals[i%slen]) + (1 - alpha) * (self.smooth + self.trend)
                        if beta != 0.0:
                                self.trend = beta * (self.smooth - last_smooth) + (1 - beta) * self.trend
                        else:
                                self.trend = 0.0
                        self.seasonals[i%slen] = gamma * (val - self.smooth) + (1 - gamma) * self.seasonals[i%slen]

                        result.append(self.smooth + self.trend + self.seasonals[i%slen])

#               print(self.smooth, self.trend, self.seasonals)
                return result

        def predict(self, npreds):
                result = []
                for i in range(npreds):
                        m = i + self.serie_length
                        result.append(self.smooth + m * self.trend + self.seasonals[m%self.slen])

                return result
class TrendNet(nn.Module):
        def __init__(self, feature_size):
                super(TrendNet, self).__init__()

                # size of first hidden layer
                self.hidden_size1 = 16

                # size of third hidden layer
                self.hidden_size2 = 16

                # output size
                self.output_size = 1

                self.fc1 = nn.Linear(feature_size, self.hidden_size1)
                self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
                self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

        def forward(self, input_data):

                input_data = input_data.transpose(0,1)

                xout = self.fc1(input_data)
                xout = self.fc2(xout)
                xout = self.fc3(xout)

                return xout

class HoltWinterNN:
        def __init__(self, feature_len, alpha, gamma, slen):
                self.seasonals = {}
                self.serie_length = 0.0
                self.smooth = 0.0
                self.slen = slen
                self.alpha = alpha
                self.gamma = gamma

                self.neuralnet = TrendNet(feature_len)
                self.criterion = Differencial_SMAPE()
                self.optimizer = optim.SGD(self.neuralnet.parameters(), lr=0.001)

        def initial_seasonal_components(self, series):
                season_averages = []
                n_seasons = int(len(series) / self.slen)

                # compute season averages
                for j in range(n_seasons):
                        season_averages.append(sum(series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))

                # compute initial values
                for i in range(self.slen):
                        sum_of_vals_over_avg = 0.0
                        for j in range(n_seasons):
                                sum_of_vals_over_avg += series[self.slen * j + i] - season_averages[j]
                                self.seasonals[i] = sum_of_vals_over_avg / n_seasons

        def update_nn(self, losses, batch_size):
                loss = 0.0
                for i in range(int(len(losses) / batch_size)):
                        mean_loss = sum(losses[i * batch_size: (i+1)*batch_size]) / batch_size
                        self.optimizer.zero_grad()
                        mean_loss.backward()
                        loss += float(mean_loss)
                        self.optimizer.step()

                print(loss)

        def train_neural_network(self, list_series, list_features, epochs, batch_size):
                for j in range(epochs):
                        for i in range(len(list_series)):
                                serie = list_series[i]
                                features = list_features[i]
                                des, losses, _ = self.fit(serie, features)
                                self.update_nn(losses, batch_size)


        def fit(self, series, features):
                result = []
                normalized_serie = [0.0 for x in range(self.slen)]
                losses = []

                self.serie_length = len(series)

                seasonals = self.initial_seasonal_components(series)

                for i in range(self.serie_length):
                        if i == 0: # initial values
                                self.smooth = series[0]
                                result.append(series[0])
                                normalized_serie.append(series[0])
                                continue

                        val = series[i]
                        last_smooth, self.smooth = self.smooth, alpha * (val - self.seasonals[i%self.slen]) + (1 - alpha) * (self.smooth)
                        self.seasonals[i%self.slen] = gamma * (val - self.smooth) + (1 - gamma) * self.seasonals[i%self.slen]

                        vec_features = features[i][:,np.newaxis]
                        vec_features = torch.Tensor(vec_features)
#                       vec_features = torch.cat([vec_features, torch.Tensor(normalized_serie[-24:]).unsqueeze(1)], dim=0)
                        nntrend = self.neuralnet(vec_features)

                        normalized_serie.append(val - self.smooth - self.seasonals[i%self.slen])
                        result.append(float(self.smooth + nntrend + self.seasonals[i%self.slen]))
                        loss = self.criterion(nntrend, torch.Tensor([normalized_serie[-1]]).reshape((1,1)))
                        if float(loss) > 0.0:
                                losses.append(loss)

                return result, losses, normalized_serie

        def predict(self, npreds, features):
                result = []
                for i in range(npreds):
                        m = i + self.serie_length

                        vec_features = features[i][:,np.newaxis]
                        vec_features = torch.Tensor(vec_features)
#                       vec_features = torch.cat([vec_features, torch.Tensor(normalized_serie[-24:]).unsqueeze(1)], dim=0)
                        nntrend = self.neuralnet(vec_features)
                        print(nntrend)

                        # normalized_serie.append(nntrend)
                        val = float(self.smooth + nntrend + self.seasonals[m%self.slen])
                        result.append(val)
#                       last_smooth, self.smooth = self.smooth, alpha * (val - self.seasonals[i%slen]) + (1 - alpha) * (self.smooth)
#                       self.seasonals[i%slen] = gamma * (val - self.smooth) + (1 - gamma) * self.seasonals[i%slen]

                return result

if __name__ == '__main__':

        df, target = get_production_data()

        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])

        df = np.array(df)

        visit_serie = df[:, target]
        visit_features = df[:, 1:].squeeze()
        print(visit_features.shape)
        

        split = 0
        size = 400
        size_test = 72
        seasonality = 365
        alpha = 0.1
        gamma = 0.1

        list_series = []
        list_features = []
        while split + size < visit_serie.shape[0]:
                list_series.append(visit_serie[split:split+size])
                list_features.append(visit_features[split:split+size])
                split += size

        split = 1500

        hw = HoltWinterNN(28, alpha, gamma, seasonality)
        hw.train_neural_network(list_series, list_features, 5, 64)
        des, losses, normalized_serie = hw.fit(visit_serie[split:split+size], visit_features[split:split+size])
        # print(normalized_serie)
        predict = hw.predict(size_test, visit_features[split+size:split+size+size_test])

        fig, ax = plt.subplots()
        sns.lineplot(data=np.array(des), color='red', ax=ax)
        sns.lineplot(data=visit_serie[split:split+size], color='blue', ax=ax)
        fig.savefig("production_train.png")
        plt.close()

        fig, ax = plt.subplots()
        sns.lineplot(data=np.array(predict), color='red', ax=ax)
        sns.lineplot(data=visit_serie[split+size:split+size+size_test], color='blue', ax=ax)
        fig.savefig("production_test.png")
        plt.close()

