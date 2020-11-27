import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.autograd import Variable

import argparse

np.random.seed(0)
torch.manual_seed(0)


def get_data(pvdata_file):
    data = pd.read_csv(pvdata_file)
    data.Date = pd.to_datetime(data['Date'],format='%Y%m%d')
    data = data.sort_values(by='Date')
    data = data[data.Date > '2012-12-31']
    data = data.set_index('Date')

    data.Energy_Generated = pd.to_numeric(data.Energy_Generated)
    data.Energy_Generated.plot(title="Energy generated (kWh)")

    return data

def preprocess_data(data, features):
    daily_data_with_weather = data[features]

    train_size = int(len(daily_data_with_weather) * 0.8)
    test_size = len(daily_data_with_weather) - train_size
    print(train_size,test_size)

    # ensure all data is float
    daily_data_with_weather = daily_data_with_weather.astype('float32')
    daily_data_with_weather = daily_data_with_weather.dropna()

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(daily_data_with_weather.iloc[:train_size])
    scaled = scaler.transform(daily_data_with_weather)

    return scaled, scaler, train_size, test_size

def train(scaled, features, train_size):
    n_features = len(features) # this is number of parallel inputs
    n_timesteps = 48 # this is number of timesteps

    # convert dataset into input/output
    X, y = split_sequences(scaled, n_timesteps)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:] 

    # create NN
    mv_net = MV_LSTM(n_features,n_timesteps)
    criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-1,weight_decay=0.001)
    train_episodes = 200
    batch_size = 128

    trainX = Variable(torch.Tensor(X_train))
    trainY = Variable(torch.Tensor(y_train))

    testX = Variable(torch.Tensor(X_test))
    testY = Variable(torch.Tensor(y_test))

    loss_values = []
    val_loss_values = []
    for t in range(train_episodes):
        mv_net.train()
        for b in range(0,len(trainX),batch_size):
            inpt = trainX[b:b+batch_size,:,:]
            target = trainY[b:b+batch_size]    

            x_batch = torch.tensor(inpt,dtype=torch.float32)    
            y_batch = torch.tensor(target,dtype=torch.float32)

            mv_net.init_hidden(x_batch.size(0))
            output = mv_net(x_batch) 
            loss = criterion(output.view(-1), y_batch)  

            loss.backward()
            optimizer.step()        
            optimizer.zero_grad() 
            
        
        loss_values.append(loss)
            
        ##add test
        mv_net.eval()
        mv_net.init_hidden(testX.size(0))
        test_predict = mv_net(testX)
        loss = criterion(test_predict.view(-1), testY)  
        val_loss_values.append(loss)
        if t % 10 == 0:
            print("%d, loss %f val_loss %f" % (t,loss_values[-1],loss))
    
    return mv_net


class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length,n_output=1,bias_init_values=[]):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20 # number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)
        self.n_output = n_output

        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, self.n_output)
    
        if bias_init_values :
            for i in range(self.n_output):
                self.l_linear.bias.data[i] = bias_init_values[i]
               


    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)


    def forward(self, x):        
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    print("len(sequences) : %d" % len(sequences))
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LSTM model for PV prediction.')
    parser.add_argument('scaler_path', type=str, help='Path to store the scaler on disk')
    parser.add_argument('model_path', type=str, help='Path to store the model on disk')
    parser.add_argument('data_path', type=str, help='Path of the data to load')
    args = parser.parse_args()

    data = get_data(args.data_path)
    features = ['Energy_Generated'] # simple LSTM

    scaled, scaler, train_size, test_size = preprocess_data(data, features)
    torch.save(scaler, args.scaler_path)

    mv_net = train(scaled, features, train_size)
    torch.save(mv_net.state_dict(), args.model_path)
