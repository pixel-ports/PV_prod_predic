from training import get_data, split_sequences, MV_LSTM
import torch
from torch.autograd import Variable
import argparse


def preprocess_data(data, features, scaler):
    daily_data_with_weather = data[features]
    daily_data_with_weather = daily_data_with_weather.astype('float32')
    daily_data_with_weather = daily_data_with_weather.dropna()
    scaled = scaler.transform(daily_data_with_weather)
    return scaled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer an LSTM model for PV prediction.')
    parser.add_argument('scaler_path', type=str, help='Path to store the scaler on disk')
    parser.add_argument('model_path', type=str, help='Path to store the model on disk')
    parser.add_argument('data_path', type=str, help='Path of the data to load')
    args = parser.parse_args()

    features = ['Energy_Generated'] # simple LSTM
    n_features = len(features) # this is number of parallel inputs
    n_timesteps = 48 # this is number of timesteps

    data = get_data(args.data_path)
    scaler = torch.load(args.scaler_path)
    scaled = preprocess_data(data, features, scaler)
    X, y = split_sequences(scaled, n_timesteps)
    testX = Variable(torch.Tensor(X))

    mv_net_2 = MV_LSTM(n_features,n_timesteps)
    mv_net_2.load_state_dict(torch.load(args.model_path))

    mv_net_2.init_hidden(testX.size(0))
    result = mv_net_2(testX).detach().numpy()

    print(result)
