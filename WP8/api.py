from flask import Flask
from flask_restful import Resource, Api
from subprocess import Popen
import time
from training import get_data, split_sequences, MV_LSTM
from inference import preprocess_data
import torch
from torch.autograd import Variable


scaler_path = "./trained.scaler"
model_path = "./trained.model"
data_path = "./one_sequence.csv"

features = ['Energy_Generated'] # simple LSTM
n_features = len(features) # this is number of parallel inputs
n_timesteps = 48 # this is number of timesteps


scaler = torch.load(scaler_path)

mv_net_2 = MV_LSTM(n_features,n_timesteps)
mv_net_2.load_state_dict(torch.load(model_path))

# Loading here is only usefull to get the size of testX to init the hidden layer 
data = get_data(data_path)
scaled = preprocess_data(data, features, scaler)
X, y = split_sequences(scaled, n_timesteps)
testX = Variable(torch.Tensor(X))

mv_net_2.init_hidden(testX.size(0))

app = Flask(__name__)
api = Api(app)


class inference(Resource):
    def get(self):
        command = "python3 inference.py %s %s %s" % (scaler_path, model_path, data_path)
        begin_time = time.time()
        worker_process = Popen(command.split(" "))
        worker_process.wait()
        worker_process.terminate()
        return time.time() - begin_time

class loaded_inference(Resource):
    def get(self):  # We reload data to simulate incoming data
        begin_time = time.time()
        # data = get_data(data_path)
        # scaled = preprocess_data(data, features, scaler)
        # X, _ = split_sequences(scaled, n_timesteps)
        # testX = Variable(torch.Tensor(X))
        result = mv_net_2(testX).detach().numpy()
        return time.time() - begin_time


api.add_resource(inference, '/')
api.add_resource(loaded_inference, '/loaded_inference/')


if __name__ == '__main__':
    app.run(threaded=True)
    