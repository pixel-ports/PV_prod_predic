from flask import Flask
from flask_restful import Resource, Api
from subprocess import Popen
import time

app = Flask(__name__)
api = Api(app)

class inference(Resource):
    def get(self):
        command = "python3 inference.py ./trained.scaler ./trained.model ./one_sequence.csv"
        begin_time = time.time()
        worker_process = Popen(command.split(" "))
        worker_process.wait()
        worker_process.terminate()
        return time.time() - begin_time

api.add_resource(inference, '/')

if __name__ == '__main__':
    app.run(debug=True)
    