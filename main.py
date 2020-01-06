from threading import Thread
from time import sleep

from collector import Collector
from predictor import Predictor
from flask import Flask, jsonify, request

c = Collector("menz.dynip.sapo.pt", "5433", "postgres", "postgres", "tripa123",
              'PNGEa0YJLxVmPZssX9hDKwu3lhRQmjsyH4bpDTBg7zM2NYYCDoGAR7vtZfQorq8k',
              'kseCG5XF731dbVAwZJHmT3g0po6NjedqyBvUohCnUcZlXhQjxk4B6q4A0jHRfW4C')

p = Predictor(c)

app = Flask(__name__)


def update_data_worker():
    while True:
        c.update_exchange_data()
        sleep(10)


def update_training_worker():
    while True:
        c.update_training_data()
        sleep(120)


def predictor_worker():
    while True:
        sleep(86400)
        p.model = p.train_model(15)


@app.route('/predictor/latest/<string:coin>', methods=['GET'])
def latest_prediction(coin):
    timestamp, prediction15 = p.get_latest_prediction(coin)
    predictions = {}
    predictions["15"] = str(prediction15)
    return jsonify({'timestamp': timestamp, 'predictions': predictions})


@app.route('/predictor/predict', methods=['POST'])
def predict():
    pass


@app.route('/collector/training/<string:coin>/<int:aggregation>', methods=['GET'])
def training_data(coin, aggregation):
    timestamp, prediction15 = p.get_latest_prediction(coin)
    predictions = {}
    predictions["15"] = str(prediction15)
    return jsonify({'timestamp': timestamp, 'predictions': predictions})


@app.route('/collector/data/latest/<string:coin>/<int:aggregation>', methods=['GET'])
def latest_data(coin, aggregation):
    df = c.get_training_data(coin, aggregation)
    return jsonify(df.values.tolist())


if __name__ == '__main__':
    print("Startup completed")
    update_data_thread = Thread(target=update_data_worker)
    update_training_thread = Thread(target=update_training_worker)
    predictor_thread = Thread(target=predictor_worker)

    update_data_thread.start()
    update_training_thread.start()
    predictor_thread.start()

    app.run(host='0.0.0.0', port=8989)
