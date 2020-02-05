import logging
from datetime import datetime
from threading import Thread
from time import sleep

import flask
import pandas as pd
import pytz
from flask import Flask, jsonify, request

from aggregator import Aggregator
from collector import Collector
from interval import IntervalCalculator
from predictor import Predictor

log_format = '[%(asctime)s] [%(levelname)s] - %(name)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger("Main")

collector = Collector("menz.dynip.sapo.pt", "5433", "postgres", "postgres", "tripa123",
                      'PNGEa0YJLxVmPZssX9hDKwu3lhRQmjsyH4bpDTBg7zM2NYYCDoGAR7vtZfQorq8k',
                      'kseCG5XF731dbVAwZJHmT3g0po6NjedqyBvUohCnUcZlXhQjxk4B6q4A0jHRfW4C',
                      logging.getLogger("Collector"))
aggregator = Aggregator("menz.dynip.sapo.pt", "5433", "postgres", "postgres", "tripa123",
                        logging.getLogger("Aggregator"))
intervalCalculator = IntervalCalculator("menz.dynip.sapo.pt", "5433", "postgres", "postgres", "tripa123",
                                        logging.getLogger("Aggregator"))
predictor = Predictor(aggregator, logging.getLogger("Predictor"))

app = Flask(__name__)


def update_data_worker():
    while True:
        try:
            collector.update_exchange_data()
            sleep(10)
        except Exception as e:
            try:
                logger.info("Collector failed (now reconnecting)", e)
                collector.connect_to_db()
            except:
                sleep(120)


def update_interval_worker():
    while True:
        try:
            intervalCalculator.update_interval_data()
            sleep(10)
        except Exception as e:
            try:
                logger.info("Interval calculation failed (now reconnecting)", e)
                collector.connect_to_db()
            except:
                sleep(120)


def update_training_worker():
    while True:
        try:
            aggregator.update_training_data()
            sleep(120)
        except Exception as e:
            try:
                logger.info("Aggregator failed (now reconnecting)", e)
                aggregator.connect_to_db()
            except:
                sleep(120)


def predictor_worker():
    while True:
        sleep(86400)
        predictor.model = predictor.train_model(15)


@app.route('/predictor/latest/<string:coin>', methods=['GET'])
def latest_prediction(coin):
    timestamp, open_value, predictions = predictor.get_latest_prediction(coin)
    result = {"open_time": timestamp.isoformat("T") + "Z", "coin": coin, "close_value": open_value}
    for key in predictions.keys():
        result["pred_" + str(key)] = float(predictions[key])

    return jsonify(result)


@app.route('/predictor/predict/<int:aggregation>', methods=['POST'])
def predict(aggregation):
    data = pd.DataFrame(request.json, columns=['open_time', 'open_value', 'high', 'low', 'close_value', 'volume',
                                               'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume',
                                               'taker_buy_quote_asset_volume', 'ma5', 'ma10'])
    logger.info(data)
    predictions = predictor.predict(data, aggregation)
    for ts in predictions:
        predictions[ts] = float(predictions[ts])

    return jsonify(predictions)


@app.route('/aggregator/training/<string:coin>/<int:aggregation>', methods=['GET'])
def training_data(coin, aggregation):
    # TODO: varias coins
    df = aggregator.get_training_data(coin, aggregation, end_time=datetime.utcnow().replace(tzinfo=pytz.UTC))
    response = flask.make_response(df.to_json(orient="records"))
    response.headers['content-type'] = 'application/json'
    return response


@app.route('/aggregator/trader/<string:coin>', methods=['GET'])
def trader_training(coin):
    coins = (coin,)
    if coin == "*":
        coins = tuple(Collector.coin_list())

    start_time = datetime.fromtimestamp(int(request.headers['start_time'])).strftime("%Y/%m/%d %H:%M:%S")
    end_time = datetime.fromtimestamp(int(request.headers['end_time'])).strftime("%Y/%m/%d %H:%M:%S")

    df = aggregator.trader_training_data(coins, start_time, end_time)
    response = flask.make_response(df.to_json(orient="records"))
    response.headers['content-type'] = 'application/json'
    return response


@app.route('/collector/data/latest/<string:coin>/<int:aggregation>', methods=['GET'])
def latest_data(coin, aggregation):
    df = aggregator.get_latest_prediction_data(coin, aggregation)
    # return jsonify({'timestamp': timestamp, 'data': df.to_json(orient="records")})
    response = flask.make_response(df.to_json(orient="records"))
    response.headers['content-type'] = 'application/json'
    return response


if __name__ == '__main__':
    logging.info("Startup completed")
    update_data_thread = Thread(target=update_data_worker)
    update_interval_thread = Thread(target=update_interval_worker)
    update_training_thread = Thread(target=update_training_worker)
    predictor_thread = Thread(target=predictor_worker)

    update_data_thread.start()
    update_interval_thread.start()
    update_training_thread.start()
    # predictor_thread.start()

    app.run(host='0.0.0.0', port=8989)
