import logging
from datetime import datetime
from threading import Thread
from time import sleep

import pandas as pd
import pytz
from flask import Flask, jsonify, request

from aggregator import Aggregator
from collector import Collector
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
predictor = Predictor(aggregator, logging.getLogger("Predictor"))

app = Flask(__name__)


def update_data_worker():
    while True:
        try:
            collector.update_exchange_data()
            sleep(10)
        except Exception as e:
            logger.info("Collector failed (now reconnecting)", e)
            collector.connect_to_db()
            sleep(60)


def update_training_worker():
    while True:
        try:
            aggregator.update_training_data()
            sleep(120)
        except Exception as e:
            logger.info("Collector failed (now reconnecting)", e)
            collector.connect_to_db()
            sleep(60)


def predictor_worker():
    while True:
        sleep(86400)
        predictor.model = predictor.train_model(15)


@app.route('/predictor/latest/<string:coin>', methods=['GET'])
def latest_prediction(coin):
    timestamp, predictions = predictor.get_latest_prediction(coin)
    for key in predictions.keys():
        predictions[key] = str(predictions[key])

    return jsonify({'timestamp': timestamp, 'predictions': predictions})


@app.route('/predictor/predict', methods=['POST'])
def predict():
    data = pd.DataFrame(request.json, columns=['open_time', 'open_value', 'high', 'low', 'close_value', 'volume',
                                               'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume',
                                               'taker_buy_quote_asset_volume', 'ma5', 'ma10'])
    logger.info(data)
    predictions = predictor.predict(data)
    str_list = []
    for pred_entry in predictions:
        str_entry = {"prediction": {}}
        for agg in pred_entry["prediction"].keys():
            str_entry["prediction"][agg] = str(pred_entry["prediction"][agg])

        if isinstance(pred_entry["timestamp"], str):
            str_entry["timestamp"] = pred_entry["timestamp"]
        else:
            str_entry["timestamp"] = datetime.fromtimestamp((pred_entry["timestamp"] / 1000.0)).strftime(
                "%Y/%m/%d %H:%M:%S")
        str_list.append(str_entry)

    return jsonify({'predictions': str_list})


@app.route('/collector/training/<string:coin>/<int:aggregation>', methods=['GET'])
def training_data(coin, aggregation):
    return aggregator.get_training_data(coin, aggregation, end_time=datetime.utcnow().replace(tzinfo=pytz.UTC))


@app.route('/collector/data/latest/<string:coin>/<int:aggregation>', methods=['GET'])
def latest_data(coin, aggregation):
    df = aggregator.get_latest_prediction_data(coin, aggregation)
    # return jsonify({'timestamp': timestamp, 'data': df.to_json(orient="records")})
    return df.to_json(orient="records")


if __name__ == '__main__':
    logging.info("Startup completed")
    update_data_thread = Thread(target=update_data_worker)
    update_training_thread = Thread(target=update_training_worker)
    predictor_thread = Thread(target=predictor_worker)

    update_data_thread.start()
    update_training_thread.start()
    # predictor_thread.start()

    app.run(host='0.0.0.0', port=8989)
