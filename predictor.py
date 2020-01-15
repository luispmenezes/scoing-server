import os
from datetime import datetime

import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

csv_collumns = ['open_value', 'high', 'low', 'close_value', 'volume', 'quote_asset_volume', 'trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ma5', 'ma10', 'prediction']
num_features = len(csv_collumns) - 1

model_output_path = "./model"


class Predictor:

    def __init__(self, aggregator, logger):
        self.aggregator = aggregator
        self.logger = logger
        self.models, self.scalers_x, self.scalers_y = self.load_model_files()

    def load_model_files(self):
        temp_models = {}
        temp_scalers_x = {}
        temp_scalers_y = {}

        for agg in self.aggregator.get_aggregations():
            base_path = model_output_path + "/" + str(agg)
            try:
                model_json = open(base_path + "/model.json", 'r')
                loaded_model_json = model_json.read()
                model_json.close()

                model = model_from_json(loaded_model_json)
                model.load_weights(base_path + "/model.h5")

                scaler_x = joblib.load(base_path + "/scaler_X.save")
                scaler_y = joblib.load(base_path + "/scaler_Y.save")

                temp_models[agg] = model
                temp_scalers_x[agg] = scaler_x
                temp_scalers_y[agg] = scaler_y
            except Exception:
                temp_models[agg], temp_scalers_x[agg], temp_scalers_y[agg] = self.train_model(agg)

        return temp_models, temp_scalers_x, temp_scalers_y

    def save_model_files(self, aggregation, model, scaler_X, scaler_Y):
        base_path = model_output_path + "/" + str(aggregation)

        if not os.path.exists(base_path):
            os.mkdir(base_path)

        model_json = model.to_json()
        with open(base_path + "/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(base_path + "/model.h5")
        joblib.dump(scaler_X, base_path + "/scaler_X.save")
        joblib.dump(scaler_Y, base_path + "/scaler_Y.save")

    def train_model(self, aggregation):
        self.logger.info("Training %d model" % (aggregation))
        dataframe_BTC = self.aggregator.get_training_data("BTCUSDT", aggregation)[csv_collumns]
        self.logger.debug("BTC dataframe recieved")
        dataframe_ETH = self.aggregator.get_training_data("ETHUSDT", aggregation)[csv_collumns]
        self.logger.debug("ETH dataframe recieved")
        dataframe_BNB = self.aggregator.get_training_data("BNBUSDT", aggregation)[csv_collumns]
        self.logger.info("Done!")

        dataframe = pd.concat([dataframe_BTC, dataframe_ETH, dataframe_BNB])

        dataset = dataframe.values
        dataset = dataset.astype('float32')

        # normalize the dataset
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_Y = MinMaxScaler(feature_range=(0, 1))
        dataset_X = scaler_X.fit_transform(dataset[:, 0:num_features])
        dataset_Y = scaler_Y.fit_transform(dataset[:, -1].reshape(-1, 1)).reshape(-1)

        # split into train and test sets
        train_size = int(len(dataset) * 0.7)
        validation_size = int(len(dataset) * 0.2)
        test_size = len(dataset) - train_size - validation_size

        train_lower_idx = 0
        train_upper_idx = train_size
        validation_lower_idx = train_upper_idx
        validation_upper_idx = validation_lower_idx + validation_size
        test_lower_idx = validation_upper_idx
        test_upper_idx = len(dataset)

        # Separate Y and Reshape X for all sets
        x_train = dataset_X[train_lower_idx: train_upper_idx:]
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        y_train = dataset_Y[train_lower_idx: train_upper_idx:]

        x_validation = dataset_X[validation_lower_idx: validation_upper_idx:]
        x_validation = np.reshape(x_validation, (x_validation.shape[0], 1, x_validation.shape[1]))
        y_validation = dataset_Y[validation_lower_idx: validation_upper_idx:]

        x_test = dataset_X[test_lower_idx: test_upper_idx:]
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
        y_test = dataset_Y[test_lower_idx: test_upper_idx:]

        # Train Model
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, num_features)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=50, verbose=2, validation_data=(x_validation, y_validation))

        self.save_model_files(aggregation, model, scaler_X, scaler_Y)

        return model, scaler_X, scaler_Y

    def get_latest_prediction(self, coin):
        predictions = {}
        timestamp = 0

        for agg in self.aggregator.get_aggregations():
            data = self.aggregator.get_latest_prediction_data(coin, agg)
            timestamp = data['open_time'].iloc[0]
            data_x = data.iloc[:, 1:].values.reshape(1, -1)
            data_x = data_x.astype('float32')
            data_x = self.scalers_x[agg].transform(data_x)
            data_x = data_x.reshape((1, 1, num_features))
            prediction = self.models[agg].predict(data_x, verbose=2)
            scaled_prediction = self.scalers_y[agg].inverse_transform(prediction)[0][0]
            predictions[agg] = scaled_prediction

        return timestamp, predictions

    def predict(self, data, agg):
        result = {}

        data_x = self.scalers_x[agg].transform(data.iloc[:, 1:].values)
        data_x = np.reshape(data_x, (data_x.shape[0], 1, data_x.shape[1]))
        raw_predictions = self.models[agg].predict(data_x, verbose=2)
        scaled_prediction = self.scalers_y[agg].inverse_transform(raw_predictions.reshape(-1, 1)).reshape(-1)

        for i in range(data.shape[0]):
            ts = data['open_time'].iloc[i]

            if isinstance(ts, np.int64):
                ts = datetime.fromtimestamp((ts / 1000.0)).strftime("%Y/%m/%d %H:%M:%S")

            result[ts] = scaled_prediction[i]

        return result
