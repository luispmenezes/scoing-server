import logging
import os

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

    def __init__(self, collector):
        self.collector = collector
        self.models, self.scalers_X, self.scalers_Y = self.load_model_files()

    def load_model_files(self):
        temp_models = {}
        temp_scalers_x = {}
        temp_scalers_y = {}

        for agg in self.collector.get_aggregations():
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
        logging.info("Training %d model" % (aggregation))
        dataframe_BTC = self.collector.get_training_data("BTCUSDT", aggregation)[csv_collumns]
        logging.debug("BTC dataframe recieved")
        dataframe_ETH = self.collector.get_training_data("ETHUSDT", aggregation)[csv_collumns]
        logging.debug("ETH dataframe recieved")
        dataframe_BNB = self.collector.get_training_data("BNBUSDT", aggregation)[csv_collumns]
        logging.info("Done!")

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
        model.fit(x_train, y_train, epochs=25, verbose=2, validation_data=(x_validation, y_validation))

        self.save_model_files(aggregation, model, scaler_X, scaler_Y)

        return model, scaler_X, scaler_Y

    def get_latest_prediction(self, coin):
        predictions = {}

        for agg in self.collector.get_aggregations():
            data = self.collector.get_latest_prediction_data(coin, agg)
            timestamp = data['open_time'].iloc[0]
            X = data.iloc[:, 1:].values.reshape(1, -1)
            X = X.astype('float32')
            X = self.scaler_X[agg].transform(X)
            X = X.reshape((1, 1, num_features))
            prediction = self.model[agg].predict(X, verbose=2)
            scaled_prediction = self.scaler_Y[agg].inverse_transform(prediction)[0][0]
            predictions[agg] = scaled_prediction

        return timestamp, predictions

    def predict(self, data):
        predictions = {}

        for agg in self.collector.get_aggregations():
            timestamp = data['open_time'].iloc[0]
            X = self.scaler_X.transform(data.iloc[:, 1:].values)
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
            raw_predictions = self.model.predict(X, verbose=2)
            scaled_prediction = data.scaler_Y.inverse_transform(raw_predictions.reshape(-1, 1)).reshape(-1)
            predictions[agg] = scaled_prediction

        return timestamp, predictions
