import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

csv_collumns = ['close_value', 'high_low_swing', 'price_swing', 'close_mdev_20', 'close_mdev_100',
                'close_oscillator', 'volume_mdev_20', 'volume_mdev_100', 'volume_oscillator', 'trades_mdev_20',
                'trades_mdev_100', 'trades_oscillator', 'tbav_mdev_20', 'tbav_mdev_100', 'tbav_oscillator', 'rsi',
                'cci', 'bb_band_range', 'bb_up_mdev', 'bb_lo_mdev', 'stoch', 'aroon_up', 'aroon_down', 'prediction']
coin_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "XRPUSDT"]
num_features = len(csv_collumns) - 1

data_path = "./data"
model_output_path = data_path + "/model"
training_output_path = data_path + "/training"


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

                if not os.path.exists(data_path):
                    os.mkdir(data_path)

                dataframe = self.get_training_from_file(coin_list, agg)
                temp_models[agg], temp_scalers_x[agg], temp_scalers_y[agg] = self.train_model(agg, dataframe)

        return temp_models, temp_scalers_x, temp_scalers_y

    def save_model_files(self, aggregation, model, scaler_X, scaler_Y):
        path = model_output_path + "/" + str(aggregation)

        if not os.path.exists(path):
            os.mkdir(path)

        model_json = model.to_json()
        with open(path + "/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(path + "/model.h5")
        joblib.dump(scaler_X, path + "/scaler_X.save")
        joblib.dump(scaler_Y, path + "/scaler_Y.save")

    def get_training_from_server(self, coins, aggregation, save=False):
        self.logger.info("Getting data for Coins:%s Aggregation:%d data from server" % (coins, aggregation))
        start_time = self.aggregator.start_time()
        end_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
        dataframe = self.aggregator.get_training_data(tuple(coins), aggregation, start_time, end_time)
        dataframe = dataframe.drop(columns=['open_time'])

        if save:
            if not os.path.exists(training_output_path):
                os.mkdir(training_output_path)

            filename = "training_%d_%s.csv" % (aggregation, '_'.join(coins))
            dataframe.to_csv(training_output_path + "/" + filename)

        return dataframe

    def get_training_from_file(self, coins, aggregation):
        filename = "training_%d_%s.csv" % (aggregation, '_'.join(coins))
        if os.path.isfile(training_output_path + "/" + filename):
            return pd.read_csv(training_output_path + "/" + filename)
        else:
            return self.get_training_from_server(coins, aggregation, True)

    def train_model(self, aggregation, dataframe, epochs=50, batch_size=1000):
        self.logger.info("Training model for %d" % aggregation)

        self.logger.debug("Shuffling dataset...")
        dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

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
        model.add(LSTM(200, input_shape=(1, num_features), activation='relu', return_sequences=True))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(25, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam',
                      metrics=['mae'])
        model.fit(x_train, y_train, epochs=epochs, verbose=2, validation_data=(x_validation, y_validation),
                  batch_size=batch_size)

        loss = model.evaluate(x_test, y_test, verbose=2)
        self.logger.info("Testing loss: %f  MeanAbsError: %f" % (loss[0], loss[1]))

        self.save_model_files(aggregation, model, scaler_X, scaler_Y)

        return model, scaler_X, scaler_Y

    def get_latest_prediction(self, coin):
        predictions = {}
        timestamp = 0
        close_value = 0.0

        for agg in self.aggregator.get_aggregations():
            data = self.aggregator.get_latest_prediction_data(coin, agg)
            timestamp = data['open_time'].iloc[0]
            close_value = data['close_value'].iloc[0]
            data_x = data.iloc[:, 3:].values.reshape(1, -1)
            data_x = data_x.astype('float32')
            data_x = self.scalers_x[agg].transform(data_x)
            data_x = data_x.reshape((1, 1, num_features))
            prediction = self.models[agg].predict(data_x, verbose=2)
            scaled_prediction = self.scalers_y[agg].inverse_transform(prediction)[0][0]
            predictions[agg] = scaled_prediction

        return timestamp, close_value, predictions

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
