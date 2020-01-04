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
        self.model, self.scaler_X, self.scaler_Y = self.load_model_files()

    def load_model_files(self):
        try:
            model_json = open(model_output_path + "/model.json", 'r')
            loaded_model_json = model_json.read()
            model_json.close()

            model = model_from_json(loaded_model_json)
            model.load_weights(model_output_path + "/model.h5")

            scaler_X = joblib.load(model_output_path + "/scaler_X.save")
            scaler_Y = joblib.load(model_output_path + "/scaler_Y.save")

            return model, scaler_X, scaler_Y
        except:
            return self.train_model(15)

    def save_model_files(self, model, scaler_X, scaler_Y):
        if not os.path.exists(model_output_path):
            os.mkdir(model_output_path)

        model_json = model.to_json()
        with open(model_output_path + "/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_output_path + "/model.h5")
        joblib.dump(scaler_X, model_output_path + "/scaler_X.save")
        joblib.dump(scaler_Y, model_output_path + "/scaler_Y.save")

    def train_model(self, aggregation):
        print("Getting dataframes...")
        dataframe_BTC = self.collector.get_training_data("BTCUSDT", aggregation)[csv_collumns]
        dataframe_ETH = self.collector.get_training_data("ETHUSDT", aggregation)[csv_collumns]
        dataframe_BNB = self.collector.get_training_data("BNBUSDT", aggregation)[csv_collumns]
        print("Done!")

        dataframe = pd.concat([dataframe_BTC, dataframe_ETH, dataframe_BNB])
        dataframe = dataframe.sample(frac=1, axis=1).reset_index(drop=True)

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

        self.save_model_files(model, scaler_X, scaler_Y)

        return model, scaler_X, scaler_Y

    def get_latest_prediction(self, coin):
        timestamp, data = self.collector.get_latest_prediction_data(coin, 15)
        X = data.values.reshape(1, -1)
        X = X.astype('float32')
        X = self.scaler_X.transform(X)
        X = X.reshape((1, 1, num_features))
        prediction = self.model.predict(X, verbose=2)
        scaled_prediction = self.scaler_Y.inverse_transform(prediction)[0][0]
        return timestamp, scaled_prediction
