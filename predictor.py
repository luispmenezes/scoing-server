from datetime import timedelta

import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


class Predictor:
    csv_collumns = ['open_time', 'open_value', 'high', 'low', 'close_value', 'volume', 'quote_asset_volume', 'trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'prediction']
    num_features = len(csv_collumns) - 1

    def __init__(self, collector):
        self.collector = collector
        self.model = self.train_model("BTCUSDT", 15)

    def train_model(self, coin, aggregation):
        dataframe = self.collector.get_training_data(coin, aggregation)
        dataset = dataframe.values
        dataset = dataset.astype('float32')

        # normalize the dataset
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_Y = MinMaxScaler(feature_range=(0, 1))
        dataset_X = scaler_X.fit_transform(dataset[:, 0:self.num_features])
        dataset_Y = scaler_Y.fit_transform(dataset[:, -1].reshape(-1, 1)).reshape(-1)

        # split into train and test sets
        train_size = int(len(dataset) * 0.6)
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
        model.add(LSTM(4, input_shape=(1, self.num_features)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=100, batch_size=24, verbose=2, validation_data=(x_validation, y_validation))

        return model

    def get_latest_prediction(self, coin):
        latest_timestamp = self.collector.training_data_get_latest_timestamp(coin)
        data = self.collector.grab_training_data(coin, 15, latest_timestamp - timedelta(15), latest_timestamp)
        X = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        return self.model.predict(X, verbose=2)
