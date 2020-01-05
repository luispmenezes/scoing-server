import pandas as pd

from trader.evolution import Evolution
from trader.simulation import TraderSimulation
from trader.trader import TraderConfiguration

coin = 'BTCUSDT'
aggregation = 15

training_csv_path = "./training.csv"
model_base_path = "../model"
model_JSON_path = model_base_path + '/model.json'
model_weights_path = model_base_path + '/model.h5'
scaler_X_path = model_base_path + '/scaler_X.save'
scaler_Y_path = model_base_path + '/scaler_Y.save'

csv_collumns = ['open_time', 'open_value', 'high', 'low', 'close_value', 'volume', 'quote_asset_volume', 'trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ma5', 'ma10', 'prediction']


class TraderSchool:

    def __init__(self, collector, predictor):
        self.collector = collector
        self.predictor = predictor

    def run_single(self, real_predictions=False):
        data, predictions = self.setup('2019-01-01 00:00:00', True, real_predictions)

        trader_conf = TraderConfiguration(
            buy_thr=0.005,
            increase_thr=0.01,
            sell_thr=-0.005,
            profit_max=0.05,
            profit_min=-0.01,
            position_sizing=0.05)

        simulation = TraderSimulation(data, predictions, trader_conf, 1000, 0.001, 0)

        print(simulation.run())

    def run_evolution(self, init_balance, fee, uncertainty, generation_size, num_generations, mutation_rate,
                      real_predictions):
        data, predictions = self.setup('2019-01-01 00:00:00', True, real_predictions)
        evo = Evolution(data, predictions, init_balance, fee, uncertainty, generation_size, num_generations,
                        mutation_rate)
        return evo.run()

    def setup(self, start_time=None, from_file=True, run_predictor=False):
        data = {}
        predictions = {}

        coin = "BTCUSDT"
        if from_file:
            dataframe = pd.read_csv("./training-" + coin + ".csv", sep=',')
        else:
            dataframe = self.collector.get_training_data(coin, aggregation)
            dataframe.to_csv("./training-" + coin + ".csv")

        dataframe["open_time"] = pd.to_datetime(dataframe["open_time"])
        dataframe.sort_values(by=['open_time'], inplace=True, ascending=True)
        if start_time is not None:
            dataframe = dataframe[dataframe['open_time'] > '2019-01-01 00:00:00'][csv_collumns]

        if run_predictor:
            coin_data, coin_prediction = self.generate_predictions(dataframe)
        else:
            coin_data, coin_prediction = self.simulated_preditiction(dataframe)

        data[coin] = coin_data
        predictions[coin] = coin_prediction

        return data, predictions

    def simulated_preditiction(self, dataframe):
        data = dataframe[csv_collumns[:-1]]
        predictions = dataframe[csv_collumns[-1]].values
        return data, predictions

    def generate_predictions(self, dataframe):
        data = dataframe[csv_collumns[1:-1]]
        predictions = self.predictor.predict(data)
        return data, predictions
