import logging
import random

from trader.simulated_wallet import SimulatedWallet
from trader.trader import Trader


class SimulatedPredictor:

    def __init__(self, predictions, data, uncertainty=0):
        self.data = data
        self.predictions = predictions
        self.index = {}
        for coin in predictions.keys():
            self.index[coin] = 0
        self.uncertainty = uncertainty

    def predict(self, coin, data_line):
        prediction = self.predictions[coin][self.index[coin]] * (
                    1 + random.uniform(-self.uncertainty, self.uncertainty))
        #current_value = self.data[coin]['close_value'].iloc[self.index[coin]]
        self.index[coin] += 1
        return prediction


class TraderSimulation:

    def __init__(self, data, predictions, config, initial_balance=1000, fee=0.001, uncertainty=0):
        self.data = data
        predictor = SimulatedPredictor(predictions, data, uncertainty)
        wallet = SimulatedWallet(initial_balance, fee)
        self.trader = Trader(config, wallet, predictor)

    def run(self):
        logging.info("Simulation Started")
        for i in range(0, self.data[list(self.data.keys())[0]].shape[0]):
            for coin in self.data.keys():
                self.trader.process_data(coin, self.data[coin].iloc[i])
                if i % 10000 == 0:
                    logging.info("Simulation Progress %f%%" % ((i/self.data[coin].shape[0])*100))

        logging.info("Simulation Ended")
        return self.trader.wallet.get_net_worth()
