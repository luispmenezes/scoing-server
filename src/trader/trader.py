class TraderConfiguration:
    def __init__(self, buy_thr=0, increase_thr=0, sell_thr=0,
                 profit_max=0, profit_min=0, position_sizing=0):
        self.buy_thr = buy_thr
        self.increase_thr = increase_thr
        self.sell_thr = sell_thr
        self.min_profit = profit_max
        self.max_loss = profit_min
        self.position_sizing = position_sizing


class Trader:

    def __init__(self, config, wallet, predictor):
        self.config = config
        self.wallet = wallet
        self.predictor = predictor

    def process_data(self, coin, data):
        current_value = data['close_value']
        self.wallet.update_coin_value(coin, current_value)
        predicted_delta = self.predictor.predict(coin, data)

        if predicted_delta >= self.config.buy_thr:
            self.wallet.buy(coin, (self.get_buy_size()) / current_value)

        elif predicted_delta < self.config.sell_thr:
            if coin in self.wallet.positions:
                for position_value in list(self.wallet.positions[coin]):
                    if (current_value / position_value) - 1 < self.config.max_loss:
                        self.wallet.sell(coin, position_value, self.wallet.positions[coin][position_value])

        if coin in self.wallet.positions:
            for position_value in list(self.wallet.positions[coin]):
                if (current_value / position_value) - 1 > self.config.min_profit:
                    self.wallet.sell(coin, position_value, self.wallet.positions[coin][position_value])

    def get_buy_size(self):
        return self.config.position_sizing * self.wallet.balance
