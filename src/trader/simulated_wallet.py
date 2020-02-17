class SimulatedWallet:

    def __init__(self, initial_balance, fee):
        self.initial_balance = initial_balance
        self.fee = fee
        self.balance = initial_balance
        self.positions = {}
        self.coin_values = {}

    def get_net_worth(self):
        return self.balance + self.get_total_position_value()

    def get_total_position_value(self):
        total_position_value = 0
        for coin in list(self.coin_values):
            if coin in self.positions:
                total_position_value += sum(self.positions[coin][pos_value] for pos_value in self.positions[coin]) * \
                                        self.coin_values[coin]
        return total_position_value

    def update_coin_value(self, coin, value):
        if value >=0:
            self.coin_values[coin] = value

    def buy(self, coin, qty):
        if coin in self.coin_values:
            if qty >= 0:
                value = self.coin_values[coin]
                transaction = value * qty
                transaction_fee = transaction * self.fee
                transaction_total = transaction + transaction_fee
                if self.balance > transaction_total:
                    self.balance -= transaction_total
                    if coin not in self.positions:
                        self.positions[coin] = {}
                    if value in self.positions[coin]:
                        self.positions[coin][value] += qty
                    else:
                        self.positions[coin][value] = qty

    def sell(self, coin, position_value, qty):
        if coin in self.coin_values:
            current_value = self.coin_values[coin]
            self.balance += current_value * qty * (1 - self.fee)
            del self.positions[coin][position_value]
            if len(self.positions[coin]) == 0:
                del self.positions[coin]
