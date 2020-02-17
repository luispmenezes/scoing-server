import unittest

from trader.simulated_wallet import SimulatedWallet

class TestWallet(unittest.TestCase):

    def test_buy(self):
        wallet = SimulatedWallet(100, 0)
        wallet.update_coin_value('BTCUST', 10)
        wallet.buy('BTCUST', 2)
        self.assertEqual(wallet.balance, 80)
        self.assertEqual(wallet.positions, {'BTCUST': {10: 2}})

    def test_sell(self):
        wallet = SimulatedWallet(100, 0)

        wallet.update_coin_value('BTCUST', 10)
        wallet.buy('BTCUST', 1)
        wallet.buy('BTCUST', 2)

        self.assertEqual(wallet.positions, {'BTCUST': {10: 3}})

        wallet.update_coin_value('BTCUST', 20)
        wallet.sell('BTCUST', 10, 3)
        self.assertEqual(wallet.balance, 130)
        self.assertEqual(len(wallet.positions), 0)

    def test_fee(self):
        wallet = SimulatedWallet(100, 0.1)
        wallet.update_coin_value('BTCUST', 10)
        wallet.buy('BTCUST', 1)
        wallet.sell('BTCUST', 10, 1)
        self.assertEqual(wallet.balance, 98)

    def test_total_pos_value(self):
        wallet = SimulatedWallet(100, 0)
        wallet.update_coin_value('BTCUST', 10)
        wallet.update_coin_value('ETHUST', 5)
        wallet.buy('BTCUST', 1)
        wallet.buy('ETHUST', 10)

        self.assertEqual(wallet.positions, {'BTCUST': {10: 1}, 'ETHUST': {5: 10}})
        self.assertEqual(wallet.get_total_position_value(), 60)

        wallet.update_coin_value('BTCUST', 20)
        wallet.update_coin_value('ETHUST', 10)
        self.assertEqual(wallet.get_total_position_value(), 120)

    def test_net_worth(self):
        wallet = SimulatedWallet(100, 0)
        wallet.update_coin_value('BTCUST', 10)
        wallet.update_coin_value('ETHUST', 5)
        wallet.buy('BTCUST', 2)
        wallet.buy('ETHUST', 10)
        wallet.update_coin_value('BTCUST', 20)
        wallet.update_coin_value('ETHUST', 10)
        self.assertEqual(wallet.get_net_worth(), 170)


if __name__ == '__main__':
    unittest.main()
