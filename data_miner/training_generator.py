import concurrent.futures
import statistics
import datetime

import pandas as pd
import psycopg2
import pytz
from psycopg2 import pool

from data_miner import binance

aggregation_list = {5, 10, 100}


class TrainingGenerator:

    def __init__(self, db_host, db_port, db_name, db_user, db_password, logger):
        try:
            self.db_host = db_host
            self.db_port = db_port
            self.db_name = db_name
            self.db_user = db_user
            self.db_password = db_password
            self.pool = None
            self.conn = None
            self.cursor = None
            self.connect_to_db()
            self.logger = logger
            self.interval_data = None
        except (Exception, psycopg2.Error) as error:
            self.logger.info("Error establishing db connection", error)

    def connect_to_db(self):
        self.pool = psycopg2.pool.SimpleConnectionPool(1, 1, user=self.db_user, password=self.db_password,
                                                       host=self.db_host, port=self.db_port, database=self.db_name)
        self.conn = self.pool.getconn()
        self.cursor = self.conn.cursor()

    @staticmethod
    def get_training_features():
        return ['open_time', 'close_value', 'high_low_swing', 'price_swing', 'close_mdev_20',
                'close_mdev_100', 'close_oscillator', 'volume_mdev_20', 'volume_mdev_100',
                'volume_oscillator', 'trades_mdev_20', 'trades_mdev_100', 'trades_oscillator',
                'tbav_mdev_20', 'tbav_mdev_100', 'tbav_oscillator', 'rsi', 'cci', 'bb_band_range',
                'bb_up_mdev', 'bb_lo_mdev', 'stoch', 'aroon_up', 'aroon_down']

    def training_data_get_latest_timestamp(self, coin, aggregation):
        self.cursor.execute(
            "SELECT MAX(open_time) FROM cointron.training_data WHERE coin=%s AND aggregation=%s",
            (coin, aggregation))
        result = self.cursor.fetchone()
        if result is None:
            return None
        else:
            return result[0]

    def interval_data_latest_ts(self, coin: str):
        self.cursor.execute(
            "SELECT MAX(open_time),MAX(idx) FROM cointron.binance_intervals WHERE coin=%s", (coin,))
        result = self.cursor.fetchone()
        if result is None:
            return None, None
        else:
            return result[0], result[1]

    def update_training_data(self):
        for aggregation in aggregation_list:
            for coin in binance.get_coin_list():
                latest_data_timestamp, latest_data_index = self.interval_data_latest_ts(coin)
                latest_training_timestamp = self.training_data_get_latest_timestamp(coin, aggregation)
                if latest_data_timestamp is not None:
                    if latest_training_timestamp is None:
                        self.logger.info("Creating training data from scratch (%s,%d)..." % (coin, aggregation))
                        self.generate_training_data(coin, aggregation, 0)
                    else:
                        self.logger.info("Updating Training Data for %s,%d from %s to %s" % (
                            coin, aggregation, latest_training_timestamp,
                            latest_data_timestamp))

                        self.cursor.execute(
                            "SELECT idx FROM cointron.binance_intervals WHERE coin=%s AND open_time=%s",
                            (coin, latest_training_timestamp))

                        self.generate_training_data(coin, aggregation,
                                                    max(self.cursor.fetchone()[0] - (aggregation * 99), 0))

    def get_training_data(self, coins, aggregation, start_time=datetime.datetime(2017, 8, 17, 4, 0, tzinfo=pytz.UTC),
                          end_time=datetime.datetime(2099, 1, 1, 0, 0, tzinfo=pytz.UTC)):
        self.cursor.execute(
            "SELECT open_time,close_value,high_low_swing,price_swing,close_mdev_20,close_mdev_100,close_oscillator," +
            "volume_mdev_20,volume_mdev_100,volume_oscillator,trades_mdev_20,trades_mdev_100,trades_oscillator," +
            "tbav_mdev_20,tbav_mdev_100,tbav_oscillator,rsi,cci,bb_band_range,bb_up_mdev,bb_lo_mdev,stoch," +
            "aroon_up,aroon_down,prediction FROM cointron.training_data WHERE coin IN %s AND aggregation=%s AND open_time >= %s  " +
            "AND open_time <= %s ORDER BY open_time ASC", (coins, aggregation, start_time, end_time))

        return pd.DataFrame(self.cursor.fetchall(),
                            columns=['open_time', 'close_value', 'high_low_swing', 'price_swing', 'close_mdev_20',
                                     'close_mdev_100', 'close_oscillator', 'volume_mdev_20', 'volume_mdev_100',
                                     'volume_oscillator', 'trades_mdev_20', 'trades_mdev_100', 'trades_oscillator',
                                     'tbav_mdev_20', 'tbav_mdev_100', 'tbav_oscillator', 'rsi', 'cci', 'bb_band_range',
                                     'bb_up_mdev', 'bb_lo_mdev', 'stoch', 'aroon_up', 'aroon_down', 'prediction'])

    def get_latest_prediction_data(self, coin, aggregation):
        self.logger.debug("Getting latest prediction data for coin:%s agg:%d" % (coin, aggregation))

        self.cursor.execute(
            "SELECT idx,open_time,open_value,high,low,close_value,volume,trades,taker_buy_asset_volume " +
            "FROM cointron.binance_intervals WHERE coin=%s ORDER BY idx DESC LIMIT %s", (coin, (aggregation + 1) * 100))

        self.interval_data = pd.DataFrame(self.cursor.fetchall(),
                                          columns=["idx", "open_time", "open_value", "high", "low", "close_value",
                                                   "volume", "trades", "taker_buy_asset_volume"])

        self.interval_data = self.interval_data.sort_values(by='open_time', ascending=True)

        return pd.DataFrame([self.training_worker(self.interval_data.shape[0] - 1, aggregation, coin, False)],
                            columns=['coin', 'aggregation', 'open_time', 'close_value', 'high_low_swing', 'price_swing',
                                     'close_mdev_20', 'close_mdev_100', 'close_oscillator', 'volume_mdev_20',
                                     'volume_mdev_100', 'volume_oscillator', 'trades_mdev_20', 'trades_mdev_100',
                                     'trades_oscillator', 'tbav_mdev_20', 'tbav_mdev_100', 'tbav_oscillator', 'rsi',
                                     'cci', 'bb_band_range', 'bb_up_mdev', 'bb_lo_mdev', 'stoch', 'aroon_up',
                                     'aroon_down'])

    @staticmethod
    def get_aggregations():
        return sorted(aggregation_list)

    def trader_training_data(self, coins, start_time, end_time):
        result = None

        for agg in TrainingGenerator.get_aggregations():
            if result is None:
                self.cursor.execute(
                    "SELECT open_time,coin,close_value,prediction FROM cointron.training_data WHERE coin IN %s AND open_time >= %s AND open_time <= %s AND aggregation = %s ORDER BY open_time ASC",
                    (coins, start_time, end_time, agg))

                result = pd.DataFrame(self.cursor.fetchall(),
                                      columns=['open_time', 'coin', 'close_value', 'pred_' + str(agg)])
            else:
                self.cursor.execute(
                    "SELECT prediction FROM cointron.training_data WHERE coin IN %s AND open_time >= %s AND open_time <= %s AND aggregation = %s ORDER BY open_time ASC",
                    (coins, start_time, end_time, agg))

                result['pred_' + str(agg)] = pd.DataFrame(self.cursor.fetchall())

        return result

    @staticmethod
    def start_time():
        return binance.get_exchange_startime()

    def generate_training_data(self, coin: str, aggregation: int, start_idx):

        self.logger.debug("Generating training data (%s,%d) from idx %s" % (coin, aggregation, start_idx))

        self.cursor.execute(
            "SELECT idx,open_time,open_value,high,low,close_value,volume,trades,taker_buy_asset_volume " +
            "FROM cointron.binance_intervals WHERE coin=%s AND idx >= %s ORDER BY idx ASC",
            (coin, start_idx))

        self.interval_data = pd.DataFrame(self.cursor.fetchall(),
                                          columns=["idx", "open_time", "open_value", "high", "low", "close_value",
                                                   "volume", "trades", "taker_buy_asset_volume"])
        training_data = []
        num_workers = 1

        for i in range(100 * aggregation, self.interval_data.shape[0] - aggregation, num_workers):
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                workers = []

                for thread_idx in range(i, min(i + num_workers, self.interval_data.shape[0] - aggregation)):
                    workers.append(executor.submit(self.training_worker, i, aggregation, coin))

                for w in concurrent.futures.as_completed(workers):
                    if w.result() is not None:
                        training_data.append(w.result())

                if (len(training_data) % 1000 == 0 or i + num_workers > self.interval_data.shape[0] - aggregation) \
                        and len(training_data) > 0:
                    self.logger.info(
                        "Training Data (%s) progress: %.2f %%" % (coin, (i / self.interval_data.shape[0]) * 100))
                    insert_query = "INSERT INTO cointron.training_data VALUES" + ','.join(['%s'] * len(training_data)) \
                                   + " ON CONFLICT DO NOTHING"

                    if len(training_data) > 0:
                        try:
                            self.cursor.execute(insert_query, training_data)
                        except psycopg2.ProgrammingError as e:
                            self.logger.info("Failed to insert interval data (rolling back) ", e)
                            self.conn.rollback()
                            exit(-1)
                        else:
                            self.conn.commit()
                            training_data.clear()

    def training_worker(self, i: int, aggregation: int, coin: str, with_prediction=True):
        features = self.compute_features(i, aggregation)
        current_interval = self.interval_data.iloc[i]
        training_entry = (coin, aggregation, current_interval['open_time'], current_interval['close_value']) + features
        if with_prediction:
            prediction = float(self.interval_data.iloc[i + aggregation]['close_value'])
            training_entry += ((prediction / current_interval['close_value']) - 1.0,)
        return training_entry

    def compute_features(self, data_idx: int, aggregation: int):
        # TODO: fix and recalculate training data
        ma20 = self.compute_moving_average(data_idx, aggregation, 100)
        ma100 = self.compute_moving_average(data_idx, aggregation, 20)

        agg_data = self.interval_data.iloc[data_idx - aggregation:data_idx]

        high = agg_data['high'].max()
        low = agg_data['low'].min()
        close = agg_data['close_value'].iloc[-1]
        volume = agg_data['volume'].sum()
        trades = int(agg_data['trades'].sum())
        tbav = agg_data['taker_buy_asset_volume'].mean()

        # Price
        high_low_swing = (high - low) / low
        price_swing = (close - agg_data['open_value'].iloc[0]) / agg_data['open_value'].iloc[0]
        close_mdev_20 = abs(close - ma20[0]) / ma20[0]
        close_mdev_100 = abs(close - ma100[0]) / ma100[0]
        close_oscillator = ma100[0] / ma20[0]
        # Volume
        volume_mdev_20 = abs(volume - ma20[1]) / ma20[1]
        volume_mdev_100 = abs(volume - ma100[1]) / ma100[1]
        volume_oscillator = ma100[1] / ma20[1]
        # Trades
        trades_mdev_20 = abs(trades - ma20[2]) / ma20[2]
        trades_mdev_100 = abs(trades - ma100[2]) / ma100[2]
        trades_oscillator = ma100[2] / ma20[2]
        # TBAV
        tbav_mdev_20 = abs(tbav - ma20[3]) / ma20[3]
        tbav_mdev_100 = abs(tbav - ma100[3]) / ma100[3]
        tbav_oscillator = ma100[3] / ma20[3]

        rsi = self.__feature_RSI(data_idx, aggregation)
        cci = self.__feature_CCI(data_idx, aggregation)
        abs_mid_band, abs_up_band, abs_lo_band = self.__feature_bollinger_bands(data_idx, aggregation)
        bb_up_mdev = abs(abs_up_band - close) / close
        bb_lo_mdev = abs(abs_lo_band - close) / close

        bb_band_range = abs(abs_lo_band - abs_up_band) / abs_mid_band

        stoch = self.__feature_STOCH(data_idx, aggregation)
        aroon_up, aroon_down = self.__feature_AROON(data_idx, aggregation)

        return high_low_swing, price_swing, close_mdev_20, close_mdev_100, close_oscillator, volume_mdev_20, volume_mdev_100, \
               volume_oscillator, trades_mdev_20, trades_mdev_100, trades_oscillator, tbav_mdev_20, tbav_mdev_100, tbav_oscillator, \
               rsi, cci, bb_band_range, bb_up_mdev, bb_lo_mdev, stoch, aroon_up, aroon_down

    def __feature_RSI(self, index: int, aggregation: int):
        up_list = []
        dn_list = []

        for i in range(index - 13 * aggregation, index, aggregation):
            close = self.interval_data.iloc[i - aggregation:i]['close_value'].mean()
            last_close = self.interval_data.iloc[i - 2 * aggregation:i - aggregation]['close_value'].mean()

            if close > last_close:
                up_list.append(close - last_close)
                dn_list.append(0)
            else:
                up_list.append(0)
                dn_list.append(last_close - close)
        up_avg = sum(up_list) / len(up_list)
        dn_avg = sum(dn_list) / len(dn_list)

        return 100 * (up_avg / (up_avg + dn_avg))

    def __feature_bollinger_bands(self, index: int, aggregation: int, intervals=14):
        tp_list = []

        for i in range(index - intervals * aggregation, index, aggregation):
            close = self.interval_data.iloc[i - aggregation:i]['close_value'].mean()
            high = self.interval_data.iloc[i - aggregation:i]['high'].max()
            low = self.interval_data.iloc[i - aggregation:i]['low'].min()

            tp_list.append((close + high + low) / 3)

        stdev = statistics.stdev(tp_list)
        midBand = sum(tp_list) / len(tp_list)
        upBand = midBand + (2 * stdev)
        loBand = midBand - (2 * stdev)

        return midBand, upBand, loBand

    def __feature_CCI(self, index: int, aggregation: int, intervals=20):
        tp_list = []

        for i in range(index - intervals * aggregation, index - aggregation, aggregation):
            close = self.interval_data.iloc[i - aggregation:i]['close_value'].mean()
            high = self.interval_data.iloc[i - aggregation:i]['high'].max()
            low = self.interval_data.iloc[i - aggregation:i]['low'].min()

            tp_list.append((close + high + low) / 3)

        current_tp = tp_list.pop(-1)
        avg_tp = sum(tp_list) / len(tp_list)
        md_tp = statistics.stdev(tp_list)

        return (current_tp - avg_tp) / (0.015 * md_tp)

    def __feature_STOCH(self, index: int, aggregation: int, intervals=20):
        n_periods = self.interval_data.iloc[index - intervals * aggregation: index - aggregation]
        lowestLow = n_periods['close_value'].min()
        higestHigh = n_periods['close_value'].max()

        return (self.interval_data.iloc[index - aggregation:index]['close_value'].mean() - lowestLow) / (
                higestHigh - lowestLow)

    def __feature_AROON(self, index: int, aggregation: int, intervals=20):
        max_value = -9999
        min_value = 9999
        max_dist = 0
        min_dist = 0

        for i in range(index - intervals * aggregation, index - aggregation, aggregation):
            close = self.interval_data.iloc[i - aggregation:i]['close_value'].mean()
            if close > max_value:
                max_value = close
                max_dist = 0
            else:
                max_dist += 1

            if close < min_value:
                min_value = close
                min_dist = 0
            else:
                min_dist += 1

        aroon_up = 100 * ((intervals - max_dist) / intervals)
        aroon_down = 100 * ((intervals - min_dist) / intervals)

        return aroon_up, aroon_down

    def compute_moving_average(self, index: int, aggregation: int, size: int):
        ma_data = self.interval_data.iloc[index - (size * aggregation):index]
        ma_price = ma_data['close_value'].sum() / size
        ma_volume = ma_data['volume'].sum() / size
        ma_trades = ma_data['trades'].sum() / size
        ma_tbav = ma_data['taker_buy_asset_volume'].sum() / size
        return ma_price, ma_volume, ma_trades, ma_tbav
