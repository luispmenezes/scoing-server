import concurrent.futures
from datetime import timedelta

import pandas as pd
import psycopg2
import pytz
from psycopg2 import pool

import binance
from collector import Collector

aggregation_list = {15, 60, 1440}


class Aggregator:

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
        except (Exception, psycopg2.Error) as error:
            self.logger.info("Error establishing db connection", error)

    def connect_to_db(self):
        self.pool = psycopg2.pool.SimpleConnectionPool(1, 1, user=self.db_user, password=self.db_password,
                                                       host=self.db_host, port=self.db_port, database=self.db_name)
        self.conn = self.pool.getconn()
        self.cursor = self.conn.cursor()

    def training_data_get_latest_timestamp(self, coin, aggregation):
        self.cursor.execute(
            "SELECT MAX(open_time) FROM cointron.training_data WHERE coin=%s AND aggregation=%s", (coin, aggregation))
        return self.cursor.fetchone()[0]

    def data_latest_ts(self, coin):
        timestamp = None
        while timestamp is None:
            try:
                self.cursor.execute("SELECT MAX(open_time) FROM cointron.binance_data WHERE coin=%s", (coin,))
            except Exception as e:
                self.logger.info("Failed getting latest training timestamp")
                self.logger.debug(e)
            else:
                result = self.cursor.fetchone()
                if result is not None:
                    timestamp = result[0]
        return timestamp

    def create_training_data_in_memory(self, coin, aggregation, start_time, end_time):
        self.cursor.execute(
            "SELECT * FROM cointron.binance_data WHERE coin=%s AND open_time >= %s  AND open_time <= %s ORDER BY open_time ASC",
            (coin, start_time, end_time))

        df = pd.DataFrame(self.cursor.fetchall(), columns=Collector.data_collumns())
        num_workers = 5
        training_data = []

        for idx in range(0, df.shape[0], num_workers):
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                workers = []
                for thread_idx in range(idx, min(idx + num_workers, df.shape[0])):
                    if thread_idx + aggregation < df.shape[0] and thread_idx >= 10 * aggregation:
                        workers.append(executor.submit(self.training_worker, coin, aggregation, df, thread_idx, True))

                for w in concurrent.futures.as_completed(workers):
                    if w.result() is not None:
                        training_data.append(w.result())

                if (len(training_data) > 5000 or idx + num_workers > df.shape[0]) and len(training_data) > 0:
                    self.logger.info(
                        "Training data(%s,%d) progress %2f %%" % (coin, aggregation, (idx / df.shape[0]) * 100))

                    insert_query = "INSERT INTO cointron.training_data VALUES" + ','.join(
                        ['%s'] * len(training_data)) + " ON CONFLICT DO NOTHING"
                    try:
                        self.cursor.execute(insert_query, training_data)
                    except psycopg2.ProgrammingError as e:
                        self.logger.info("Failed to insert training data (rolling back) ", e)
                        self.conn.rollback()
                        return
                    else:
                        self.conn.commit()
                        training_data.clear()

    @staticmethod
    def training_worker(coin, aggregation, df, idx, training):
        agg_range = df.iloc[idx - aggregation:idx]

        open_value = agg_range['open_value'].iloc[0]
        high = agg_range['high'].max()
        low = agg_range['low'].min()
        close_value = agg_range['close_value'].iloc[-1]
        volume = agg_range['volume'].sum()
        quote_asset_volume = agg_range['quote_asset_volume'].mean()
        trades = int(agg_range['trades'].sum())
        taker_buy_base_asset_volume = agg_range['taker_buy_base_asset_volume'].mean()
        taker_buy_quote_asset_volume = agg_range['taker_buy_quote_asset_volume'].mean()
        ma5 = df.iloc[idx - (5 * aggregation):idx]['close_value'].mean()
        ma10 = df.iloc[idx - (10 * aggregation):idx]['close_value'].mean()

        if training:
            prediction = df['open_value'].iloc[idx + aggregation]
            prediction_delta = (prediction / close_value) - 1.0

            return (coin, aggregation, df['open_time'].iloc[idx], open_value, high, low, close_value, volume,
                    quote_asset_volume, trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ma5, ma10,
                    prediction_delta)
        else:
            return (df['open_time'].iloc[idx], open_value, high, low, close_value, volume, quote_asset_volume, trades,
                    taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ma5, ma10)

    def update_training_data(self):
        for aggregation in aggregation_list:
            for coin in binance.get_coin_list():
                latest_data_timestamp = self.data_latest_ts(coin)
                latest_training_timestamp = self.training_data_get_latest_timestamp(coin, aggregation)
                if latest_data_timestamp is not None:
                    if latest_training_timestamp is None:
                        self.logger.info("Creating training data from scratch (%s,%d)..." % (coin, aggregation))
                        latest_training_timestamp = binance.get_exchange_startime()
                    else:
                        self.logger.info("Updating Training Data for %s,%d from %s to %s" % (
                            coin, aggregation, latest_training_timestamp,
                            latest_data_timestamp))
                        latest_training_timestamp -= timedelta(minutes=aggregation * 10)
                    self.create_training_data_in_memory(coin, aggregation,
                                                        latest_training_timestamp.replace(tzinfo=pytz.UTC),
                                                        latest_data_timestamp.replace(tzinfo=pytz.UTC))

    def get_training_data(self, coin, aggregation, start_time, end_time):
        self.cursor.execute(
            "SELECT open_time,open_value,high,low,close_value,volume,quote_asset_volume,trades," +
            "taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ma5,ma10,prediction FROM cointron.training_data " +
            "WHERE coin=%s AND aggregation=%s AND open_time >= %s  AND open_time <= %s ORDER BY open_time ASC",
            (coin, aggregation, start_time, end_time))

        return pd.DataFrame(self.cursor.fetchall(),
                            columns=['open_time', 'open_value', 'high', 'low', 'close_value',
                                     'volume', 'quote_asset_volume', 'trades',
                                     'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ma5', 'ma10',
                                     'prediction'])

    def get_latest_prediction_data(self, coin, aggregation):
        timestamp = self.data_latest_ts(coin)

        self.cursor.execute(
            "SELECT * FROM cointron.binance_data WHERE coin=%s AND open_time >= %s AND open_time <= %s ORDER BY open_time ASC",
            (coin, timestamp - timedelta(minutes=10 * aggregation), timestamp))

        df = pd.DataFrame(self.cursor.fetchall(), columns=Collector.data_collumns())

        return pd.DataFrame([self.training_worker(coin, aggregation, df, df.shape[0] - 1, False)],
                            columns=['open_time', 'open_value', 'high', 'low', 'close_value', 'volume',
                                     'quote_asset_volume',
                                     'trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ma5', 'ma10'])

    @staticmethod
    def get_aggregations():
        return aggregation_list

    def trader_training_data(self, coins, start_time, end_time):
        result = None

        for agg in Aggregator.get_aggregations():
            if result is None:
                self.cursor.execute(
                    "SELECT to_char(current_timestamp, 'FXYYYY-MM-DD\"T\"HH:MI:SS\"Z00:00\"'),coin,open_value,prediction FROM cointron.training_data WHERE coin IN %s AND open_time >= %s AND open_time <= %s AND aggregation = %s ORDER BY open_time ASC",
                    (coins, start_time, end_time, agg))

                result = pd.DataFrame(self.cursor.fetchall(),
                                      columns=['open_time', 'coin', 'open_value', 'pred_' + str(agg)])
            else:
                self.cursor.execute(
                    "SELECT prediction FROM cointron.training_data WHERE coin IN %s AND open_time >= %s AND open_time <= %s AND aggregation = %s ORDER BY open_time ASC",
                    (coins, start_time, end_time, agg))

                result['pred_' + str(agg)] = pd.DataFrame(self.cursor.fetchall())

        return result

    @staticmethod
    def start_time():
        return binance.get_exchange_startime()
