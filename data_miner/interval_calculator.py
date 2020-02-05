from datetime import timedelta

import pandas as pd
import psycopg2
import pytz

import binance
from data_miner.rawdata_collector import RawDataCollector
from psycopg2 import pool


class IntervalCalculator:

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

    def interval_data_latest_ts(self, coin: str):
        self.cursor.execute(
            "SELECT MAX(open_time),MAX(idx) FROM cointron.binance_intervals WHERE coin=%s", (coin,))
        result = self.cursor.fetchone()
        if result is None:
            return None, None
        else:
            return result[0], result[1]

    def data_latest_ts(self, coin):
        self.cursor.execute("SELECT MAX(open_time) FROM cointron.binance_data WHERE coin=%s", (coin,))
        return self.cursor.fetchone()[0]

    def update_interval_data(self):
        for coin in binance.get_coin_list():
            latest_data_timestamp = self.data_latest_ts(coin)
            latest_interval_timestamp, latest_interval_index = self.interval_data_latest_ts(coin)
            if latest_data_timestamp is not None:
                if latest_interval_timestamp is None:
                    self.logger.info("Creating interval data from scratch (%s)..." % coin)
                    latest_interval_timestamp = binance.get_exchange_startime()
                    latest_interval_index = 0
                else:
                    latest_interval_index += 1
                    latest_interval_timestamp += timedelta(minutes=1)
                    self.logger.info("Updating Training Data for %s from %s to %s" % (
                        coin, latest_interval_timestamp, latest_data_timestamp))
                self.generate_interval_data(coin, latest_interval_timestamp.replace(tzinfo=pytz.UTC),
                                            latest_data_timestamp.replace(tzinfo=pytz.UTC), latest_interval_index)

    def generate_interval_data(self, coin: str, start_time, end_time, latest_index=0, interval_value=100000):
        self.logger.debug("Grabbing data for %s from %s to %s" % (coin, start_time, end_time))

        self.cursor.execute(
            "SELECT * FROM cointron.binance_data WHERE coin = %s AND open_time >= %s AND open_time <= %s ORDER BY open_time ASC",
            (coin, start_time, end_time))
        dataframe = pd.DataFrame(self.cursor.fetchall(),
                                 columns=RawDataCollector.data_collumns())

        self.logger.info("Computing interval data for %d records of %s" % (dataframe.shape[0], coin))

        interval_total = 0
        last_idx = 0
        interval_data = []

        for idx in range(dataframe.shape[0]):
            interval_total += dataframe.iloc[idx]["quote_asset_volume"]
            if interval_total > interval_value:
                start_time = dataframe.iloc[last_idx]["open_time"]
                end_time = dataframe.iloc[idx]["open_time"]
                open_value = dataframe.iloc[last_idx]['open_value']
                close_value = dataframe.iloc[idx]['close_value']
                high = dataframe.iloc[last_idx:idx+1]['high'].max()
                low = dataframe.iloc[last_idx:idx+1]['low'].min()
                volume = dataframe.iloc[last_idx:idx+1]['volume'].sum()
                trades = int(dataframe.iloc[last_idx:idx+1]['trades'].sum())
                taker_buy_asset_volume = dataframe.iloc[last_idx:idx+1]['taker_buy_base_asset_volume'].sum()
                interval_data.append(
                    (coin, latest_index, start_time, end_time, open_value, close_value, high, low, volume, trades,
                     taker_buy_asset_volume))
                latest_index += 1
                last_idx = idx
                interval_total = 0

                if len(interval_data) % 10000 == 0:
                    self.logger.info("Interval (%s) progress: %.2f %%" % (coin, (idx / dataframe.shape[0]) * 100))
                    insert_query = "INSERT INTO cointron.binance_intervals VALUES" + ','.join(
                        ['%s'] * len(interval_data))

                    try:
                        self.cursor.execute(insert_query, interval_data)
                    except psycopg2.ProgrammingError as e:
                        self.logger.info("Failed to insert interval data (rolling back) ", e)
                        self.conn.rollback()
                        exit(-1)
                    else:
                        self.conn.commit()
                        interval_data.clear()

        insert_query = "INSERT INTO cointron.binance_intervals VALUES" + ','.join(['%s'] * len(interval_data))

        if len(interval_data) > 0:
            try:
                self.cursor.execute(insert_query, interval_data)
            except psycopg2.ProgrammingError as e:
                self.logger.info("Failed to insert interval data (rolling back) ", e)
                self.conn.rollback()
                exit(-1)
            else:
                self.conn.commit()
