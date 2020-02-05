from datetime import datetime, timedelta

import psycopg2
import pytz
from psycopg2 import pool

import binance

aggregations = {15, 60, 1440}


class RawDataCollector:

    def __init__(self, db_host, db_port, db_name, db_user, db_password, binance_api_key, binance_api_secret, logger):
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
            self.binance_client = binance.Client(binance_api_key, binance_api_secret)
            self.logger = logger
        except (Exception, psycopg2.Error) as error:
            self.logger.info("Error establishing db connection", error)

    def connect_to_db(self):
        self.pool = psycopg2.pool.SimpleConnectionPool(1, 1, user=self.db_user, password=self.db_password, host=self.db_host,port=self.db_port,database=self.db_name)
        self.conn = self.pool.getconn()
        self.cursor = self.conn.cursor()

    @staticmethod
    def data_collumns():
        return ['coin', 'open_time', 'open_value', 'high', 'low', 'close_value', 'volume', 'quote_asset_volume',
                'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

    def data_latest_ts(self, coin):
        self.cursor.execute("SELECT MAX(open_time) FROM cointron.binance_data WHERE coin=%s", (coin,))
        return self.cursor.fetchone()[0]

    def grab_exchange_data(self, coin, start_time, end_time):
        loop_start_time = start_time

        while loop_start_time <= end_time:
            loop_end_time = loop_start_time + timedelta(days=5) - timedelta(minutes=1)
            if loop_end_time > end_time:
                loop_end_time = end_time

            self.logger.info('Getting binance data for %s to %s' % (loop_start_time, loop_end_time))

            records_to_insert = []
            for data in self.binance_client.klines(coin, "1m", loop_start_time, loop_end_time):
                records_to_insert.append(
                    (coin, datetime.fromtimestamp((data[0] / 1000.0), tz=pytz.UTC), float(data[1]), float(data[2]),
                     float(data[3]), float(data[4]), float(data[5]), float(data[7]), int(data[8]),
                     float(data[9]), float(data[10])))

            if len(records_to_insert) > 0:
                insert_query = "INSERT INTO cointron.binance_data VALUES" + ','.join(
                    ['%s'] * len(records_to_insert)) + " ON CONFLICT DO NOTHING"
                try:
                    self.cursor.execute(insert_query, records_to_insert)
                except Exception as e:
                    self.logger.info("Exchange data insert failed ", e)
                    self.conn.rollback()
                else:
                    self.conn.commit()

            loop_start_time += timedelta(days=5)

    def update_exchange_data(self):
        for coin in binance.get_coin_list():
            latest_timestamp = self.data_latest_ts(coin)
            if latest_timestamp is None:
                latest_timestamp = binance.get_exchange_startime()
            else:
                latest_timestamp = latest_timestamp.replace(tzinfo=pytz.UTC) - timedelta(minutes=1)

            self.logger.info("Updating binance data from %s" % latest_timestamp)
            self.grab_exchange_data(coin, latest_timestamp,
                                    datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(minutes=1))

    @staticmethod
    def coin_list():
        return binance.get_coin_list()
