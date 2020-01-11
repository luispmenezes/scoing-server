import concurrent.futures
import logging
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import pytz
from psycopg2 import pool

import binance

aggregations = {15,60, 1440}


class Collector:

    def __init__(self, db_host, db_port, db_name, db_user, db_password, binance_api_key, binance_api_secret):
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(1, 20, user=db_user,
                                                             password=db_password,
                                                             host=db_host,
                                                             port=db_port,
                                                             database=db_name)
            self.data_conn = self.pool.getconn()
            self.data_cursor = self.data_conn.cursor()
            self.training_conn = self.pool.getconn()
            self.training_cursor = self.training_conn.cursor()
            self.binance_client = binance.Client(binance_api_key, binance_api_secret)
        except (Exception, psycopg2.Error) as error:
            logging.info("Error establishing db connection", error)

    def get_binance_data_collumns(self):
        return ['coin', 'open_time', 'open_value', 'high', 'low', 'close_value', 'volume', 'quote_asset_volume',
                'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

    def exchange_data_get_latest_timestamp(self, coin):
        timestamp = None
        while timestamp is None:
            try:
                self.data_cursor.execute("SELECT MAX(open_time) FROM cointron.binance_data WHERE coin=%s", (coin,))
            except Exception:
                logging.info("Failed getting latest training timestamp")
            else:
                result = self.data_cursor.fetchone()
                if result is not None:
                    timestamp = self.data_cursor.fetchone()[0]
        return timestamp

    def grab_exchange_data(self, coin, start_time=binance.get_exchange_startime(),
                           end_time=datetime.utcnow().replace(tzinfo=pytz.UTC)):
        loop_start_time = start_time

        while loop_start_time <= end_time:
            loop_end_time = loop_start_time + timedelta(days=5) - timedelta(minutes=1)
            if loop_end_time > end_time:
                loop_end_time = end_time

            logging.info('Getting binance data for %s to %s' % (loop_start_time, loop_end_time))

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
                    self.data_cursor.execute(insert_query, records_to_insert)
                except Exception as e:
                    logging.info("Exchange data insert failed ", e)
                    self.data_conn.rollback()
                else:
                    self.data_conn.commit()

            loop_start_time += timedelta(days=5)

    def update_exchange_data(self):
        for coin in binance.get_coin_list():
            latest_timestamp = self.exchange_data_get_latest_timestamp(coin)
            if latest_timestamp is None:
                latest_timestamp = binance.get_exchange_startime()
            else:
                latest_timestamp = latest_timestamp.replace(tzinfo=pytz.UTC) - timedelta(minutes=1)

            logging.info("Updating binance data from %s" % (latest_timestamp))
            self.grab_exchange_data(coin, latest_timestamp,
                                    datetime.utcnow().replace(tzinfo=pytz.UTC) - timedelta(minutes=1))

    def training_data_get_latest_timestamp(self, coin, aggregation):
        self.training_cursor.execute(
            "SELECT MAX(open_time) FROM cointron.training_data WHERE coin=%s AND aggregation=%s", (coin, aggregation))
        return self.training_cursor.fetchone()[0]

    def create_training_data_in_memory(self, coin, aggregation, start_time, end_time):
        self.training_cursor.execute(
            "SELECT * FROM cointron.binance_data WHERE coin=%s AND open_time >= %s  AND open_time <= %s ORDER BY open_time ASC",
            (coin, start_time, end_time))

        df = pd.DataFrame(self.training_cursor.fetchall(), columns=self.get_binance_data_collumns())
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
                    logging.info(
                        "Training data(%s,%d) progress %2f %%" % (coin, aggregation, (idx / df.shape[0]) * 100))

                    insert_query = "INSERT INTO cointron.training_data VALUES" + ','.join(
                        ['%s'] * len(training_data)) + " ON CONFLICT DO NOTHING"
                    try:
                        self.training_cursor.execute(insert_query, training_data)
                    except Exception as e:
                        logging.info("Failed to insert training data ", e)
                        self.training_conn.rollback()
                        return
                    else:
                        self.training_conn.commit()
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
        for aggregation in aggregations:
            for coin in binance.get_coin_list():
                latest_data_timestamp = self.exchange_data_get_latest_timestamp(coin)
                latest_training_timestamp = self.training_data_get_latest_timestamp(coin, aggregation)
                if latest_data_timestamp is not None:
                    if latest_training_timestamp is None:
                        logging.info("Creating training data from scratch (%s,%d)..." % (coin, aggregation))
                        latest_training_timestamp = binance.get_exchange_startime()
                    else:
                        logging.info("Updating Training Data for %s,%d from %s to %s" % (
                            coin, aggregation, latest_training_timestamp,
                            latest_data_timestamp))
                        latest_training_timestamp -= timedelta(minutes=aggregation * 10)
                    self.create_training_data_in_memory(coin, aggregation,
                                                        latest_training_timestamp.replace(tzinfo=pytz.UTC),
                                                        latest_data_timestamp.replace(tzinfo=pytz.UTC))

    def get_training_data(self, coin, aggregation=15, start_time=binance.get_exchange_startime(),
                          end_time=datetime.utcnow().replace(tzinfo=pytz.UTC)):
        self.training_cursor.execute(
            "SELECT open_time,open_value,high,low,close_value,volume,quote_asset_volume,trades," +
            "taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ma5,ma10,prediction FROM cointron.training_data " +
            "WHERE coin=%s AND aggregation=%s AND open_time >= %s  AND open_time <= %s ORDER BY open_time ASC",
            (coin, aggregation, start_time, end_time))

        return pd.DataFrame(self.training_cursor.fetchall(),
                            columns=['open_time', 'open_value', 'high', 'low', 'close_value',
                                     'volume', 'quote_asset_volume', 'trades',
                                     'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ma5', 'ma10',
                                     'prediction'])

    def get_latest_prediction_data(self, coin, aggregation):
        timestamp = self.exchange_data_get_latest_timestamp(coin)

        self.data_cursor.execute(
            "SELECT * FROM cointron.binance_data WHERE coin=%s AND open_time >= %s AND open_time <= %s ORDER BY open_time ASC",
            (coin, timestamp - timedelta(minutes=10 * aggregation), timestamp))

        df = pd.DataFrame(self.data_cursor.fetchall(), columns=self.get_binance_data_collumns())

        return pd.DataFrame([self.training_worker(coin, aggregation, df, df.shape[0] - 1, False)],
                            columns=['open_time', 'open_value', 'high', 'low', 'close_value', 'volume',
                                     'quote_asset_volume',
                                     'trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ma5', 'ma10'])

    @staticmethod
    def get_aggregations():
        return aggregations
