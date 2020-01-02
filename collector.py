import concurrent.futures
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
from psycopg2 import pool
import pytz

import binance


class Collector:

    def __init__(self, db_host, db_port, db_name, db_user, db_password, binance_api_key, binance_api_secret):
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(1, 20, user=db_user,
                                                             password=db_password,
                                                             host=db_host,
                                                             port=db_port,
                                                             database=db_name)
            self.connection = self.pool.getconn()
            self.db_cursor = self.connection.cursor()
            self.binance_client = binance.Client(binance_api_key, binance_api_secret)
        except (Exception, psycopg2.Error) as error:
            print("Error establishing db connection", error)

    def exchange_data_get_latest_timestamp(self, coin):
        self.db_cursor.execute("SELECT MAX(open_time) FROM cointron.binance_data WHERE coin=%s", (coin,))
        return self.db_cursor.fetchone()[0]

    def grab_exchange_data(self, coin, start_time=binance.get_exchange_startime(),
                           end_time=datetime.utcnow().replace(tzinfo=pytz.UTC)):
        loop_start_time = start_time

        while loop_start_time <= end_time:
            loop_end_time = loop_start_time + timedelta(days=5) - timedelta(minutes=1)
            if loop_end_time > end_time:
                loop_end_time = end_time

            print('Getting binance data for %s to %s' % (loop_start_time, loop_end_time))

            records_to_insert = []
            for data in self.binance_client.klines(coin, "1m", loop_start_time, loop_end_time):
                records_to_insert.append(
                    (coin, datetime.fromtimestamp((data[0] / 1000.0), tz=pytz.UTC), float(data[1]), float(data[2]),
                     float(data[3]), float(data[4]), float(data[5]), float(data[7]), int(data[8]),
                     float(data[9]), float(data[10])))

            if len(records_to_insert) > 0:
                insert_query = "INSERT INTO cointron.binance_data VALUES" + ','.join(['%s'] * len(records_to_insert))
                self.db_cursor.execute(insert_query, records_to_insert)
                self.connection.commit()

            loop_start_time += timedelta(days=5)

    def update_exchange_data(self):
        for coin in binance.get_coin_list():
            latest_timestamp = self.exchange_data_get_latest_timestamp(coin)
            if latest_timestamp is None:
                latest_timestamp = binance.get_exchange_startime()
            else:
                latest_timestamp = latest_timestamp.replace(tzinfo=pytz.UTC) + timedelta(minutes=1)

            print("Updating binance data from %s" % (latest_timestamp))
            self.grab_exchange_data(coin, latest_timestamp)

    def training_data_get_latest_timestamp(self, coin):
        self.db_cursor.execute("SELECT MAX(open_time) FROM cointron.training_data WHERE coin=%s", (coin,))
        return self.db_cursor.fetchone()[0]

    def grab_training_data(self, coin, aggregation=15,
                           start_time=binance.get_exchange_startime() + timedelta(minutes=15),
                           end_time=(datetime.utcnow() - timedelta(minutes=15)).replace(tzinfo=pytz.UTC)):
        cursor_time = start_time

        records_to_insert = []

        batch_size = 5
        worker_connections = [self.pool.getconn() for i in range(batch_size)]
        worker_cursors = [conn.cursor() for conn in worker_connections]

        while cursor_time <= end_time:

            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                agg_futures = []

                for i in range(0, batch_size - 1):
                    print("Create Training Data for %s (%s)" % (coin, cursor_time))
                    agg_futures.append(executor.submit(self.training_data_worker, coin, aggregation, cursor_time,
                                                       worker_cursors[i]))
                    cursor_time += timedelta(minutes=1)

                for future in concurrent.futures.as_completed(agg_futures):
                    if future.result() is not None:
                        records_to_insert.append(future.result())

            if len(records_to_insert) > 0:
                print("Insert Training")
                insert_query = "INSERT INTO cointron.training_data VALUES" + ','.join(['%s'] * len(records_to_insert))
                self.db_cursor.execute(insert_query, records_to_insert)
                self.connection.commit()
                records_to_insert.clear()

        for i in range(batch_size):
            worker_cursors[i].close()
            self.pool.putconn(worker_connections[i])

    def training_data_worker(self, coin, aggregation, cursor_time, cursor):
        cursor.execute("SELECT open_value FROM cointron.binance_data WHERE coin=%s AND open_time=%s",
                       (coin, cursor_time + timedelta(minutes=aggregation)))
        prediction = cursor.fetchone()

        if prediction is not None:

            prediction = prediction[0]

            cursor.execute(
                "SELECT * FROM cointron.binance_data WHERE coin=%s AND open_time >= %s AND open_time <= %s ORDER BY open_time ASC",
                (coin, cursor_time - timedelta(minutes=aggregation), cursor_time))

            df = pd.DataFrame(cursor.fetchall(), columns=['coin', 'open_time', 'open_value', 'high', 'low',
                                                                 'close_value', 'volume', 'quote_asset_volume',
                                                                 'trades',
                                                                 'taker_buy_base_asset_volume',
                                                                 'taker_buy_quote_asset_volume'])

            if df.shape[0] > 0:
                open_value = df['open_value'].iloc[0]
                high = df['high'].max()
                low = df['low'].min()
                close_value = df['close_value'].iloc[-1]
                volume = df['volume'].sum()
                quote_asset_volume = df['quote_asset_volume'].mean()
                trades = int(df['trades'].sum())
                taker_buy_base_asset_volume = df['taker_buy_base_asset_volume'].mean()
                taker_buy_quote_asset_volume = df['taker_buy_quote_asset_volume'].mean()
                prediction_delta = (prediction / close_value) - 1

                return (coin, aggregation, cursor_time, open_value, high, low, close_value, volume,
                        quote_asset_volume, trades, taker_buy_base_asset_volume,
                        taker_buy_quote_asset_volume,
                        prediction_delta)
            else:
                print("No entries to aggregate")

        else:
            print("No prediction at timestamp %s" % (cursor_time + timedelta(minutes=aggregation)))

    def update_training_data(self):
        for coin in binance.get_coin_list():
            latest_data_timestamp = self.exchange_data_get_latest_timestamp(coin)
            latest_timestamp = self.training_data_get_latest_timestamp(coin)
            if latest_timestamp is None:
                latest_timestamp = binance.get_exchange_startime()
            print("Updating Training Data for %s from %s to %s" % (coin, latest_timestamp, latest_data_timestamp))
            self.grab_training_data(coin, 15, latest_timestamp.replace(tzinfo=pytz.UTC) + timedelta(minutes=1),
                                    latest_data_timestamp.replace(tzinfo=pytz.UTC) - timedelta(minutes=15))

    def get_training_data(self, coin, aggregation=15, start_time=binance.get_exchange_startime(),
                          end_time=datetime.utcnow().replace(tzinfo=pytz.UTC)):
        self.db_cursor.execute(
            "SELECT open_time,open_value,high,low,close_value,volume,quote_asset_volume,trades," +
            "taker_buy_base_asset_volume,taker_buy_quote_asset_volume,prediction FROM cointron.training_data " +
            "WHERE coin=%s AND aggregation=%s AND open_time >= %s  AND open_time <= %s ORDER BY open_time ASC",
            (coin, aggregation, start_time, end_time))

        return pd.DataFrame(self.db_cursor.fetchall(), columns=['open_time', 'open_value', 'high', 'low', 'close_value',
                                                                'volume', 'quote_asset_volume', 'trades',
                                                                'taker_buy_base_asset_volume',
                                                                'taker_buy_quote_asset_volume',
                                                                'prediction'])

    def get_latest_prediction_data(self, coin, aggregation):
        timestamp = datetime.now(tz=pytz.UTC)
        self.db_cursor.execute(
            "SELECT * FROM cointron.binance_data WHERE coin=%s AND open_time >= %s AND open_time <= %s ORDER BY open_time ASC",
            (coin, timestamp - timedelta(minutes=aggregation), timestamp))

        df = pd.DataFrame(self.db_cursor.fetchall(), columns=['coin', 'open_time', 'open_value', 'high', 'low',
                                                              'close_value', 'volume', 'quote_asset_volume',
                                                              'trades', 'taker_buy_base_asset_volume',
                                                              'taker_buy_quote_asset_volume'])
        open_value = df['open_value'].iloc[0]
        high = df['high'].max()
        low = df['low'].min()
        close_value = df['close_value'].iloc[-1]
        volume = df['volume'].sum()
        quote_asset_volume = df['quote_asset_volume'].mean()
        trades = int(df['trades'].sum())
        taker_buy_base_asset_volume = df['taker_buy_base_asset_volume'].mean()
        taker_buy_quote_asset_volume = df['taker_buy_quote_asset_volume'].mean()

        return pd.DataFrame([(open_value, high, low, close_value, volume, quote_asset_volume, trades,
                              taker_buy_base_asset_volume, taker_buy_quote_asset_volume)],
                            columns=['coin', 'open_time', 'open_value', 'high', 'low',
                                     'close_value', 'volume', 'quote_asset_volume',
                                     'trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume'])
