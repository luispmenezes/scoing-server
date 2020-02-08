import datetime

import pandas as pd
import psycopg2
import pytz
from psycopg2 import pool

from data_miner.binance import get_coin_list
from data_miner.training_generator import TrainingGenerator


class TraderManager:

    def __init__(self, db_host, db_port, db_name, db_user, db_password, predictor, training_generator, logger):
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
            self.predictor = predictor
            self.training_generator = training_generator
        except (Exception, psycopg2.Error) as error:
            self.logger.info("Error establishing db connection", error)

    def connect_to_db(self):
        self.pool = psycopg2.pool.SimpleConnectionPool(1, 1, user=self.db_user, password=self.db_password,
                                                       host=self.db_host, port=self.db_port, database=self.db_name)
        self.conn = self.pool.getconn()
        self.cursor = self.conn.cursor()

    def generate_training_data(self):
        self.logger.info("Generating trader training data")

        self.cursor.execute("DELETE FROM cointron.trader_training_data")
        self.conn.commit()

        for coin in get_coin_list():
            data = {}
            preds = {}
            training_data = []
            aggregations = TrainingGenerator.get_aggregations()

            for agg in aggregations:
                data[agg] = self.training_generator.get_training_data((coin,), agg)
                data_values = data[agg].drop(columns=['prediction'])
                self.logger.info("Getting predictions for %d model" % agg)
                preds[agg] = self.predictor.predict(data_values, agg)

            biggest_agg = max(aggregations)

            for i in range(len(data[biggest_agg])):
                valid_record = True
                open_time = data[biggest_agg].iloc[i]['open_time']
                training_entry = (open_time, coin, float(data[biggest_agg].iloc[i]['close_value']))
                for agg in aggregations:
                    if open_time in preds[agg]:
                        training_entry += (float(preds[agg][open_time]),)
                    else:
                        valid_record = False

                if valid_record:
                    training_data.append(training_entry)

            insert_query = "INSERT INTO cointron.trader_training_data VALUES" + ','.join(['%s'] * len(training_data)) \
                           + " ON CONFLICT DO NOTHING"

            if len(training_data) > 0:
                try:
                    self.cursor.execute(insert_query, training_data)
                except psycopg2.ProgrammingError as e:
                    self.logger.info("Failed to insert trader training data (rolling back) ", e)
                    self.conn.rollback()
                    exit(-1)
                else:
                    self.conn.commit()

    def get_training_data(self, coins, start_time=datetime.datetime(2017, 8, 17, 4, 0, tzinfo=pytz.UTC),
                          end_time=datetime.datetime(2099, 1, 1, 0, 0, tzinfo=pytz.UTC)):
        self.cursor.execute(
            "SELECT * FROM cointron.trader_training_data WHERE coin IN %s AND open_time >= %s  " +
            "AND open_time <= %s ORDER BY open_time ASC", (coins, start_time, end_time))

        df_headers = ["open_time", "coin", "close_value"]

        for agg in TrainingGenerator.get_aggregations():
            df_headers.append("pred_" + str(agg))

        return pd.DataFrame(self.cursor.fetchall(), columns=df_headers)
