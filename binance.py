import collections
import hashlib
import hmac

import pytz
import requests
import datetime
import logging
from enum import Enum


class Symbol(Enum):
    BITCOIN = 'BTCUSDT'
    ETHEREUM = 'ETHUSDT'
    BINANCE_COIN = 'BNBUSDT'
    LITECOIN = 'LTCUSDT'
    RIPPLE = 'XRPUSDT'


class Client:
    ENDPOINT = 'https://api.binance.com'
    KLINE_PATH = '/api/v3/klines'
    PING_PATH = '/api/v3/ping'
    TIME_PATH = '/api/v3/time'
    EXCHANGE_INFO_PATH = '/api/v3/exchangeInfo'
    PRICE_TICKER_PATH = '/api/v3/ticker/price'
    BOOK_TICKER_PATH = '/api/v3/ticker/bookTicker'

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

    def send_request(self, method, path, params):
        headers = {'apiKey': self.api_key, 'secretKey': self.api_secret}
        logging.debug('Sending binance request (%s %s %s)' % (method, self.ENDPOINT + path, params))
        response = requests.request(method, headers=headers, params=params, url=self.ENDPOINT + path)
        if response.status_code < 200 or response.status_code >= 300:
            raise Exception('Binance request failed %d (%s)' % (response.status_code, response.text))
        else:
            return response.status_code, response.json()

    def send_signed_request(self, method, path, params, recv_window=5000):
        params['recvWindow'] = recv_window
        params['signature'] = self.generate_signature(params)
        self.send_request(method, path, params)

    def generate_signature(self, params):
        ordered_params = collections.OrderedDict(sorted(params.items()))
        query_string = '&'.join(["{}={}".format(d[0], d[1]) for d in ordered_params])
        signature = hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256)
        return signature.hexdigest()

    def ping(self):
        return self.send_request('GET', self.PING_PATH, {})

    def server_time(self):
        return self.send_request('GET', self.TIME_PATH, {})

    def exchange_info(self):
        return self.send_request('GET', self.EXCHANGE_INFO_PATH, {})

    def klines(self, symbol, interval, start_time=None, end_time=None):
        logging.debug('Get Klines (symbol:%s, interval:%s, start_time=%s, end_time=%s)' % (
            symbol, interval, start_time, end_time))
        params = {'symbol': symbol, 'interval': interval, 'limit': 1000}
        if start_time is not None:
            params['startTime'] = get_epoch(start_time)
        if end_time is not None:
            params['endTime'] = get_epoch(end_time)

        jsonData = []
        if len(params) == 5 and interval == '1m' and abs((end_time - start_time).total_seconds() / 60.0) > 1000:
            multipart_end_time = start_time + datetime.timedelta(minutes=1000)
            while multipart_end_time <= end_time:
                params['startTime'] = get_epoch(start_time)
                params['endTime'] = get_epoch(multipart_end_time)
                jsonData += self.send_request('GET', self.KLINE_PATH, params)[1]
                start_time = multipart_end_time + datetime.timedelta(minutes=1)
                multipart_end_time = start_time + datetime.timedelta(minutes=1000)
                if multipart_end_time > end_time:
                    params['startTime'] = get_epoch(start_time)
                    params['endTime'] = get_epoch(end_time)
                    jsonData += self.send_request('GET', self.KLINE_PATH, params)[1]
        else:
            jsonData += self.send_request('GET', self.KLINE_PATH, params)[1]

        return jsonData

    def price_ticker(self, coin):
        return self.send_request('GET', self.PRICE_TICKER_PATH, {'symbol': coin})

    def book_ticker(self, coin):
        return self.send_request('GET', self.BOOK_TICKER_PATH, {'symbol': coin})


def get_epoch(date):
    return int(date.strftime('%s')) * 1000


def get_exchange_startime():
    return datetime.datetime(2017, 8, 17, 4, 0, tzinfo=pytz.UTC)


def get_coin_list():
    return [s.value for s in Symbol]
