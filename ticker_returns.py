from yahooquery import Ticker
import time
from requests.exceptions import ConnectionError, Timeout
import numpy as np
import pandas as pd
from datetime import datetime as dt

class TickerData:
    """
    outputs stock prices/returns data
    for given ticker/s, start and end dates
    with daily/weekly/semi-monthly/monthly/semi-annual/annual frequency 
    ------------------------------------
    HOW TO USE:
    initiate class with following arguments
    ticker/s
    start_date 
    end_date
    --------------------
    ex: 
    start = dt(1990,1,1)
    end = dt(2024,12,31)
    weekly_prices_data = TickerData('AAPL, MSFT', start, end).get_prices('weekly')
    """
    def __init__(self,ticker,start_date,end_date):
        self._ticker = ticker
        self._start_date = start_date
        self._end_date = end_date
         
    def get_prices(self,prices_frequency):
        """
        returns close price data of stock/s given by ticker/s
        for the timeperiod and frequency
        """
        self._print_invalid_tickers() 
        daily_prices = self._daily_prices()
        if daily_prices is not None:
            if prices_frequency == 'daily':
                return daily_prices
            else:
                to_freq = self._convert_frequency_label(prices_frequency)
                return daily_prices.resample(to_freq).last()
        else:
            print("Failed to fetch data after multiple times.")    

    def get_returns(self,returns_frequency,is_log_return):
        daily_returns = self._daily_returns(is_log_return)
        if returns_frequency == 'daily':
            return daily_returns.dropna()
        else:
            to_freq = self._convert_frequency_label(returns_frequency)
            return daily_returns.resample(to_freq).agg(lambda x: (x + 1).prod() - 1)    
    
    def _print_invalid_tickers(self):
        """
        prints invalid tickers
        check before pulling data
        """
        t = Ticker(self._ticker, validate=True)
        if t.invalid_symbols is not None:
            print("Invalid Tickers entered: ",t.invalid_symbols)
            print()
    
    def _daily_prices(self, retries=5, delay = 2):
        """
        returns daily close price data of stock/s given by ticker/s
        for the input time period
        """
        for attempt in range(retries):
            try:
                yahoo_ticker = Ticker(self._ticker, asynchronous=True)
                ticker_data = yahoo_ticker.history(start=self._start_date,
                                                end=self._end_date,
                                                interval = '1d',
                                                adj_ohlc=True).reset_index()
                # to get time (hr,min,sec) into date format
                ticker_data['date']= pd.to_datetime(ticker_data['date']).dt.date
                # pivot table 
                daily_prices = ticker_data.pivot(index='date', 
                                                columns='symbol', 
                                                values='close')
                daily_prices.index = pd.to_datetime(pd.to_datetime(daily_prices.index).date)
                daily_prices = daily_prices.fillna(0)
                return daily_prices
            except (ConnectionError, Timeout) as e:
                print(f"Error: {e} Retrying in {delay} seconds.." )
                time.sleep(delay)
        return None    

    def _daily_returns(self,is_log_return):
        """
        returns daily returns calculated from daily close price data of stock/s 
        ------------------------------------
        HOW TO USE:
        is_log_return: bool
            can take True/False. 
        """
        daily_returns = self._daily_prices().pct_change(axis=0)
        if is_log_return:
            return np.log(1+daily_returns)
        else:
            return daily_returns
      
    def _convert_frequency_label(self,frequency):
        if frequency == 'daily':
            return 'D'
        elif frequency == 'weekly':
            return 'W-FRI'
        elif frequency == 'semi-monthly':
            return 'SE'
        elif frequency == 'monthly':
            return 'ME'
        elif frequency == 'quarterly':
            return 'Q-DEC'
        elif frequency == 'semi-annual':
            return '2Q-DEC'
        elif frequency == 'annual':
            return 'A-DEC'
        else:
            raise ValueError("Define frequency : daily/weekly/semi-monthly/monthly/quarterly/semi-annual/annual")