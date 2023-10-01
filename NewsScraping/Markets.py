# This File aims at Getting Open markets in every moment of the day this Algorithm will be launched.

import pandas as pd
import numpy as np
import yfinance as yf

# How to do it ? Just checking if the latest date of the main stock index of various countries corresponds to today.

class Markets:
    name = "Check Markets conditions"

    def __init__(self):
        pass

    def getOpenMarkets (self):

        import pandas as pd
        import yfinance as yf
        import pytz
        from sqlalchemy import create_engine
        from datetime import datetime

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/YahooFinance')
        query = 'SELECT * FROM public."WorldIndexTickers"'
        marketIndex = pd.read_sql(query, engine)

        stocks = marketIndex[marketIndex['Type'] == 'StockIndex']
        futures = marketIndex[marketIndex['Type'] == 'future']

        openMarkets = list()
        for index in stocks['YFTicker']:
            index1 = yf.Ticker(index).history('1d', '1m')
            if index1.empty == False:
                # Define the time zone
                actual_timezone = pytz.timezone('Europe/Rome')

                # Convert your index to the time zone
                index1.index = index1.index.tz_convert(actual_timezone)

                # Take last observation
                lastTrade = pd.to_datetime(index1.reset_index()['Datetime'].tail(1).reset_index()['Datetime'])[0]

                # Check the last trade date to be the same as today's date
                if lastTrade.date() == datetime.today().date():
                    # Check the last trade Hour to be the same hour of now (just with hours,because with minutes as
                    # well could be misleading
                    if lastTrade.hour == datetime.today.hour:
                        openMarkets.append(stocks['Country'][stocks['YFTicker'] == index].reset_index()['Country'][0])

            else:
                print('Not Found: check connection')

        # Same for the futures tickers

        openMarketsFutures = list()
        for index in futures['YFTicker']:
            index1 = yf.Ticker(index).history('2d')
            if index1.empty == False:
                actual_timezone = pytz.timezone('Europe/Rome')
                index1.index = index1.index.tz_convert(actual_timezone)
                lastTrade = pd.to_datetime(index1.reset_index()['Date'].tail(1).reset_index()['Date'])[0]

                if lastTrade.date() == datetime.today().date():
                    if lastTrade.hour == datetime.today.hour:
                        openMarketsFutures.append(futures['Country'][futures['YFTicker'] == index].reset_index()['Country'][0])
            else:
                print('Not Found: check connection')

        return openMarkets + openMarketsFutures

    #TODO: Implement the Stock index Picker according to open markets.
