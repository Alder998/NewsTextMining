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
        futures = marketIndex[(marketIndex['Type'] == 'future')]

        openMarkets = list()
        for index in stocks['YFTicker']:
            index1 = yf.Ticker(index).history('2d', '1m')
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
                    if lastTrade.hour == datetime.today().hour:
                        openMarkets.append(stocks['Country'][stocks['YFTicker'] == index].reset_index()['Country'][0])

        # Same for the futures tickers

        openMarketsFutures = list()
        for index in futures['YFTicker']:
            index1 = yf.Ticker(index).history('2d')
            if index1.empty == False:
                actual_timezone = pytz.timezone('Europe/Rome')
                index1.index = index1.index.tz_convert(actual_timezone)
                lastTrade = pd.to_datetime(index1.reset_index()['Date'].tail(1).reset_index()['Date'])[0]

                if lastTrade.date() == datetime.today().date():
                    if lastTrade.hour == datetime.today().hour:
                        openMarketsFutures.append(futures['Country'][futures['YFTicker'] == index].reset_index()['Country'][0])

        return openMarkets + openMarketsFutures


    def getStockIndex (self, randomStocksUS=10, randomStocksExUS=50):

        import pandas as pd
        import yfinance as yf
        import psycopg2
        import random
        from sqlalchemy import create_engine
        from datetime import datetime

        # Import the Database with all the stocks traded

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/YahooFinance')
        query = 'SELECT * FROM public."AllStockTraded"'
        allStocks = pd.read_sql(query, engine)

        # Get all the Open Markets

        openMarkets = self.getOpenMarkets()

        # Filter the allStocks sample for the open Markets
        IndexPresentStocks = list()
        randomStocks = list()
        for market in openMarkets:
            marketStocks = allStocks[allStocks['Country'] == market]
            # Start taking the main market Index
            IPStocks = list(marketStocks['Ticker'][marketStocks['IndexPresent'] == 'Y'])
            IndexPresentStocks.append(IPStocks)

            # Then, to add variability, take 50 random stocks for countries that are not US, and 20 for US (because
            # We have the S&P 500 for US)

            if market == 'USA':
                randomPick = randomStocksUS
            else:
                randomPick = randomStocksExUS

            marketStocksExIndex = list(marketStocks['Ticker'][marketStocks['IndexPresent'] != 'Y'])
            rp = list()
            for stockR in range(randomPick):
                rp.append(random.choice(marketStocksExIndex))
            randomStocks.append(rp)

        # Insert every element in the same list

        IndexList = list()
        randomList = list()

        # Iterate through the list of indexes
        for IntListIndex in IndexPresentStocks:
            IndexList.extend(IntListIndex)

        # Iterate through the list of random stocks
        for IntListRandom in randomStocks:
            randomList.extend(IntListRandom)

        # return both the elements
        return IndexList+randomList


    def updateIndexComponentsOnDatabase (self):

        # Fast Scraping to update the index Components data

        import requests
        from bs4 import BeautifulSoup
        import numpy as np
        import pandas as pd
        import psycopg2
        from sqlalchemy import create_engine

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/YahooFinance')
        query = 'SELECT * FROM public."WorldIndexTickers"'
        marketIndex = pd.read_sql(query, engine)

        stockIndex = marketIndex[marketIndex['Type'] == 'StockIndex']

        indexPresent = list()
        for indexName in stockIndex['YFTicker']:

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

            index = indexName.replace('^', '%5E')
            target_url = "https://finance.yahoo.com/quote/" + index + "/components"

            resp = requests.get(target_url, headers=headers)
            soup = BeautifulSoup(resp.text, 'html.parser')  # Pare che l'URL sia libero per fare scraping

            a_tag = soup.findAll('a')

            fs = list()
            for i in a_tag:
                fs.append(i)

            a = pd.Series(fs).astype(str)

            rawData = a[(a.str.contains('href="/quote/')) & (~a.str.contains('aria'))].reset_index()
            del [rawData['index']]

            if rawData.empty == False:

                componentsList = list()
                for i in range(len(rawData[0])):
                    start = rawData[0][i].find('href="/quote/') + len('href="/quote/')
                    end = rawData[0][i][start:].find('?')
                    comp = rawData[0][i][start:start + end]
                    componentsList.append(comp)

                componentForm = pd.Series(componentsList)
                componentForm = pd.concat([componentForm, pd.Series(np.full(len(componentForm), indexName))],
                                          axis=1).set_axis(['Component', 'Index'], axis=1)
                indexPresent.append(componentForm)

        indexPresent = pd.concat([series for series in indexPresent], axis=0).reset_index()
        del [indexPresent['index']]

        # Add SP500

        SP = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\S&P500 Constituents.xlsx")['Ticker']
        SP = pd.DataFrame(SP.set_axis(['Component'], axis=1))

        indexPresent = pd.concat([indexPresent, SP], axis=0).reset_index()
        del [indexPresent['index']]

        # Import AllStocks DB

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/YahooFinance')
        query = 'SELECT * FROM public."AllStockTraded"'
        allStocks = pd.read_sql(query, engine)

        # indexPresent['Component'] = indexPresent['Component'].str.split('.').str[0]

        for stock in indexPresent['Component']:
            allStocks.loc[allStocks['Ticker'] == stock, 'IndexPresent'] = 'Y'

        # Save and Update

        file = allStocks

        connection = psycopg2.connect(
            database="YahooFinance",
            user="postgres",
            password="Davidescemo",
            host="localhost",
            port="5432"
        )

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/YahooFinance')
        file.to_sql('AllStockTraded', engine, if_exists='replace', index=False)

        return allStocks







