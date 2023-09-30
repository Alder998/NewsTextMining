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

        openMarkets = list()
        for index in marketIndex['YFTicker']:
            index1 = yf.Ticker(index).history('1d', '1m')
            if index1.empty == False:
                # Definisci il fuso a cui vuoi convertire
                actual_timezone = pytz.timezone('Europe/Rome')

                # Converti il tuo indice a quel fuso
                index1.index = index1.index.tz_convert(actual_timezone)

                # prendi l'ultimo dato dell'indice
                lastTrade = pd.to_datetime(index1.reset_index()['Datetime'].tail(1).reset_index()['Datetime'])[0]
                # print(lastTrade)

                # Controlla che la data sia la stessa di oggi
                if lastTrade.date() == datetime.today().date():
                    # controlla che l'ora sia la stessa di ora (al minuto è rischioso per la velocita della connessione)
                    if lastTrade.hour == datetime.today.hour:
                        openMarkets.append(marketIndex['Country'][marketIndex['YFTicker'] == index].reset_index()['Country'][0])

            else:
                print('Not Found: check connection')

        return openMarkets
