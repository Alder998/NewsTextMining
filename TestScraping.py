import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import create_engine
from NewsScraping.Markets import Markets
from NewsScraping.Scraper import Scraper

openMarkets = Markets().getOpenMarkets()
print('Markets Open Now:', len(openMarkets))

stockIndex = Markets().getStockIndex()

print('Number of stoks selected:', len(stockIndex))

#newsSample = Scraper(stockIndex).getSingleStockMarketNews(source = 'Bing')
#returnsAndVolumes = Scraper(stockIndex).getStocksData()

#total = Scraper(stockIndex).mergeStockNewsData()

engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
query = 'SELECT * FROM public."News_Scraping_DailyV2"'
base = pd.read_sql(query, engine)

updated = Scraper(stockIndex).updateDataBase(base)

print(updated)
