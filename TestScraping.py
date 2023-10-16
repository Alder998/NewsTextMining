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

total = Scraper(stockIndex).mergeStockNewsData()

updated = Scraper(stockIndex).updateDataBase(total)

print(updated)

