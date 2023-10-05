import pandas as pd
import numpy as np
import yfinance as yf
from NewsScraping.Markets import Markets
from NewsScraping.Scraper import Scraper

openMarkets = Markets().getOpenMarkets()
print('Markets Open Now:', len(openMarkets))

stockIndex = Markets().getStockIndex()[0:10]

print('Number of stoks selected:', len(stockIndex))

#newsSample = Scraper(stockIndex).getSingleStockMarketNews(source = 'Bing')
returnsAndVolumes = Scraper(stockIndex).getStocksData()

print(returnsAndVolumes)
