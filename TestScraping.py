import pandas as pd
import numpy as np
import yfinance as yf
from NewsScraping.Markets import Markets

openMarkets = Markets().getOpenMarkets()
print('Markets Open Now:', len(openMarkets))

stockIndex = Markets().getStockIndex()

print('Number of stoks selected:', len(stockIndex))
