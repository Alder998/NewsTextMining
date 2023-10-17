from NewsScraping.Markets import Markets
from NewsScraping.Scraper import Scraper

openMarkets = Markets().getOpenMarkets()
print('Markets Open Now:', len(openMarkets))

stockIndex = Markets().getStockIndex()

print('Number of stocks selected:', len(stockIndex))

total = Scraper(stockIndex).mergeStockNewsData()

updated = Scraper(stockIndex).updateDataBase(total)

print(updated)

