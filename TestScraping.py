from NewsScraping.Markets import Markets
from NewsScraping.Scraper import Scraper
from datetime import datetime

# Get the Open Markets
openMarkets = Markets().getOpenMarkets()
# Gather the stock Index
stockIndex = Markets().getStockIndex(option = 'Random only', randomStocksUS=200, randomStocksExUS=100)

if len(openMarkets) != 0:

    print('Markets Open Now:', len(openMarkets))
    print('Number of stocks selected:', len(stockIndex))

    # Take the News, and the related Return and Volume % Change
    total = Scraper(stockIndex).MassiveScraper()

    updated = Scraper(stockIndex).updateDataBase(total)

    print(updated)

    # See the download Statistics about today
    if 'USA' in openMarkets:
        stat = Scraper(stockIndex).generateStatistics('today')
    if 'Italy' or 'France' or 'Germany' or 'Spain' or'India' or 'Hong Kong' or 'Singapore' in openMarkets:
        stat = Scraper(stockIndex).generateStatistics('today', expand=['Europe', 'Asia'])

else:
    print('All the markets are closed')

# if it is Saturday or Sunday, update the financial data
if (datetime.today().weekday() == 5) | (datetime.today().weekday() == 6):
    upd = Scraper(stockIndex).updateFinancialData()

    # See the download Statistics about the total database
    stat = Scraper(stockIndex).generateStatistics('total')





