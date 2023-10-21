from NewsScraping.Markets import Markets
from NewsScraping.Scraper import Scraper
from datetime import datetime

# Get the Open Markets
openMarkets = Markets().getOpenMarkets()
# Gather the stock Index
stockIndex = Markets().getStockIndex()

if len(openMarkets) != 0:

    print('Markets Open Now:', len(openMarkets))
    print('Number of stocks selected:', len(stockIndex))

    # Take the News, and the related Return and Volume % Change
    total = Scraper(stockIndex).MassiveScraper()

    updated = Scraper(stockIndex).updateDataBase(total)

    print(updated)

    # See the download Statistics about today
    stat = Scraper(stockIndex).generateStatistics('today')

else:
    print('All the markets are closed')

# if it is Saturday, update the financial data
if datetime.today().weekday() == 5:
    upd = Scraper(stockIndex).updateFinancialData()

    # See the download Statistics about the total database
    stat = Scraper(stockIndex).generateStatistics('total')





