from NewsScraping.Markets import Markets
from NewsScraping.Scraper import Scraper
from datetime import datetime

# Get the Open Markets
openMarkets = Markets().getOpenMarkets()
print('Markets Open Now:', len(openMarkets))

# Gather the stock Index
stockIndex = Markets().getStockIndex()

print('Number of stocks selected:', len(stockIndex))

# Take the News, and the related Return and Volume % Change
total = Scraper(stockIndex).MassiveScraper()

updated = Scraper(stockIndex).updateDataBase(total)

print(updated)

# if it is Saturday, update the financial data
if datetime.today().weekday() == 5:
    upd = Scraper(stockIndex).updateFinancialData()

# See the download Statistics about today, or total (but we are gonna see the total JUST in the case of return Update)

todayData = updated[updated['Date'] == datetime.today().strftime('%Y.%m.%d')]

if todayData.empty == False:
    stat = Scraper(stockIndex).generateStatistics('today')
else:
    stat = Scraper(stockIndex).generateStatistics('total')

