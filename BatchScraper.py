from NewsScraping.Markets import Markets
from NewsScraping.Scraper import Scraper
from datetime import datetime

# Get the Open Markets
openMarkets = Markets().getOpenMarkets()
# Gather the stock Index
default = input('Default Download?[Y/N]:')

if default == 'Y':
    stockIndex = Markets().getStockIndex(option = 'Index and Random', randomStocksUS=200, randomStocksExUS=50,
                                     excludeAlreadyProcessed=True)

else:
    # Define Inputs
    option = int(input('Select the way to gather the Stock Index [0:Index only, 1:Index and Random, 2:Random only]:'))
    optionList = ['Index only', 'Index and Random', 'Random only']
    optionChoice = optionList[option]
    print('Selected:', optionChoice)
    if (optionChoice == 'Index and Random') | (optionChoice == 'Random only'):
        randomUS = int(input('Select the Random Stocks from US Stocks [number]:'))
        randomExUS = int(input('Select the Random Stocks from ex-US Stocks [number]:'))
        if optionChoice == 'Random only':
            excludeAlreadyProcessed = input('Exclude Already Processed?[Y/N]:')
            if excludeAlreadyProcessed == 'Y':
                exclude = True
            else:
                exclude = False
        else:
            exclude = False

    else:
        randomUS = 0
        randomExUS = 0
        exclude = False

    # Launch the Scraper
    stockIndex = Markets().getStockIndex(option=optionChoice, randomStocksUS=randomUS, randomStocksExUS=randomExUS,
                                         excludeAlreadyProcessed=exclude)


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