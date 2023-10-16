# The proper scraper class: given an index of tickers, it will go on the internet and get the relevant news

class Scraper:
    name = "Check Markets conditions"

    def __init__(self, stockList):
        self.stockList = stockList
        pass

    # Scarico notizie stock mirati

    def getSingleStockMarketNews(self, source='Bing'):

        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        import numpy as np
        from datetime import datetime
        from IPython.display import clear_output
        from sqlalchemy import create_engine

        # take the stock index
        stockIndex = self.stockList

        # Import the allStock Database to get the country

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/YahooFinance')
        query = 'SELECT * FROM public."AllStockTraded"'
        allStocks = pd.read_sql(query, engine)

        if source == 'CNBC':

            results = list()
            for iterat, stock in enumerate(stockIndex):

                # Track back the market the stock is from
                country = allStocks['Country'][allStocks['Ticker'] == stock].reset_index()['Country'][0]

                # prepare the ticker to be processed
                if '.' in stock:
                    stock = stock[:stock.find('.')]
                else:
                    stock = stock

                print('CNBC: Article Gathering (1 out of 2) - Ticker:', stock, ' - Progress:',
                      round(iterat / len(stockIndex) * 100), '%')

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

                # Mettiamo su l'URL di base: la ricerca di Google News sul Mercato finanziario

                target_url = "https://www.cnbc.com/quotes/" + stock + "?qsearchterm=" + stock

                resp = requests.get(target_url, headers=headers)
                soup = BeautifulSoup(resp.text, 'html.parser')  # Pare che l'URL sia libero per fare scraping

                a_tag = soup.findAll('a', href=True)

                fs = list()
                for i in a_tag:
                    fs.append(i)

                a = pd.Series(fs).astype(str)

                # a.to_excel(r"C:\Users\39328\OneDrive\Desktop\zio pino indice.xlsx")

                newsTitle = a[(a.str.contains('title=') == True)].drop_duplicates().reset_index()
                del [newsTitle['index']]

                if len(newsTitle[0]) > 1:
                    cnbsNews = list()
                    for sNumber in range(1, len(newsTitle)):
                        cnbsNews.append(newsTitle[0][sNumber][
                                        newsTitle[0][sNumber].find('title=') + len('title=') + 1: newsTitle[0][
                                                                                                      sNumber].find(
                                            '>') - 1])

                    # print(pd.DataFrame(cnbsNews).set_axis(['Article'], axis = 1))

                    cnbsNews = pd.concat([pd.DataFrame(
                        np.full(len(cnbsNews), datetime.today().strftime('%Y.%m.%d'))).set_axis(['Date'], axis=1),
                                          pd.DataFrame(cnbsNews).set_axis(['Article'], axis=1),
                                          pd.DataFrame(np.full(len(cnbsNews), 'CNBC News')).set_axis(['Author'],
                                                                                                     axis=1),
                                          pd.DataFrame(np.full(len(cnbsNews), stock)).set_axis(['Ticker'], axis=1),
                                          pd.DataFrame(np.full(len(cnbsNews), country)).set_axis(['Country'], axis = 1)],
                                         axis=1)

                    results.append(cnbsNews)

                else:
                    print('No News found for:', stock, 'on CNBC')

            finalDF = pd.concat(results).dropna()
            finalDF = finalDF.drop_duplicates(subset='Article')

        if source == 'MarketWatch':

            results = list()
            for iterat, stock in enumerate(stockIndex):

                # Track back the market the stock is from
                country = allStocks['Country'][allStocks['Ticker'] == stock].reset_index()['Country'][0]

                # prepare the ticker to be processed
                if '.' in stock:
                    stock = stock[:stock.find('.')]
                else:
                    stock = stock

                print('MarketWatch: Article Gathering (1 out of 2) - Ticker:', stock, ' - Progress:',
                      round(iterat / len(stockIndex) * 100), '%')

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

                # Mettiamo su l'URL di base: la ricerca di Google News sul Mercato finanziario

                target_url = "https://www.marketwatch.com/investing/stock/" + stock

                resp = requests.get(target_url, headers=headers)
                soup = BeautifulSoup(resp.text, 'html.parser')  # Pare che l'URL sia libero per fare scraping

                a_tag = soup.findAll('a', href=True)

                fs = list()
                for i in a_tag:
                    fs.append(i)

                a = pd.Series(fs).astype(str)

                newsTitle = a[(a.str.contains('mw_quote_news">') == True)].drop_duplicates().reset_index()
                del [newsTitle['index']]

                if (newsTitle.empty == False) & (len(newsTitle[0]) > 1):

                    cnbsNews = list()
                    for sNumber in range(1, len(newsTitle)):
                        stringF = newsTitle[0][sNumber][newsTitle[0][sNumber].find('mw_quote_news">') +
                                                        len('mw_quote_news">') + 87: newsTitle[0][sNumber].find(
                            '</a>') - len('</a>') - 21]

                        cnbsNews.append(stringF)

                    cnbsNews = pd.concat([pd.DataFrame(
                        np.full(len(cnbsNews), datetime.today().strftime('%Y.%m.%d'))).set_axis(['Date'], axis=1),
                                          pd.DataFrame(cnbsNews).set_axis(['Article'], axis=1),
                                          pd.DataFrame(np.full(len(cnbsNews), 'MarketWatch')).set_axis(['Author'],
                                                                                                       axis=1),
                                          pd.DataFrame(np.full(len(cnbsNews), stock)).set_axis(['Ticker'], axis=1),
                                          pd.DataFrame(np.full(len(cnbsNews), country)).set_axis(['Country'], axis = 1)],
                                         axis=1)

                    results.append(cnbsNews)

                else:
                    print('No News found for:', stock, 'on MarketWatch')

            finalDF = pd.concat(results).dropna()
            finalDF = finalDF.drop_duplicates(subset='Article')

        if source == 'Bing':

            results = list()
            for iterat, stock in enumerate(stockIndex):

                # Track back the market the stock is from
                country = allStocks['Country'][allStocks['Ticker'] == stock].reset_index()['Country'][0]

                # Mettiamo su l'URL di base: la ricerca di Google News sul Mercato finanziario

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

                target_url = f"https://www.bing.com/news/search?q=stock+market+news+" + stock
                resp = requests.get(target_url, headers=headers)
                soup = BeautifulSoup(resp.text, 'html.parser')

                fs = list()

                a_tag = soup.findAll('a', href=True)
                for i in a_tag:
                    fs.append(i)
                a = pd.Series(fs).astype(str)
                newsTitle = a[(a.str.contains('class="title"') == True)].drop_duplicates().reset_index()

                authors = list()
                for valueA in range(len(newsTitle[0])):
                    authors.append(newsTitle[0][valueA][
                                   newsTitle[0][valueA].find('data-author=') + 13: newsTitle[0][valueA].find(
                                       'h="') - 2])

                authors = pd.Series(authors)

                titleList = list()
                for value in range(len(newsTitle[0])):

                    if len(newsTitle[0][value][
                           newsTitle[0][value].find('>') + 1: newsTitle[0][value].find('<') - 4]) > 60:
                        titleList.append(
                            newsTitle[0][value][newsTitle[0][value].find('>') + 1: newsTitle[0][value].find('<') - 4])

                titleList = pd.Series(titleList)
                finalDF = pd.concat([titleList, authors], axis=1).set_axis(['Article', 'Author'], axis=1)

                finalDF = finalDF.drop_duplicates(subset='Article').reset_index()
                del [finalDF['index']]

                today = pd.DataFrame(np.full(len(finalDF['Author']), datetime.today().strftime('%Y.%m.%d'))).set_axis(
                    ['Date'], axis=1)

                results.append(pd.concat([today, titleList, authors, pd.Series(np.full(len(titleList), stock)),
                                          pd.Series(np.full(len(titleList), country))],
                                         axis=1).set_axis(['Date', 'Article', 'Author', 'Ticker'], axis=1))

                print('Bing: Article Gathering (1 out of 2) - Ticker:', stock, ' - Progress:',
                      round(iterat / len(stockIndex) * 100), '%')

                clear_output(wait=True)

            finalDF = pd.concat(results).dropna()

        return finalDF

    # Method to get the stock data
    def getStocksData (self):

        import pandas as pd
        import numpy as np
        from datetime import datetime
        import yfinance as yf

        if len(self.stockList) != 0:

            stockIndex = self.stockList

            # set return of last 5 days
            labels = list()
            for i,ticker in enumerate(stockIndex):
                baseS = yf.Ticker(ticker).history('5d')

                if baseS.empty == False:
                    returns = ((pd.DataFrame(baseS['Close']).pct_change().dropna())*100).set_axis(['Returns'], axis = 1).reset_index()
                    returns['Date'] = pd.to_datetime(returns['Date']).dt.strftime('%Y.%m.%d')
                    volumes = ((pd.DataFrame(baseS['Volume']).pct_change().dropna())*100).set_axis(['Volume'], axis = 1).reset_index()
                    volumes['Date_V'] = pd.to_datetime(volumes['Date']).dt.strftime('%Y.%m.%d')
                    del[volumes['Date']]

                    lab = pd.concat([returns, volumes, pd.DataFrame(np.full(len(volumes['Volume']),
                                                ticker)).set_axis(['Ticker'],axis = 1)], axis = 1)
                    del[lab['Date_V']]
                    labels.append(lab)
                    # Progress

                    print('Taking Returns and Volumes (2 out of 2)... Ticker:', ticker, 'Progress:', round((i/len(stockIndex))*100,2), '%')

            labels = pd.concat([df for df in labels], axis = 0).reset_index().dropna()
            del[labels['index']]

            return labels

        else:
            print('No Open Market Found!')
            return 0


    def mergeStockNewsData (self):

        import pandas as pd
        import psycopg2
        from sqlalchemy import create_engine
        from datetime import datetime

        # Merge on Ticker and on date
        sources = ['MarketWatch', 'CNBC', 'Bing']
        news = list()
        for source in sources:
            newsData = self.getSingleStockMarketNews(source)
            news.append(newsData)
        news = pd.concat([series for series in news], axis = 0).reset_index()
        del[news['index']]

        markets = self.getStocksData()

        # Merge the two DataFrames

        total = news.merge(markets, on = ['Date', 'Ticker'])

        # Save to SQL as DailyV2
        file = total

        connection = psycopg2.connect(
            database="News_Data",
            user="postgres",
            password="Davidescemo",
            host="localhost",
            port="5432"
        )

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
        file.to_sql('News_Scraping_DailyV2', engine, if_exists='replace', index=False)
        file.to_sql('News_Scraping_Data_V2', engine, if_exists='replace', index=False)

        connection.close()

        # Daily Download Statistics
        print('\n')
        print('Date:', datetime.today().strftime('%Y.%m.%d'))
        print('News Downloaded:', len(total['Article'].unique()), '- Tickers Affeted:', len(total['Ticker'].unique()))
        print('\n')

        return total

    def updateDataBase (self, dailyNews):

        import pandas as pd
        import yfinance as yf
        import psycopg2
        import random
        from sqlalchemy import create_engine
        from datetime import datetime

        # Import the Database with the old news

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
        query = 'SELECT * FROM public."News_Scraping_Data_V2"'
        baseQuery = pd.read_sql(query, engine)

        # Take the daily news
        dailyNews = dailyNews

        # We don't want to lose any data, for multiple runs, so we add just the news that are not currently present in the
        # entire database.

        dailyNews = dailyNews[dailyNews['Article'].isin(baseQuery['Article']) == False].reset_index()
        del[dailyNews['index']]

        # Update the database of return, and add the new data

        allDf = pd.concat([baseQuery, dailyNews], axis = 0).drop_duplicates(subset=['Article'], keep='first')

        #finalDf = allDf.merge(dailyNews, on = ['Ticker', 'Date'], how = 'left')

        MVNumber = len(dailyNews['Article'][dailyNews['Author'] == 'MarketWatch'].unique())
        CNBCNumber = len(dailyNews['Article'][dailyNews['Author'] == 'CNBC'].unique())
        BINGNumber = len(dailyNews['Article'][(dailyNews['Author'] != 'MarketWatch') & (dailyNews['Author'] != 'CNBC')].unique())

        MVTicker = len(dailyNews['Ticker'][dailyNews['Author'] == 'MarketWatch'].unique())
        CNBCTicker = len(dailyNews['Ticker'][dailyNews['Author'] == 'CNBC'].unique())
        BINGTicker = len(dailyNews['Ticker'][(dailyNews['Author'] != 'MarketWatch') & (dailyNews['Author'] != 'CNBC')].unique())

        print('\n')
        print('-----LAST RUN-----')
        print('News added in the last Run:', len(dailyNews['Article'].unique()))
        print('Bing Search:', BINGNumber, 'News, Tickers:', BINGTicker)
        print('CNBC News:', CNBCNumber, 'News, Tickers:', CNBCTicker)
        print('MarketWatch:', MVNumber, 'News, Tickers:', MVTicker)

        print('\n')

        print('-----TOTAL DATABASE (V2)-----')
        print('Total News in Dataset:', len(allDf['Article'].unique()))
        print('Total Number of Ticker in Dataset:', len(allDf['Ticker'].unique()))
        print('\n')

        # Save the updated database to sql, only if it contains more information than the base one

        if len(baseQuery['Article']) < len(allDf['Article']):

            file = allDf

            connection = psycopg2.connect(
                database="News_Data",
                user="postgres",
                password="Davidescemo",
                host="localhost",
                port="5432"
            )

            engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
            #file.to_sql('News_Scraping_Data_V2', engine, if_exists='replace', index=False)

            connection.close()

        return allDf

    # Once a week, it is needed to Update the returns and the volumes of the stocks (otherwhise the data could not
    # be corrected)

    def updateFinancialData (self):

        import pandas as pd
        import yfinance as yf
        import psycopg2
        import numpy as np
        from sqlalchemy import create_engine

        # Take the base dataset

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
        query = 'SELECT * FROM public."News_Scraping_Data_V2"'
        baseData = pd.read_sql(query, engine)

        # Take the 30 days return, and put them in the same return format of the database

        base30dRet = list()
        base30dVol = list()
        for i, ticker in enumerate(baseData['Ticker'].unique()):
            baseS = yf.Ticker(ticker).history('5d')

            if baseS.empty == False:
                returns = ((pd.DataFrame(baseS['Close']).pct_change().dropna()) * 100).set_axis(['Returns'], axis=1).reset_index()
                returns = pd.concat([returns, pd.DataFrame(np.full(len(returns['Returns']), ticker)).set_axis(['Ticker'], axis = 1)], axis = 1)
                returns['Date'] = pd.to_datetime(returns['Date']).dt.strftime('%Y.%m.%d')

                volumes = ((pd.DataFrame(baseS['Volume']).pct_change().dropna()) * 100).set_axis(['Volume'], axis=1).reset_index()
                volumes = pd.concat([volumes, pd.DataFrame(np.full(len(volumes['Volume']), ticker)).set_axis(['Ticker'], axis = 1)], axis = 1)
                volumes['Date'] = pd.to_datetime(volumes['Date']).dt.strftime('%Y.%m.%d')

                print('Updating Returns and Volumes... Ticker:', ticker, 'Progress:',
                      round((i / len(baseData['Ticker'].unique())) * 100, 2), '%')

                base30dRet.append(returns)
                base30dVol.append(volumes)

        base30dRet = pd.concat([series for series in base30dRet], axis=0)
        base30dVol = pd.concat([series for series in base30dVol], axis=0)

        # Merge everything

        updatedData = baseData.merge(base30dRet, on = ['Date','Ticker'])
        FUpdatedData = updatedData.merge(base30dVol, on = ['Date','Ticker'])

        # Take note of Dataset dimension

        print('Total News in Dataset:', len(FUpdatedData['Article'].unique()))
        print('Total Number of Ticker in Dataset:', len(FUpdatedData['Ticker'].unique()))

        return FUpdatedData










