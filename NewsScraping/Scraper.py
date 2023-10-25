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
            for iterat, stockBase in enumerate(stockIndex):

                # Track back the market the stock is from
                country = allStocks['Country'][allStocks['Ticker'] == stockBase].reset_index()['Country'][0]
                countryCode = allStocks['CountryCode'][allStocks['Ticker'] == stockBase].reset_index()['CountryCode'][0]

                # Prepare the Ticker to be processes
                if country == 'USA':
                    target_url = "https://www.cnbc.com/quotes/" + stockBase + "?qsearchterm=" + stockBase
                else:
                    if '.' in stockBase:
                        stock = stockBase[:stockBase.find('.')]
                    else:
                        stock = stockBase
                    target_url = "https://www.cnbc.com/quotes/" + stock + '-' + str(countryCode) + "?qsearchterm=" + stock + '-' + str(countryCode)

                print('CNBC: Article Gathering (1 out of 2) - Ticker:', stockBase, ' - Progress:',
                      round(iterat / len(stockIndex) * 100), '%')

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

                # Mettiamo su l'URL di base: la ricerca di Google News sul Mercato finanziario

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
                                          pd.DataFrame(np.full(len(cnbsNews), stockBase)).set_axis(['Ticker'], axis=1),
                                          pd.DataFrame(np.full(len(cnbsNews), country)).set_axis(['Country'], axis = 1)],
                                         axis=1)

                    print('Found:', len(cnbsNews['Article'].unique()), 'News')

                    results.append(cnbsNews)

                else:
                    print('No News found for:', stockBase, 'on CNBC')

            if len(results) != 0:
                finalDF = pd.concat(results).dropna()
                finalDF = finalDF.drop_duplicates(subset='Article')
            else:
                finalDF = pd.DataFrame([])

        if source == 'MarketWatch':

            results = list()
            for iterat, stockBase in enumerate(stockIndex):

                # Track back the market the stock is from
                country = allStocks['Country'][allStocks['Ticker'] == stockBase].reset_index()['Country'][0]
                countryCode = allStocks['CountryCode'][allStocks['Ticker'] == stockBase].reset_index()['CountryCode'][0]

                print('MarketWatch: Article Gathering (1 out of 2) - Ticker:', stockBase, ' - Progress:',
                      round(iterat / len(stockIndex) * 100), '%')

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

                # change the target URL According to the market index

                if country == 'USA':
                    target_url = "https://www.marketwatch.com/investing/stock/" + stockBase
                else:

                    # prepare the ticker to be processed
                    if '.' in stockBase:
                        stock = stockBase[:stockBase.find('.')]
                    else:
                        stock = stockBase

                    target_url = "https://www.marketwatch.com/investing/stock/" + stock + "?countryCode=" + str(countryCode)

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
                                          pd.DataFrame(np.full(len(cnbsNews), stockBase)).set_axis(['Ticker'], axis=1),
                                          pd.DataFrame(np.full(len(cnbsNews), country)).set_axis(['Country'], axis = 1)],
                                         axis=1)

                    print('Found:', len(cnbsNews['Article'].unique()), 'News')

                    results.append(cnbsNews)

                else:
                    print('No News found for:', stockBase, 'on MarketWatch')

            if len(results) != 0:
                finalDF = pd.concat(results).dropna()
                finalDF = finalDF.drop_duplicates(subset='Article')
            else:
                finalDF = pd.DataFrame([])

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
                                         axis=1).set_axis(['Date', 'Article', 'Author', 'Ticker', 'Country'], axis=1))

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


    def MassiveScraper (self):

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
        #file.to_sql('News_Scraping_Data_V2', engine, if_exists='replace', index=False)

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
        CNBCNumber = len(dailyNews['Article'][dailyNews['Author'] == 'CNBC News'].unique())
        BINGNumber = len(dailyNews['Article'][(dailyNews['Author'] != 'MarketWatch') & (dailyNews['Author'] != 'CNBC News')].unique())

        MVTicker = len(dailyNews['Ticker'][dailyNews['Author'] == 'MarketWatch'].unique())
        CNBCTicker = len(dailyNews['Ticker'][dailyNews['Author'] == 'CNBC News'].unique())
        BINGTicker = len(dailyNews['Ticker'][(dailyNews['Author'] != 'MarketWatch') & (dailyNews['Author'] != 'CNBC News')].unique())

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

        #if len(baseQuery['Article']) < len(allDf['Article']):

        file = allDf

        connection = psycopg2.connect(
                database="News_Data",
                user="postgres",
                password="Davidescemo",
                host="localhost",
                port="5432"
        )

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
        file.to_sql('News_Scraping_Data_V2', engine, if_exists='replace', index=False)

        connection.close()

        allDf = allDf.reset_index()
        del[allDf['index']]

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
            baseS = yf.Ticker(ticker).history('30d')

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

        updatedData = baseData.merge(base30dRet, on=['Date', 'Ticker'], how='left')
        FUpdatedData = updatedData.merge(base30dVol, on=['Date', 'Ticker'], how='left')

        # Take note of Dataset dimension

        print('Total News in Dataset:', len(FUpdatedData['Article'].unique()))
        print('Total Number of Ticker in Dataset:', len(FUpdatedData['Ticker'].unique()))

        # Where the return and volumes of the columns "Return_y" is not NaN, substitute it. Else, take the old one

        FUpdatedData.loc[FUpdatedData['Returns_y'].isna() == False, 'Returns'] = FUpdatedData['Returns_y']
        FUpdatedData.loc[FUpdatedData['Returns_y'].isna() == True, 'Returns'] = FUpdatedData['Returns_x']

        FUpdatedData.loc[FUpdatedData['Volume_y'].isna() == False, 'Volume'] = FUpdatedData['Volume_y']
        FUpdatedData.loc[FUpdatedData['Volume_y'].isna() == True, 'Volume'] = FUpdatedData['Volume_x']

        # Drop the columns produced by the merge method

        del [FUpdatedData['Volume_x']]
        del [FUpdatedData['Volume_y']]
        del [FUpdatedData['Returns_y']]
        del [FUpdatedData['Returns_x']]

        # Save the Database

        file = FUpdatedData

        connection = psycopg2.connect(
            database="News_Data",
            user="postgres",
            password="Davidescemo",
            host="localhost",
            port="5432"
        )

        engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
        file.to_sql('News_Scraping_Data_V2', engine, if_exists='replace', index=False)

        connection.close()

        return FUpdatedData

    def generateStatistics (self, database = 'total', expand = 'None'):

        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from datetime import datetime
        from sqlalchemy import create_engine

        if database == 'total':
            engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
            query = 'SELECT * FROM public."News_Scraping_Data_V2"'
            data = pd.read_sql(query, engine)
            title = 'TOTAL DATABASE'

        if database == 'today':
            engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
            query = 'SELECT * FROM public."News_Scraping_Data_V2"'
            data = pd.read_sql(query, engine)
            data = data[data['Date'] == datetime.today().strftime('%Y.%m.%d')]
            title = 'TODAY'

        # Distribuzione per classi

        plt.figure(figsize=(12, 12))
        plt.suptitle(title, fontsize=16)

        # Source

        # Preparo i dati
        data.loc[data['Author'].str.contains('CNBC'), 'Source'] = 'CNBC News'
        data.loc[data['Author'].str.contains('MarketWatch'), 'Source'] = 'MarketWatch'
        data.loc[(~data['Author'].str.contains('CNBC')) & (
            ~data['Author'].str.contains('MarketWatch')), 'Source'] = 'Bing Search'

        pieSource = data.groupby('Source').count()['Article'].transpose()

        plt.subplot(2, 2, 1)
        plt.pie(pieSource, labels=pieSource.index, autopct='%1.1f%%')
        plt.title('Source')

        # Country

        # Preparo i dati
        if 'Europe' in expand:
            data = data
        else:
            data.loc[(data['Country'] == 'Italy') | (data['Country'] == 'France') | (data['Country'] == 'Germany') |
                     (data['Country'] == 'Spain') | (data['Country'] == 'Netherlands') | (data['Country'] == 'Greece') |
                     (data['Country'] == 'Denmark') | (data['Country'] == 'Belgium') | (
                             data['Country'] == 'United Kingdom') |
                     (data['Country'] == 'Norway') | (data['Country'] == 'Sweden'),
                     'Area'] = 'Europe'

        if 'Asia' in expand:
            data = data
        else:
            data.loc[(data['Country'] == 'India') | (data['Country'] == 'Hong Kong') |
                     (data['Country'] == 'Singapore'), 'Area'] = 'Asia'

        data.loc[(data['Country'] == 'Mexico') | (data['Country'] == 'Brazil'), 'Area'] = 'South America'
        data.loc[(data['Country'] == 'USA') | (data['Country'] == 'Canada'), 'Area'] = 'USA and Canada'

        pieCountry = data.groupby('Area').count()['Article'].transpose()

        plt.subplot(2, 2, 2)
        plt.pie(pieCountry, labels=pieCountry.index, autopct='%1.1f%%')
        plt.title('Geographic Area')

        # Rendimenti (threshold: 1.5%)

        # Preparo i dati

        data.loc[data['Returns'] > 0, 'Returns_class'] = 'UP'
        data.loc[data['Returns'] < 0, 'Returns_class'] = 'DOWN'
        data.loc[data['Returns'] > 1.5, 'Returns_class'] = 'STRONG UP'
        data.loc[data['Returns'] < -1.5, 'Returns_class'] = 'STRONG DOWN'

        pieClassR = data.groupby('Returns_class').count()['Article'].transpose()

        plt.subplot(2, 2, 3)
        plt.pie(pieClassR, labels=pieClassR.index, autopct='%1.1f%%')
        plt.title('Returns')

        # Volume (threshold: 20%)

        # Preparo i dati

        data.loc[data['Volume'] > 0, 'Volume_class'] = 'UP'
        data.loc[data['Volume'] < 0, 'Volume_class'] = 'DOWN'
        data.loc[data['Volume'] > 20, 'Volume_class'] = 'STRONG UP'
        data.loc[data['Volume'] < -20, 'Volume_class'] = 'STRONG DOWN'

        pieClassV = data.groupby('Volume_class').count()['Article'].transpose()

        plt.subplot(2, 2, 4)
        plt.pie(pieClassV, labels=pieClassR.index, autopct='%1.1f%%')
        plt.title('Volume')

        plt.show()








