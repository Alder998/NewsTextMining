# Scarico notizie stock mirati

def getSingleStockMarketNews(stockIndex):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import yfinance as yf
    import numpy as np
    from datetime import datetime
    import random
    import os
    from IPython.display import clear_output

    results = list()
    for iterat, stock in enumerate(stockIndex):

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
                           newsTitle[0][valueA].find('data-author=') + 13: newsTitle[0][valueA].find('h="') - 2])

        authors = pd.Series(authors)

        titleList = list()
        for value in range(len(newsTitle[0])):

            if len(newsTitle[0][value][newsTitle[0][value].find('>') + 1: newsTitle[0][value].find('<') - 4]) > 60:
                titleList.append(
                    newsTitle[0][value][newsTitle[0][value].find('>') + 1: newsTitle[0][value].find('<') - 4])

        titleList = pd.Series(titleList)
        finalDF = pd.concat([titleList, authors], axis=1).set_axis(['Article', 'Author'], axis=1)

        finalDF = finalDF.drop_duplicates(subset='Article').reset_index()
        del [finalDF['index']]

        today = pd.DataFrame(np.full(len(finalDF['Author']), datetime.today().strftime('%Y.%m.%d'))).set_axis(['Date'],
                                                                                                              axis=1)

        results.append(pd.concat([today, titleList, authors, pd.Series(np.full(len(titleList), stock))],
                                 axis=1).set_axis(['Date', 'Article', 'Author', 'Stock'], axis=1))

        print('Article Gathering (1 out of 2) - Ticker:', stock, ' - Progress:',
              round(iterat / len(stockIndex) * 100), '%')

        clear_output(wait=True)

    finalDF = pd.concat(results).dropna()

    # print(finalDF)

    # Prendiamo il rendimento dello stock mirato

    marketTrend = list()
    for iterat, ticker in enumerate(stockIndex):

        if yf.Ticker(ticker).history('1d').empty == False:
            history = (yf.Ticker(ticker).history('7d')['Close'].pct_change() * 100).reset_index()

            retHist = history['Close'][len(history['Close']) - 1]
            dateHist = pd.to_datetime(history['Date'][len(history['Close']) - 1]).strftime('%Y.%m.%d')

            mt = pd.concat([pd.Series(retHist), pd.Series(ticker)], axis=1).set_axis(['Return', 'Stock'], axis=1)

            print('Stock Performance Gathering (2 out of 2) - Ticker:', ticker, ' - Progress:',
                  round(iterat / len(stockIndex) * 100), '%')

            marketTrend.append(mt)

        clear_output(wait=True)

    marketTrend = pd.concat([series for series in marketTrend], axis=0)

    SingleStockData = finalDF.merge(marketTrend, on='Stock')

    return SingleStockData


def MassiveNewsScaper (numberOfRandomStocks=50):
    import pandas as pd
    from IPython.display import clear_output
    import time
    import random

    start = time.time()

    stockList = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\S&P500 Constituents.xlsx")['Ticker']

    # Esperimento: prendere ogni giorno 30 stocks a random in un sample IMMENSO con tutte i ticker di Yahoo finance

    # Importa i dati
    allStocks = pd.read_excel(r"D:\Credit Rating Data\All Yahoo Finance Stock Tickers.xlsx").dropna()
    allStocks = allStocks[allStocks['Country'] == 'USA'].reset_index()

    # Seleziona 50 Ticker in modo random

    AtickerNumber = numberOfRandomStocks

    alternativeTicker = list()
    for i in range(0, AtickerNumber):
        alternativeTicker.append(random.choice(allStocks['Ticker']))
    alternativeTicker = pd.DataFrame(alternativeTicker).set_axis([0], axis=1)

    stockList = pd.concat([stockList, alternativeTicker], axis=0).reset_index()
    del [stockList['index']]

    today = getSingleStockMarketNews(stockList[0])

    base = pd.read_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\SingleStockNews.xlsx")

    total = pd.concat([base, today], axis=0).drop_duplicates(subset='Article')
    total.to_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\SingleStockNews.xlsx",
        index=False)

    print('\n')
    print('DOWNLOAD HIGHLIGHTS')
    print('Total News:', len(today['Date']))
    print('Number of stock Analyzed:', len(today['Stock'].unique()))

    print('\n')
    print('COMPLETE DATASET')
    print('Total News:', len(total['Date']))
    print('Number of stock Analyzed:', len(total['Stock'].unique()))
    print('Net number of news added:', len(total) - len(base))

    print('\n')
    end = time.time()
    print('Data Gathered in:', round(((end - start) / 60), 2), 'Minutes')

    return total


def getNewsWithIndexPerformance():
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import os
    from IPython.display import clear_output
    import yfinance as yf

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    # Mettiamo su l'URL di base: la ricerca di Google News sul Mercato finanziario

    results = []
    for page in range(1, 100):
        target_url = f"https://www.bing.com/news/search?q=stock+market+news&first={(page - 1) * 8 + 1}&FORM=PERE"
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
            authors.append(
                newsTitle[0][valueA][
                newsTitle[0][valueA].find('data-author=') + 13: newsTitle[0][valueA].find('h="') - 2])

        authors = pd.Series(authors)

        titleList = list()
        for value in range(len(newsTitle[0])):
            titleList.append(newsTitle[0][value][newsTitle[0][value].find('>') + 1: newsTitle[0][value].find('<') - 4])

        titleList = pd.Series(titleList)
        results.append(pd.concat([titleList, authors], axis=1).set_axis(['Article', 'Author'], axis=1))

        print('Page', page, 'Analyzed')
        clear_output(wait=True)

    finalDF = pd.concat(results)

    finalDF = finalDF.drop_duplicates(subset='Article').reset_index()
    del [finalDF['index']]

    today = pd.DataFrame(np.full(len(finalDF['Author']), datetime.today().strftime('%Y.%m.%d'))).set_axis(['Date'],
                                                                                                          axis=1)

    finalDF = pd.concat([finalDF, today], axis=1)

    # print(finalDF)

    finalDF.to_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\Raw News\MarketNews - " + datetime.today().strftime(
            '%Y.%m.%d') + ".xlsx",
        index=False)

    # Update the final Dataset

    path = r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\Raw News"
    fileList = os.listdir(path)

    eachFile = list()
    for i in fileList:
        file = pd.read_excel(path + '/' + i)
        eachFile.append(file)

    finalData = pd.concat([series for series in eachFile])

    # Salva senza mercati
    finalData.to_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\finalDataset.xlsx")

    # Aggiungi perfromance mercati

    dataWithMarkets = pd.read_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\completeDataset.xlsx")

    # I returns vanno presi da un punto di partenza, che non è il giorno di oggi, ma deve essere preso dalla serie storica

    # dataWithMarkets['Date'] = dataWithMarkets['Unnamed: 0']
    # del[dataWithMarkets['Unnamed: 0']]

    dataWithMarkets['Date'] = pd.to_datetime(dataWithMarkets['Date'])
    dataWithMarkets = dataWithMarkets.set_index(dataWithMarkets['Date'])

    lastFiveDays = pd.DataFrame(dataWithMarkets['Date'].unique()[
                                len(dataWithMarkets['Date'].unique()) - 5: len(
                                    dataWithMarkets['Date'].unique())]).set_axis(
        ['Date'],
        axis=1)
    lastFiveDays['Date'] = pd.to_datetime(lastFiveDays['Date'])

    # Abbiamo isolato gli ultimi giorni. Adesso possiamo prendere la DATA DI OGGI da FinalDF

    today = finalDF['Date'].unique()[0].replace('.', '-')

    # Scarichiamo il rendimento dummy degli stock

    stockList = ['^GSPC', '^IXIC', '^RUT', 'FTSEMIB.MI', '^FTSE',
                 '^FCHI', '^IBEX', '^STOXX50E', '^STOXX', '000001.SS',
                 '^N225', '^BSESN', 'CL=F',
                 'NG=F', 'GC=F', 'SI=F', 'HG=F']

    directions = list()
    for t in stockList:
        ticker = t

        stock = (yf.Ticker(ticker).history(start=lastFiveDays['Date'][0], end=today,
                                           interval='1d')['Close'].pct_change() * 100).dropna()
        stock = pd.DataFrame(stock)
        stock.loc[stock['Close'] > 0, ticker] = 'UP'
        stock.loc[stock['Close'] < 0, ticker] = 'DOWN'

        date = pd.Series(stock.index)
        date = date.dt.tz_localize(None)
        date = pd.DataFrame(date).set_axis(['Date'], axis=1)

        stock = stock.set_index(date['Date'])

        directions.append(stock[ticker])

    directions = pd.concat([series for series in directions], axis=1)

    # Cancella la data (colonna duplicata) dal dataset base

    finalData['Date'] = pd.to_datetime(finalData['Date'])

    finalSet = finalData[finalData['Date'].isin(pd.Series(directions.index))].merge(directions, on='Date')

    completeSet = pd.concat([dataWithMarkets, finalSet], axis=0)
    completeSet = completeSet.drop_duplicates().set_index('Date')

    # del[completeSet['Unnamed: 0']]
    # del[completeSet['Date']]

    print(completeSet)

    # Mettiamo su un Dataset con i mercati
    completeSet.to_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\completeDataset.xlsx")

    # Riassunto della creazione del dataset

    print('\n')
    print('DOWNLOAD HIGHLIGHTS')

    lastObsMkt = (completeSet.index[len(completeSet) - 1])
    finalData = finalData.reset_index()
    del [finalData['index']]
    lastObsNoMkt = (finalData['Date'][len(finalData) - 1])

    print('Last date registered (without Markets):', datetime.date(lastObsNoMkt), ', News Added:', len(finalDF['Date']))
    print('Last date registered (with Markets):', datetime.date(lastObsMkt), ', News Added:',
          len(completeSet[completeSet.index == lastObsMkt]))

    print('\n')
    print('COMPLETE DATASET:')
    print('Dataset size without Market performance:', len(finalData['Article']))
    print('Dataset size with Market performance:', len(completeSet['Article']))

