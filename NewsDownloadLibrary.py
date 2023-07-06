# Scarico notizie stock mirati

def getSingleStockMarketNews(stockIndex):

    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import yfinance as yf
    import numpy as np
    from datetime import datetime
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


def MassiveNewsScaper():

    import pandas as pd
    from IPython.display import clear_output
    import time

    start = time.time()

    stockList = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\S&P500 Constituents.xlsx")['Ticker']

    today = getSingleStockMarketNews(stockList)

    base = pd.read_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\SingleStockNews.xlsx")

    total = pd.concat([base, today], axis=0).drop_duplicates(subset='Article')
    total.to_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\SingleStockNews.xlsx")

    print('\n')
    print('DOWNLOAD HIGHLIGHTS')
    print('Total News:', len(today['Date']))
    print('Number of stock Analyzed:', len(today['Stock'].unique()))

    print('\n')
    print('COMPLETE DATASET')
    print('Total News:', len(total['Date']))
    print('Number of stock Analyzed:', len(total['Stock'].unique()))

    print('\n')
    end = time.time()
    print('Data Gathered in:', round(((end - start) / 60), 2), 'Minutes')

    return total