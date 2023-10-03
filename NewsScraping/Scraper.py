# The proper scraper class: given an index of tickers, it will go on the internet and get the relevant news

class Scraper:
    name = "Check Markets conditions"

    def __init__(self):
        pass

    # Scarico notizie stock mirati

    def getSingleStockMarketNews(self, source='Bing'):

        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        import numpy as np
        from datetime import datetime
        from IPython.display import clear_output
        from NewsScraping.Markets import Markets

        # take the stock index
        stockIndex = Markets().getStockIndex()

        if source == 'CNBC':

            results = list()
            for iterat, stock in enumerate(stockIndex):

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

                # Mettiamo su l'URL di base: la ricerca di Google News sul Mercato finanziario

                target_url = "https://www.cnbc.com/quotes/" + stock + "?qsearchterm=" + stock

                resp = requests.get(target_url, headers=headers)
                soup = BeautifulSoup(resp.text, 'html.parser')  # Pare che l'URL sia libero per fare scraping

                a_tag = soup.findAll('a')

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
                                          pd.DataFrame(np.full(len(cnbsNews), stock)).set_axis(['Stock'], axis=1)],
                                         axis=1)

                    results.append(cnbsNews)

                    print('CNBC: Article Gathering (1 out of 2) - Ticker:', stock, ' - Progress:',
                          round(iterat / len(stockIndex) * 100), '%')

                    clear_output(wait=True)

            finalDF = pd.concat(results).dropna()
            finalDF = finalDF.drop_duplicates(subset='Article')

        if source == 'MarketWatch':

            results = list()
            for iterat, stock in enumerate(stockIndex):

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

                # Mettiamo su l'URL di base: la ricerca di Google News sul Mercato finanziario

                target_url = "https://www.marketwatch.com/investing/stock/" + stock

                resp = requests.get(target_url, headers=headers)
                soup = BeautifulSoup(resp.text, 'html.parser')  # Pare che l'URL sia libero per fare scraping

                a_tag = soup.findAll('a')

                fs = list()
                for i in a_tag:
                    fs.append(i)

                a = pd.Series(fs).astype(str)

                a.to_excel(r"C:\Users\39328\OneDrive\Desktop\zio pino indice1.xlsx")

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
                                          pd.DataFrame(np.full(len(cnbsNews), stock)).set_axis(['Stock'], axis=1)],
                                         axis=1)

                    results.append(cnbsNews)

                    print('MarketWatch: Article Gathering (1 out of 2) - Ticker:', stock, ' - Progress:',
                          round(iterat / len(stockIndex) * 100), '%')

                    clear_output(wait=True)

                finalDF = pd.concat(results).dropna()
                finalDF = finalDF.drop_duplicates(subset='Article')

        if source == 'Bing':

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

                results.append(pd.concat([today, titleList, authors, pd.Series(np.full(len(titleList), stock))],
                                         axis=1).set_axis(['Date', 'Article', 'Author', 'Stock'], axis=1))

                print('Bing: Article Gathering (1 out of 2) - Ticker:', stock, ' - Progress:',
                      round(iterat / len(stockIndex) * 100), '%')

                clear_output(wait=True)

            finalDF = pd.concat(results).dropna()

        return finalDF