import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import os
from IPython.display import clear_output

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
            newsTitle[0][valueA][newsTitle[0][valueA].find('data-author=') + 13: newsTitle[0][valueA].find('h="') - 2])

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

today = pd.DataFrame(np.full(len(finalDF['Author']), datetime.today().strftime('%Y.%m.%d'))).set_axis(['Date'], axis=1)

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
                            len(dataWithMarkets['Date'].unique()) - 5: len(dataWithMarkets['Date'].unique())]).set_axis(
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