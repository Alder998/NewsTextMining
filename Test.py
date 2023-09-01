import NewsDownloadLibrary as nl
import NewsAlgorithm as ns
import pandas as pd
from sklearn.model_selection import train_test_split

#print('\n')
#print('-----NEWS PANO-----')
#print('\n')

#pano = nl.getNewsWithIndexPerformance()

#print('\n')
#print('-----NEWS DETAILS-----')
#print('\n')

#massive = nl.MassiveNewsScaper()

textList = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\SingleStockNews1.xlsx")

print(textList)

cleanData = ns.PreProcessing(textList, target='returns', threshold=3).preProcess(POS_tagging=True)

BoWEmbedding = ns.Vectorize(cleanData).Embedding(method = 'Bag-of-Word')

modelSet = ns.Model(BoWEmbedding, testSize=0.20, epochs=6).NNProcessing()

