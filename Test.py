import NewsDownloadLibrary as nl
import NewsAlgorithm as ns
import pandas as pd

#print('\n')
#print('-----NEWS PANO-----')
#print('\n')

#pano = nl.getNewsWithIndexPerformance()

#print('\n')
#print('-----NEWS DETAILS-----')
#print('\n')

#massive = nl.MassiveNewsScaper()

textList = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\completeDataset.xlsx")

cleanData = ns.PreProcessing(textList).preProcess()

BoWEmbedding = ns.Vectorize(cleanData).Embedding(method = 'Bag-of-word')

print(BoWEmbedding)

