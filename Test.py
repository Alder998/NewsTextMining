import NewsDownloadLibrary as nl
import NewsAlgorithm as ns
import pandas as pd
from sklearn.model_selection import train_test_split

textList = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\Velleità\Text Mining & Sentiment Analysis\Stock Market News\finalDataSet\SingleStockNews1.xlsx")

print(textList)

cleanData = ns.PreProcessing(textList, target='returns', threshold=1.5).preProcess(POS_tagging=False)

BoWEmbedding = ns.Vectorize(cleanData).Embedding(method = 'Bag-of-Word', vectorSize=10, components = 200)

modelSet = ns.Model(BoWEmbedding, testSize=0.20, epochs=10).NNProcessing(NNType='FF', shapeRec=[64], shape = [200, 200, 200],
                                                                        activation = 'relu')

