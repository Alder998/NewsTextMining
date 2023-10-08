# NLP algorithm with stock market data

import NewsAlgorithm as ns
import pandas as pd
from sqlalchemy import create_engine

# Getting the text data

engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
query = 'SELECT * FROM public."News_Scraping_Data"'
textList = pd.read_sql(query, engine)

#textList = textList[0:500]

# Preprocessing the data: taking english news, removing stop words, taking the words' root
cleanData = ns.PreProcessing(textList, target='returns', classes=4, threshold=1.5).preProcess(POS_tagging=False)

# Data Vectorization: turning text data into a vector, numerically processable by an algorithm
BoWEmbedding = ns.Vectorize(cleanData).Embedding(method = 'Bag-of-Word', vectorSize=30, components = 200)

# Setting the model, compile, train, evaluate the performance on a test set
sample = ns.Sampling(BoWEmbedding, testSize=0.20).TrainTestSplit()

#modelSet = ns.NNModel(sample, epochs=10).NNProcessing(NNType='FF', shapeRec=[64], shape = [200, 200, 200],
#                                                                        activation = 'relu')

modelSet = ns.MLModel(sample).SVCProcessing(kernel = 'rbf')