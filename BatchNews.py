# NLP algorithm with stock market data

import NewsAlgorithm as ns
import pandas as pd
from sqlalchemy import create_engine

# Getting the text data

engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
query = 'SELECT * FROM public."News_Scraping_Data_V2"'
textList = pd.read_sql(query, engine)

#textList = textList[0:1000]

# Take the Inputs

target = input('Select the Target [returns / volumes]:')

# Preprocessing the data: taking english news, removing stop words, taking the words' root
cleanData = ns.PreProcessing(textList, target=target, classes=2, threshold=1, databaseVersion='V2').preProcess(POS_tagging=False)

# Data Vectorization: turning text data into a vector, numerically processable by an algorithm
method = 'Word2Vec'
BoWEmbedding = ns.Vectorize(cleanData).Embedding(method = method)

# Setting the model, compile, train, evaluate the performance on a test set
sample = ns.Sampling(BoWEmbedding, testSize=0.15).TrainTestSplit()

#modelSet = ns.NNModel(sample, epochs=10).NNProcessing(NNType='FF', shapeRec=[128], shape = [200, 200, 200],
#                                                            activation = 'relu', reportEmbedding = method)

modelSet = ns.MLModel(sample).SVCProcessing(kernel = 'rbf', reportEmbedding = method)