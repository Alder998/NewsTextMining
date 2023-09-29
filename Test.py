# NLP algorithm with stock market data

import NewsAlgorithm as ns
import pandas as pd
from sqlalchemy import create_engine

# Getting the text data

engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
query = 'SELECT * FROM public."News_Scraping_Data"'
textList = pd.read_sql(query, engine)

# Preprocessing the data: taking english news, removing stop words, taking the words' root
cleanData = ns.PreProcessing(textList, target='returns', classes=4, threshold=1.5).preProcess(POS_tagging=False)

# Data Vectorization: turning text data into a vector, numerically processable by an algorithm
BoWEmbedding = ns.Vectorize(cleanData).Embedding(method = 'Word2Vec', vectorSize=30, components = 200)

# Setting the model, compile, train, evaluate the performance on a test set
modelSet = ns.Model(BoWEmbedding, testSize=0.20, epochs=5).NNProcessing(NNType='recurrent', shapeRec=[64], shape = [200],
                                                                        activation = 'relu')

