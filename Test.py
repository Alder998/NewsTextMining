# NLP algorithm with stock market data

import NewsAlgorithm as ns
import pandas as pd
from sqlalchemy import create_engine

# Getting the text data

engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
query = 'SELECT * FROM public."News_Scraping_Data_V2"'
textList = pd.read_sql(query, engine)

textList = textList[0:1000]

# Define the outer parameters
target='returns'
classes=4
databaseVersion='V2'

# Preprocessing the data: taking english news, removing stop words, taking the words' root
cleanData = ns.PreProcessing(textList, target=target, classes=classes, threshold=1, databaseVersion=databaseVersion).preProcess(POS_tagging=False)

# Data Vectorization: turning text data into a vector, numerically processable by an algorithm
method = 'Word2Vec'
BoWEmbedding = ns.Vectorize(cleanData).Embedding(method = method)

# Setting the model, compile, train, evaluate the performance on a test set
sample = ns.Sampling(BoWEmbedding, testSize=0.15).TrainTestSplit()

modelSet = ns.NNModel(sample, epochs=10).NNProcessing(NNType='FF', shapeRec=[128], shape = [200, 200, 200],
                                                            activation = 'relu')

#modelSet = ns.MLModel(sample).SVCProcessing(kernel = 'rbf')

# Integrate the DB and save

outerDB = {
    'Target': [target],
    'classes': [classes],
    'DatabaseVersion': [databaseVersion],
    'Embedding': [method]
}

#baseData = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\News Model Results\News Model Results.xlsx")
resultDB = pd.concat([pd.DataFrame(outerDB), modelSet[1]], axis = 1)
#finalDB = pd.concat([baseData, resultDB], axis = 0)
resultDB.to_excel(r"C:\Users\39328\OneDrive\Desktop\News Model Results\News Model Results.xlsx", index=False)

