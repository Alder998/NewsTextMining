# NLP algorithm with stock market data

import NewsAlgorithm as ns
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Getting the text data

engine = create_engine('postgresql://postgres:Davidescemo@localhost:5432/News_Data')
query = 'SELECT * FROM public."News_Scraping_Data_V2"'
textList = pd.read_sql(query, engine)

#textList = textList[0:500]

# Define all the Parameters needed

# Define the outer parameters - as inputs to be processed
target = input('Choose target value [returns / volume]:')
classes = int(input('Choose the Number of classes [2 / 4]:'))
databaseVersion = 'V2'
embeddingType = int(input('Choose the Embedding from [Bag-of-Word:0, TF-IDF:1, Word2Vec:2]:'))
embeddingList = ['Bag-of-Word', 'TF-IDF', 'Word2Vec']
method = embeddingList[embeddingType]
print('Embedding Chosen:', method)
modelType = input('Choose the Model Type [NN / ML]:')

# Preprocessing the data: taking english news, removing stop words, taking the words' root
if target == 'volume':
    t=20
if target == 'returns':
    t=1

if modelType == 'NN':
    # Set parameters
    epochs = int(input('Choose the number of Epochs:'))
    typeNN = input('Choose the NN Type [FF / recurrent]:')
    if typeNN == 'recurrent':
        recL = int(input('choose the number of recurrent layers:'))
        shapeRec = list(np.full(recL, 32))
    else:
        shapeRec = None
    FFL = int(input('choose the number of FF layers:'))
    shape = list(np.full(FFL, 200))
    act = input('Choose the Activation [relu / tanh]:')

    # Run the Algorithm

    cleanData = ns.PreProcessing(textList, target=target, classes=classes, threshold=t,
                                 databaseVersion=databaseVersion).preProcess(POS_tagging=False)

    # Data Vectorization: turning text data into a vector, numerically processable by an algorithm
    BoWEmbedding = ns.Vectorize(cleanData).Embedding(method=method)

    # Setting the model, compile, train, evaluate the performance on a test set
    sample = ns.Sampling(BoWEmbedding, testSize=0.15).TrainTestSplit()

    modelSet = ns.NNModel(sample, epochs=epochs).NNProcessing(NNType=typeNN, shapeRec=shapeRec, shape = shape,
                                                            activation = act)

if modelType == 'ML':

    # Run the Algorithm

    cleanData = ns.PreProcessing(textList, target=target, classes=classes, threshold=t,
                                 databaseVersion=databaseVersion).preProcess(POS_tagging=False)

    # Data Vectorization: turning text data into a vector, numerically processable by an algorithm
    BoWEmbedding = ns.Vectorize(cleanData).Embedding(method=method)

    # Setting the model, compile, train, evaluate the performance on a test set
    sample = ns.Sampling(BoWEmbedding, testSize=0.15).TrainTestSplit()

    modelSet = ns.MLModel(sample).SVCProcessing(kernel = 'rbf')

# Integrate the DB and save

outerDB = {
    'Target': [target],
    'classes': [classes],
    'DatabaseVersion': [databaseVersion],
    'Embedding': [method]
}

baseData = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\News Model Results\News Model Results.xlsx")
resultDB = pd.concat([pd.DataFrame(outerDB), modelSet[1]], axis = 1)
finalDB = pd.concat([baseData, resultDB], axis = 0)
finalDB.to_excel(r"C:\Users\39328\OneDrive\Desktop\News Model Results\News Model Results.xlsx", index=False)

