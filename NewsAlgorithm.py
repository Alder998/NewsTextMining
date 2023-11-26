# this is the main file for this project.

# The general aim is to give an Object-Oriented pattern the main project idea: that is to build an algorithm
# that explores the web to get the relevant stock-market related news, and to test different text-mining techniques
# to get relevant info about the market trends.

# Therefore, the project structure needs to be divided in four main parts:
# - Text Mining: scraping techniques combined with the Yahoo Finance library, to grab the relevant news every day,
# And store them into a common file base.
# - Text Pre-Processing: exploring and experimenting different techniques for Text Vectorization and pre-processing
# to transform text pieces into matrices.
# - Modelling: Experiment different supervised and non supervised models to get the best accuracy for text
# Classification
# - Performance Testing: Compare the models' different performances.
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PreProcessing:
    name = "Text Pre-Processing"

    def __init__(self, textList, target, classes, threshold, databaseVersion = 'V1'):
        self.textList = textList
        self.target = target
        self.classes = classes
        self.threshold = threshold
        self.databaseVersion = databaseVersion
        pass

    # Different methods to get different News

    def preProcess (self, POS_tagging = False):

        # Librerie
        import nltk
        import numpy as np
        from langdetect import detect
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from nltk.tokenize import wordpunct_tokenize
        from nltk.stem import WordNetLemmatizer
        import pandas as pd
        import tensorflow as tf
        import numpy as np
        from nltk.stem import PorterStemmer
        from IPython.display import clear_output
        # Models
        import tensorflow as tf
        from sklearn.model_selection import train_test_split

        if self.databaseVersion == 'V1':
            retColumn = 'Same-Day Close'
            volColumn = 'Same-Day Volume'
        if self.databaseVersion == 'V2':
            retColumn = 'Returns'
            volColumn = 'Volume'

        print('Importing data...')
        stockNews = self.textList

        # Encoding dei rendimenti

        if self.classes == 4:
            if self.target == 'returns':
                 stockNews.loc[stockNews[retColumn] > 0, 'Return_enc'] = 'UP'
                 stockNews.loc[stockNews[retColumn] < 0, 'Return_enc'] = 'DOWN'
                 stockNews.loc[stockNews[retColumn] > self.threshold, 'Return_enc'] = 'STRONG UP'
                 stockNews.loc[stockNews[retColumn] < - (self.threshold), 'Return_enc'] = 'STRONG DOWN'

                 del[stockNews[volColumn]]

            if self.target == 'volume':
                 stockNews.loc[stockNews[volColumn] > 0, 'Return_enc'] = 'UP'
                 stockNews.loc[stockNews[volColumn] < 0, 'Return_enc'] = 'DOWN'
                 stockNews.loc[stockNews[volColumn] > self.threshold, 'Return_enc'] = 'STRONG UP'
                 stockNews.loc[stockNews[volColumn] < - (self.threshold), 'Return_enc'] = 'STRONG DOWN'

                 del[stockNews[retColumn]]

        if self.classes == 2:

            if self.target == 'returns':
                stockNews.loc[stockNews[retColumn] > 0, 'Return_enc'] = 'UP'
                stockNews.loc[stockNews[retColumn] < 0, 'Return_enc'] = 'DOWN'

                del [stockNews[volColumn]]

            if self.target == 'volume':
                stockNews.loc[stockNews[volColumn] > 0, 'Return_enc'] = 'UP'
                stockNews.loc[stockNews[volColumn] < 0, 'Return_enc'] = 'DOWN'

                del [stockNews[retColumn]]


        # Filtriamo per le sole notizie in inglese

        langList = list()
        for i, value in enumerate(stockNews['Article']):
            langList.append(detect(value))
            print('Taking English News...', round((i / len(stockNews['Article'])) * 100, 2), '%')
            clear_output(wait=True)

        stockNews = pd.concat([pd.DataFrame(langList).set_axis(['Language'], axis=1), stockNews], axis=1)

        stockNews = stockNews[stockNews['Language'] == 'en'].reset_index()
        del [stockNews['index']]
        del [stockNews['Language']]
        stockNews['Article'] = stockNews['Article'].astype(str)

        preProcSentences = list()
        allWords = list()
        stocks = list()
        POS = list()

        for i, article in enumerate(stockNews['Article']):

            if len(article) > 5:

                # Tokenization: Separiamo ogni singola parola come fosse un token
                tokens = nltk.word_tokenize(article)

                # Adesso bisogna eliminare le STOP WORDS: non sono utili se vogliamo prevedere qualcosa
                # Prendiamo tutte le stop words che ci sono nella lingua inglese
                stop_words = set(stopwords.words("english"))
                punctuation = ['’', '.', ',', ':', ';', '!', '?', '-', '(', ')', ']', '[', '}', '–', '{', "'", '"',
                               "'s", "‘", '_', '-', '“', '”', '...']

                # Lo facciamo in modo  macchinoso: Creiamo una lista, e con un for appendiamo TUTTE LE PAROLE che non sono presenti
                # Nella lista delle stop words in inglese
                filtered_list = list()
                for word in tokens:
                    if word.casefold() not in stop_words:
                        filtered_list.append(word)

                # Eliminiamo la punteggiatura

                filtered_list_np = list()
                for word in filtered_list:
                    if word.casefold() not in punctuation:
                        filtered_list_np.append(word)

                # In questo caso, è forse meglio usare una tecnica di lemmatizing, vale a dire, ricondurre una parola alla sua
                # radice, ma mantenendo il significato sintattico della parola.

                lemmatizer = WordNetLemmatizer()
                lemmatized_sentence = [lemmatizer.lemmatize(word) for word in filtered_list_np]

                # print('Article:', i, 'Processed')

                # print(lemmatized_sentence)
                stokP = stockNews['Return_enc'][stockNews['Article'] == article].reset_index()['Return_enc'][0]

                if len(lemmatized_sentence) > 0:
                    lS = pd.DataFrame(pd.DataFrame(lemmatized_sentence).set_axis(['Tokens'],
                                                                                 axis=1)[
                                          'Tokens'].value_counts()).reset_index()
                    stocks.append(stokP)

                    # Ora applichiamo un POS tagging, vale a dire capire il ruolo sintattico delle parole all'interno di una frase

                    if POS_tagging == True:

                         POS_tag = nltk.pos_tag(lemmatized_sentence)

                         # for syntax in POS_tag:
                         #    POS_SW.append(syntax)

                         POS_S = list()
                         for word in range(0, len(POS_tag)):
                             POS_S.append(pd.DataFrame(POS_tag[word]).T)

                         POS_S = pd.concat([df for df in POS_S], axis=0).set_axis(['word', 'POS'], axis=1).reset_index()
                         del [POS_S['index']]

                         POS_S = POS_S.dropna()

                         # Selezioniamo delle "Proxy" per delineare l'importanza delle parole nel testo

                         POS_S.loc[(POS_S['POS'] != 'VBD') & (POS_S['POS'] != 'VBG') & (POS_S['POS'] != 'VBN') &
                                   (POS_S['POS'] != 'VBP') & (POS_S['POS'] != 'VBZ') & (POS_S['POS'] != 'NN') &
                                   (POS_S['POS'] != 'NNS') & (POS_S['POS'] != 'NNP') & (
                                               POS_S['POS'] != 'NNPS'), 'POS'] = 10e-5

                         POS_S.loc[(POS_S['POS'] == 'NN') | (POS_S['POS'] == 'NNS') | (POS_S['POS'] == 'NNP') |
                                   (POS_S['POS'] == 'NNPS'), 'POS'] = 0.8

                         POS_S.loc[(POS_S['POS'] == 'VBD') | (POS_S['POS'] == 'VBG') | (POS_S['POS'] == 'VBN') |
                                   (POS_S['POS'] == 'VBP') | (POS_S['POS'] == 'VBZ'), 'POS'] = 1

                         POS.append(POS_S)

                    # BAG-OF-WORDS MODEL: Creare una matrice composta di vettori di lunghezza N (=totale parole univoche in ogni
                    # articolo) popolata di un '1' qualora la parola appaia nell'articolo, e 0 altrimenti

                    for word in lemmatized_sentence:
                        allWords.append(word)

                    preProcSentences.append(lS)

                    print('Processing Articles...', round((i / len(stockNews['Article'])) * 100, 2), '%')

        if POS_tagging == True:
            return [preProcSentences, allWords, stocks, POS]
        if POS_tagging == False:
            return [preProcSentences, allWords, stocks]

class Vectorize:
    name = "Text Pre-Processing"

    def __init__(self, processedData):
        self.processedData = processedData
        pass

    def Embedding (self, method = 'Bag-of-Word', vectorSize = 1000, components = 5, getMethod = False):

        if method == 'Bag-of-Word':

            import pandas as pd

            # La lunghezza di ciascun vettore deve essere la lunghezza di allWords

            print('Creating the BoW Matrix...')

            allWords = pd.Series(self.processedData[1]).drop_duplicates().reset_index()
            del [allWords['index']]

            allWords = allWords.set_axis(['count'], axis=1)

            bowMatrix = list()
            for sentenceN in range(0, len(self.processedData[0])):
                f = allWords.merge(self.processedData[0][sentenceN], left_on='count', right_on='Tokens', how='left')
                f = f[['count_x', 'count_y']].set_axis(['word', sentenceN], axis=1).fillna(0)
                bowMatrix.append(f[sentenceN].astype(int))

            bowMatrix = pd.concat([series for series in bowMatrix], axis=1).transpose()

            stocks = pd.DataFrame(self.processedData[2]).set_axis(['Return_enc'], axis=1)
            bowMatrix = pd.concat([bowMatrix, stocks], axis=1)

            print('Number of Articles:', len(self.processedData[0]))
            print(bowMatrix)

            # Ogni riga è un articolo, ogni colonna è la presenza di una parola in quell'articolo
            # Ora possiamo costruire il modello fine a se stesso

            # Encoding delle variabili categoriche

            bowMatrix = bowMatrix.dropna()

            for i, uniqueVar in enumerate(bowMatrix['Return_enc'].unique()):
                bowMatrix.loc[bowMatrix['Return_enc'] == uniqueVar, 'Perf_Encoded'] = i

            del [bowMatrix['Return_enc']]

            bowMatrix = bowMatrix.dropna()

            return bowMatrix

        if method == 'AR-Reduced Bag-of-word':

            import pandas as pd
            import numpy as np

            # La lunghezza di ciascun vettore deve essere la lunghezza di allWords

            print('Creating the BoW Matrix...')

            allWords = pd.Series(self.processedData[1]).drop_duplicates().reset_index()
            del [allWords['index']]

            allWords = allWords.set_axis(['count'], axis=1)

            bowMatrix = list()
            for sentenceN in range(0, len(self.processedData[3])):
                f = allWords.merge(self.processedData[3][sentenceN], left_on='count', right_on='word', how='left')
                f = f[['POS']].set_axis([sentenceN], axis=1).fillna(0)
                bowMatrix.append(f[sentenceN])

            bowMatrix = pd.concat([series for series in bowMatrix], axis=1).dropna().transpose()

            stocks = pd.DataFrame(self.processedData[2]).set_axis(['Return_enc'], axis=1)
            bowMatrix = pd.concat([bowMatrix, stocks], axis=1)

            print('Number of Articles:', len(self.processedData[0]))
            print(bowMatrix)

            # Ogni riga è un articolo, ogni colonna è la presenza di una parola in quell'articolo
            # Ora possiamo costruire il modello fine a se stesso

            # Encoding delle variabili categoriche

            bowMatrix = bowMatrix.dropna()

            for i, uniqueVar in enumerate(bowMatrix['Return_enc'].unique()):
                bowMatrix.loc[bowMatrix['Return_enc'] == uniqueVar, 'Perf_Encoded'] = i

            del [bowMatrix['Return_enc']]

            bowMatrix = bowMatrix.dropna()

            # Metodi per generare la matrice e ridurla

            def AutoregressiveProcess(coefficient=0.5, data_length=10000, prediction_length=50, lags=10):

                import numpy as np
                import statsmodels.api as sm
                from statsmodels.tsa.ar_model import AutoReg
                import pandas as pd

                # AR(1) parameters
                ar_coefficient = coefficient
                data_length = data_length

                # Generate the random noise
                np.random.seed(0)
                error = np.random.normal(loc=0, scale=1, size=data_length)

                # Create an empty array to hold the AR data
                ar_data = np.zeros(data_length)

                # Generate the AR process
                for i in range(1, data_length):
                    ar_data[i] = ar_coefficient * ar_data[i - 1] + error[i]

                # Fit the AR model
                model = AutoReg(ar_data, lags=[lags])
                ar_model = model.fit()

                prediction = np.abs(ar_model.forecast(steps=prediction_length))

                return prediction

            def generateARBoWMatrix(BoWMatrix):

                AREmb = list()
                for column in BoWMatrix.columns:

                    occurence = list(BoWMatrix[(BoWMatrix[column] == 0.8) | (BoWMatrix[column] == 1)].index)

                    if len(occurence) > 0:

                        data_length = len(BoWMatrix[0])

                        occurenceS = [occurence[0]]

                        for value in pd.Series(occurence).astype(float).diff().dropna():
                            occurenceS.append(value)

                        lastEvent = max((data_length) - (np.array(occurenceS).sum()), 0)

                        if lastEvent != 0:
                            occurenceS.append(lastEvent)

                        occurenceS = list(pd.Series(occurenceS).astype(int))

                        # print(occurenceS)

                        l = list()
                        for ev in range(len(occurenceS)):
                            arP = AutoregressiveProcess(coefficient=0.5, prediction_length=occurenceS[ev], lags=3)
                            l.append(pd.DataFrame(arP))

                        l = pd.concat([series for series in l], axis=0).reset_index()

                        del [l['index']]

                        AREmb.append(l)

                        print('Encoding a AR-BoW Matrix: Progress:', round((column / len(BoWMatrix.columns)) * 100, 2),
                              '%')

                AREmb = pd.concat([series for series in AREmb], axis=1).dropna().transpose().reset_index()
                del [AREmb['index']]

                return AREmb

            bowMatrixAR = bowMatrix.loc[:, bowMatrix.columns != 'Perf_Encoded'].transpose()

            ARBoWMatrix = generateARBoWMatrix(bowMatrixAR)

            ARBoWMatrix = pd.concat([ARBoWMatrix, bowMatrix['Perf_Encoded']], axis=1)

            # riduciamo la matrice con una PCA

            # trasporta in formato matrice, ed isola la colonna dei rendimenti

            ARBoWMatrix = ARBoWMatrix.dropna()

            retS = ARBoWMatrix['Perf_Encoded']

            ARBoWMatrix = ARBoWMatrix.loc[:, ARBoWMatrix.columns != 'Perf_Encoded']

            ARBoWMatrix = np.array(ARBoWMatrix)

            # Standardizza i dati
            scaler = StandardScaler()
            ARBoWMatrix_scaled = scaler.fit_transform(ARBoWMatrix)

            # Specifica il numero di componenti desiderate
            n_components = components

            # Applica la PCA
            pca = PCA(n_components=n_components)
            ARBoWMatrix_reduced = pca.fit_transform(ARBoWMatrix_scaled)

            ARBoWMatrix = pd.concat([pd.DataFrame(ARBoWMatrix_reduced), retS], axis=1)

            ARBoWMatrix = ARBoWMatrix.dropna()

            print('\n')

            return ARBoWMatrix

        if method == 'TF-IDF':

            import pandas as pd

            # Here we will try to implement a TF-IDF document: it basically is a formula starting from the Bag of Word Model.
            # Therefore, it is still a so-called "Sparse Embedding method". The formula is:
            # TF(i) = ln(frequency(i, j) / log(N(i))
            # The frequency is the number of times that a word i occurs in the text j, while N(i) is the total length of the
            # text itself. In this wayy, it evaluate more the wrds that Occur the most in a corpus.

            from sklearn.feature_extraction.text import TfidfVectorizer

            # Introduce the Vectorizer
            vectorizer = TfidfVectorizer()

            # Prepare the text
            TFIDFList = list()
            for series in self.processedData[0]:
                TFIDFList.append(list(series['Tokens']))

            # Convert the values to be processed

            TFIDFListj = [' '.join(tokens) for tokens in TFIDFList]

            # Calculate the value of the embedding
            print('Generating the TF-IDF Embedding...')
            tfidf_matrix = vectorizer.fit_transform(TFIDFListj)

            # Obtain the final matrix
            tfidf_embedding = tfidf_matrix.toarray()

            # Put it into Pandas DataFrame Format

            tfidf_embedding = pd.DataFrame(tfidf_embedding)

            # Work on the stocks

            stocks = pd.DataFrame(self.processedData[2]).set_axis(['Return_enc'], axis=1)
            TFIDF = pd.concat([tfidf_embedding, stocks], axis=1)

            # Categorical Variables Encoding

            TFIDF = TFIDF.dropna()

            for i, uniqueVar in enumerate(TFIDF['Return_enc'].unique()):
                TFIDF.loc[TFIDF['Return_enc'] == uniqueVar, 'Perf_Encoded'] = i
            del [TFIDF['Return_enc']]

            TFIDF = TFIDF.dropna()

            return TFIDF

        # Implementation of a BERT Embedding. It is a dense embedding based on Neural Network. It is pre-trained and
        # Based on a Bidirectional recurrent Neural Network
        if method == 'BERT':

            from transformers import BertTokenizer, BertModel
            import torch

            # Take the base-size list of the pre-processed words

            # Convert the values to be processed
            BERTList = list()
            for series in self.processedData[0]:
                BERTList.append(list(series['Tokens']))

            # Load the pre-trained Tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            # Tokenize and convert in tensor with Torch
            token_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in BERTList]
            input_ids = torch.tensor(token_ids)

            # Pass all the tensor created above through the BERT Model
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            # Take only the dimension of the last layer
            last_hidden_states = outputs.last_hidden_state

            # The last layer contains the Embedding values. It is pre-trained, so it has a standard size (768 tokens,
            # with the base BERT)
            print(last_hidden_states)

            return last_hidden_states


        if method == "Word2Vec":

            import pandas as pd
            from gensim.models import Word2Vec
            import numpy as np

            print('Generating the W2V Model...')

            w2VList = list()
            for series in self.processedData[0]:
                w2VList.append(list(series['Tokens']))

            model = Word2Vec(sentences=w2VList, vector_size = 3, window=1, min_count=1, workers=4)

            modelEmbedding = list()
            for sentence in w2VList:
                embedding = model.wv[sentence]
                modelEmbedding.append((embedding))

            # Preparazione del database, padding per adattarsi alla lunghezza variabile

            DataPad = list()
            for sentence in modelEmbedding:
                flat = sentence.flatten()
                DataPad.append(pd.DataFrame(flat).transpose())

            DataPad = pd.concat([series for series in DataPad], axis=0).fillna(0).reset_index()
            del [DataPad['index']]

            stocks = pd.DataFrame(self.processedData[2]).set_axis(['Return_enc'], axis=1)
            W2V = pd.concat([DataPad, stocks], axis=1)

            print('Number of Articles:', len(self.processedData[0]))
            print(W2V)

            # Ogni riga è un articolo, ogni colonna è la presenza di una parola in quell'articolo
            # Ora possiamo costruire il modello fine a se stesso

            # Encoding delle variabili categoriche

            W2V = W2V.dropna()

            for i, uniqueVar in enumerate(W2V['Return_enc'].unique()):
                W2V.loc[W2V['Return_enc'] == uniqueVar, 'Perf_Encoded'] = i

            del [W2V['Return_enc']]

            W2V = W2V.dropna()

            return W2V

class Sampling:
    name = "Sampling class"

    def __init__(self, textEmbedding, testSize):
        self.textEmbedding = textEmbedding
        self.testSize = testSize
        pass

    def TrainTestSplit (self):

        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split

        # dividi in train e test set

        df = train_test_split(self.textEmbedding, test_size=self.testSize, stratify=self.textEmbedding['Perf_Encoded'],
                              random_state=42)

        bowMatrixTrain = df[0]
        bowMatrixTest = df[1]

        print('\n')
        print('Train Size:', len(bowMatrixTrain[0]))
        print('\n')

        # Preparo i dati

        train_set = np.array(bowMatrixTrain.loc[:, bowMatrixTrain.columns != 'Perf_Encoded']).reshape(
            bowMatrixTrain.loc[:, bowMatrixTrain.columns != 'Perf_Encoded'].shape)
        train_labels = np.array(bowMatrixTrain['Perf_Encoded'])

        test_set = np.array(bowMatrixTest.loc[:, bowMatrixTest.columns != 'Perf_Encoded']).reshape(
            bowMatrixTest.loc[:, bowMatrixTest.columns != 'Perf_Encoded'].shape)
        test_labels = np.array(bowMatrixTest['Perf_Encoded'])

        # Statistiche su train e test

        def sampleStats (dataset):
            stats = list()
            for diffRet in dataset['Perf_Encoded'].sort_values().unique():
                st = dataset['Perf_Encoded'][dataset['Perf_Encoded'] == diffRet].count()
                stats.append(st)

            stats = pd.DataFrame(stats)

            # Converti in percentuali sul totale delle osservazioni

            stats = (stats / len(dataset['Perf_Encoded'])) * 100

            stats = (stats.set_index(pd.Series([value for value in dataset['Perf_Encoded'].sort_values().unique()]))).set_axis(['% frequency'], axis=1)

            return stats

        print(sampleStats(bowMatrixTrain))
        print(sampleStats(bowMatrixTest))

        return [train_set, train_labels, test_set, test_labels, bowMatrixTrain, bowMatrixTest]

class NNModel:
    name = "Text Modelling"

    def __init__(self, sampleMatrix, epochs):
        self.epochs = epochs
        self.sampleMatrix = sampleMatrix
        pass

    def NNProcessing (self, NNType = 'FF', shape = [200, 200, 200], shapeRec = [64], activation = 'relu'):

        import pickle
        import os
        import tensorflow as tf
        import numpy as np
        import pandas as pd

        base = self.sampleMatrix

        train_set = base[0]
        train_labels = base[1]
        test_set = base[2]
        test_labels = base[3]

        # Sample Stats Method
        def sampleStats (dataset):
            stats = list()
            for diffRet in dataset['Perf_Encoded'].sort_values().unique():
                st = dataset['Perf_Encoded'][dataset['Perf_Encoded'] == diffRet].count()
                stats.append(st)

            stats = pd.DataFrame(stats)

            # Converti in percentuali sul totale delle osservazioni

            stats = (stats / len(dataset['Perf_Encoded'])) * 100

            stats = (stats.set_index(pd.Series([value for value in dataset['Perf_Encoded'].sort_values().unique()]))).set_axis(['% frequency'], axis=1)

            return stats

        # Just for reporting Purpose

        if len(base[4]['Perf_Encoded'].unique()) == 4:
            ClassDivTrain = (sampleStats(base[4])).set_axis([0], axis = 1).transpose().set_axis(['TrainClass1%', 'TrainClass2%',
                                                                                    'TrainClass3%', 'TrainClass4%'], axis = 1)
            ClassDivTest = (sampleStats(base[5])).set_axis([0], axis = 1).transpose().set_axis(['TestClass1%', 'TestClass2%',
                                                                                    'TestClass3%', 'TestClass4%'], axis = 1)
        elif len(base[4]['Perf_Encoded'].unique()) == 2:
            ClassDivTrain = (sampleStats(base[4])).set_axis([0], axis = 1).transpose().set_axis(['TrainClass1%',
                                                                                                 'TrainClass2%'], axis = 1)
            ClassDivTest = (sampleStats(base[5])).set_axis([0], axis = 1).transpose().set_axis(['TestClass1%',
                                                                                                'TestClass2%'], axis = 1)
        ClassSizes = pd.concat([ClassDivTrain, ClassDivTest], axis = 1)

        if NNType == 'FF':

            # Definizione del modello - proviamo a rendere gli Input settabili dall'utente

            model = tf.keras.Sequential()

            # Definizione dei macroparametri

            # il primo Layer schiaccia la matrice e la rende utilizzabile per la rete: è quindi uguale per tutti

            model.add(tf.keras.layers.Flatten(input_shape=train_set[0].shape))

            # Setta il numero di Layer intermedio

            for sizeDIm in shape:
                model.add(tf.keras.layers.Dense(sizeDIm, activation=activation))

            # Aggiungi il layer SoftMax che rende l'output categorico

            model.add(tf.keras.layers.Dense(len(list(set(train_labels)))))

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            # Fitting del modello

            model.fit(train_set, train_labels, epochs=self.epochs, batch_size=len(list(set(train_labels))))

            # Prediction sui dati di test

            probability_model = tf.keras.Sequential([model,
                                                     tf.keras.layers.Softmax()])

            predictions = probability_model.predict(test_set)

            # Visualizziamo le predictions

            predList = list()
            for value in range(len(predictions)):
                predTest = np.argmax(predictions[value])
                predList.append(predTest)

            predList = pd.Series(predList)

            loss, accuracy = model.evaluate(test_set, test_labels)

            print('\n')
            print('Test Loss:', loss)
            print('Test Accuracy:', accuracy * 100, '%')

            # Saving on a Record Excel file

            dataDictionary = {
                'Date': [datetime.today().strftime('%Y.%m.%d')],
                'Train Size': len(train_set),
                'Test Size': len(test_set),
                'Learning Type': 'Feed-Forward Neural Network',
                'Activation': activation,
                'Layers (recurrent)': 0,
                'Layers': len(shape),
                'Epochs': self.epochs,
                'Test Loss': loss,
                'Test Accuracy': accuracy
            }

            ExcelData = pd.concat([pd.DataFrame(dataDictionary), ClassSizes], axis=1)

            return [predList, ExcelData]

        if NNType == 'recurrent':

            seq_length = 1
            embedded_data_3d = tf.reshape(train_set, (train_set.shape[0], train_set.shape[1], seq_length))

            # Creazione di un modello Recurrent LSTM

            model = tf.keras.Sequential()

            # Definizione dei macroparametri

            # Imposta il numero di Layer ricorrenti
            for counter, sizeDimR in enumerate(shapeRec):
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(sizeDimR, input_shape=(train_set.shape[1], train_set.shape[0]),
                                                                            return_sequences=True)))
                if counter == len(shapeRec) - 1:
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(sizeDimR, input_shape=(train_set.shape[1], train_set.shape[0]),
                                                                            return_sequences=False)))

            # Setta il numero di Layer di tipo FF
            for sizeDim in shape:
                model.add(tf.keras.layers.Dense(sizeDim, activation=activation))

            # Aggiungi il layer SoftMax che rende l'output categorico

            model.add(tf.keras.layers.Dense(len(list(set(train_labels)))))

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            # Fitting del modello
            model.fit(embedded_data_3d, train_labels, epochs=self.epochs,
                      batch_size=len(list(set(train_labels))))

            # Prediction sui dati di test

            probability_model = tf.keras.Sequential([model,
                                                     tf.keras.layers.Softmax()])

            predictions = probability_model.predict(test_set)

            # Visualizziamo le predictions

            predList = list()
            for value in range(len(predictions)):
                predTest = np.argmax(predictions[value])
                predList.append(predTest)

            predList = pd.Series(predList)

            loss, accuracy = model.evaluate(test_set, test_labels)

            # Show result metrics
            print('\n')
            print('Test Loss:', loss)
            print('Test Accuracy:', accuracy * 100, '%')

            # Saving on a Record Excel file

            dataDictionary = {
                'Date': [datetime.today().strftime('%Y.%m.%d')],
                'Train Size': len(train_set),
                'Test Size': len(test_set),
                'Learning Type': 'Recurrent Neural Network',
                'Activation': activation,
                'Layers (recurrent)': len(shapeRec),
                'Layers': len(shape),
                'Epochs': self.epochs,
                'Test Loss': loss,
                'Test Accuracy': accuracy
            }

            ExcelData = pd.concat([pd.DataFrame(dataDictionary), ClassSizes], axis=1)

            return [predList, ExcelData]

class MLModel:
    name = "Text Modelling"

    def __init__(self, sampleMatrix):
        self.sampleMatrix = sampleMatrix
        pass

    def SVCProcessing (self, kernel = 'rbf'):

        from sklearn.svm import SVC
        import pickle
        from sklearn.metrics import accuracy_score
        import numpy as np
        import os
        import pandas as pd

        base = self.sampleMatrix

        # Sample stats method to get the Metrics

        def sampleStats (dataset):
            stats = list()
            for diffRet in dataset['Perf_Encoded'].sort_values().unique():
                st = dataset['Perf_Encoded'][dataset['Perf_Encoded'] == diffRet].count()
                stats.append(st)

            stats = pd.DataFrame(stats)

            # Converti in percentuali sul totale delle osservazioni

            stats = (stats / len(dataset['Perf_Encoded'])) * 100

            stats = (stats.set_index(pd.Series([value for value in dataset['Perf_Encoded'].sort_values().unique()]))).set_axis(['% frequency'], axis=1)

            return stats

        # Just for reporting Purpose
        if len(base[4]['Perf_Encoded'].unique()) == 4:
            ClassDivTrain = (sampleStats(base[4])).set_axis([0], axis = 1).transpose().set_axis(['TrainClass1%', 'TrainClass2%',
                                                                                    'TrainClass3%', 'TrainClass4%'], axis = 1)
            ClassDivTest = (sampleStats(base[5])).set_axis([0], axis = 1).transpose().set_axis(['TestClass1%', 'TestClass2%',
                                                                                    'TestClass3%', 'TestClass4%'], axis = 1)
        if len(base[4]['Perf_Encoded'].unique()) == 2:
            ClassDivTrain = (sampleStats(base[4])).set_axis([0], axis = 1).transpose().set_axis(['TrainClass1%',
                                                                                                 'TrainClass2%'], axis = 1)
            ClassDivTest = (sampleStats(base[5])).set_axis([0], axis = 1).transpose().set_axis(['TestClass1%',
                                                                                                'TestClass2%'], axis = 1)
        ClassSizes = pd.concat([ClassDivTrain, ClassDivTest], axis = 1)

        # Define the train and test set
        train_set = base[0]
        train_labels = base[1]
        test_set = base[2]
        test_labels = base[3]

        SVM = SVC(kernel = kernel)

        print('\n')
        print('Fitting Support Vector Classifier...')

        model = SVM.fit(train_set, train_labels)

        # The prediction is not useful now, since we are basically evaluating the performance
        prediction = model.predict(test_set)

        # Evaluate the accuracy

        accuracy = np.mean(prediction == test_labels)

        print('\n')
        print('SVC % Accuracy:', accuracy*100)

        dataDictionary = {
            'Date': [datetime.today().strftime('%Y.%m.%d')],
            'Train Size': len(train_set),
            'Test Size': len(test_set),
            'Learning Type': 'Support Vector Machine',
            'Activation': 'None',
            'Layers (recurrent)': 'None',
            'Layers': 'None',
            'Epochs': 'None',
            'Test Loss': 'None',
            'Test Accuracy': accuracy
        }

        ExcelData = pd.concat([pd.DataFrame(dataDictionary), ClassSizes], axis=1)

        return [prediction, ExcelData]















