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

class PreProcessing:
    name = "Text Pre-Processing"

    def __init__(self, textList):
        self.textList = textList
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

        print('Importing data...')
        stockNews = self.textList

        # Encoding dei rendimenti

        stockNews.loc[stockNews['Same-Day Close'] > 0, 'Return_enc'] = 'UP'
        stockNews.loc[stockNews['Same-Day Close'] < 0, 'Return_enc'] = 'DOWN'
        stockNews.loc[stockNews['Same-Day Close'] > 2, 'Return_enc'] = 'STRONG UP'
        stockNews.loc[stockNews['Same-Day Close'] < -2, 'Return_enc'] = 'STRONG DOWN'

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
                    clear_output(wait=True)

            if POS_tagging == True:
                return [preProcSentences, allWords, stocks, POS]
            if POS_tagging == False:
                return [preProcSentences, allWords, stocks]

class Vectorize:
    name = "Text Pre-Processing"

    def __init__(self, processedData):
        self.processedData = processedData
        pass

    def Embedding (self, method = 'Bag-of-Word'):

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

        bowMatrix.loc[bowMatrix['Return_enc'] == 'UP', 'Perf_Encoded'] = 0
        bowMatrix.loc[bowMatrix['Return_enc'] == 'DOWN', 'Perf_Encoded'] = 1
        bowMatrix.loc[bowMatrix['Return_enc'] == 'STRONG UP', 'Perf_Encoded'] = 2
        bowMatrix.loc[bowMatrix['Return_enc'] == 'STRONG DOWN', 'Perf_Encoded'] = 3

        del [bowMatrix['Return_enc']]

        bowMatrix = bowMatrix.dropna()

        return bowMatrix








