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

    def preProcess (self):

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

        # Importiamo i dati ed eliminiamo gli NaN
        print('Importing data...')
        stockNews = self.textList.dropna()

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
                stokP = stockNews['^GSPC'][stockNews['Article'] == article].reset_index()['^GSPC'][0]

                lS = pd.DataFrame(pd.DataFrame(lemmatized_sentence).set_axis(['Tokens'],
                                                                             axis=1)[
                                      'Tokens'].value_counts()).reset_index()
                stocks.append(stokP)

                # Ora applichiamo un POS tagging, vale a dire capire il ruolo sintattico delle parole all'interno di una frase
                # POS_tag = nltk.pos_tag(lemmatized_sentence)

                # BAG-OF-WORDS MODEL: Creare una matrice composta di vettori di lunghezza N (=totale parole univoche in ogni
                # articolo) popolata di un '1' qualora la parola appaia nell'articolo, e 0 altrimenti

                for word in lemmatized_sentence:
                    allWords.append(word)

                preProcSentences.append(lS)

                print('Processing Articles...', round((i / len(stockNews['Article'])) * 100, 2), '%')

        return preProcSentences, allWords






