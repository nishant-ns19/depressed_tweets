import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log
from math import exp
from math import e
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.random import RandomState
from nltk.corpus import stopwords
import csv
import sys
import nltk
import random
nltk.download('punkt')
nltk.download('stopwords')


def Message_Processing(message, lower_case = True, stem = True, stop_words = True, gram = 1):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 1]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


class Corpus:
    Vocabulary_Length = 0
    sum_tf_idf_weights_all_terms = 0
    Topic_of_Rows = 0
    Vocabulary_Doc = 0
    

    def __init__(self, csv_file, topic):
        vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',ngram_range=(1, 2))
        term_document_matrix = vectorizer.fit_transform(self.Generate_Document(csv_file, topic, textcol=1, skipheader=True))
        self.Vocabulary_Length = len(vectorizer.get_feature_names())
        col = [i for i in vectorizer.get_feature_names()]
        Vocabulary_Doc = pd.DataFrame(term_document_matrix.todense(), columns=col)
        self.Topic_of_Rows = Vocabulary_Doc.shape[0]
        self.tf_idf_per_term = Vocabulary_Doc.sum(axis=0, skipna=True)
        sum_tf_idf_weights_all_terms_temp = 0
        for i in self.tf_idf_per_term:
            sum_tf_idf_weights_all_terms_temp += i
        self.sum_tf_idf_weights_all_terms = sum_tf_idf_weights_all_terms_temp


    def get_term_tf_idf(self,term):
        try:
            return self.tf_idf_per_term.loc[term]
        except:
            return 0


    def Generate_Document(self,filepath, topic, textcol=0, skipheader=True):
        porter = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        with open(filepath) as f:
            reader = csv.reader(f)
            if skipheader:
                next(reader, None)
            if (topic == '-1'):
                for row in reader:
                   stem = self.stemSentence(porter, row[textcol])
                   yield stem
            else:
                for row in reader:
                    if (topic == row[2]):
                        stem = self.stemSentence(porter, row[textcol])
                        yield stem


    def stemSentence(self,porter, sentence):
        words_token = word_tokenize(sentence)
        sentence_stem = []
        for word in words_token:
            sentence_stem.append(porter.stem(word))
            sentence_stem.append(" ")
        return "".join(sentence_stem)


    def lemmatizeSentence(self,lemmatizer, sentence):
        words_token = word_tokenize(sentence)
        sentence_stem = []
        for word in words_token:
            sentence_stem.append(lemmatizer.lemmatize(word))
            sentence_stem.append(" ")
        return "".join(sentence_stem)


class ClassifyTweet(object):
    Corpus, Non_Depressed_Corpus, Depressed_Corpus = None, None, None
    feature_names_size, Total_Docs, Non_Depressed_Docs, Depressed_Docs, Non_Depressed_Topic_P, Depressed_Topic_P = 0, 0, 0, 0, 0, 0


    def __init__(self, csv_file):
        print('Converting a collection of ENTIRE Corpus to a matrix of TF-IDF features')
        self.Corpus = Corpus(csv_file, '-1')
        self.feature_names_size = self.Corpus.Vocabulary_Length
        self.Total_Docs = self.Corpus.Topic_of_Rows

        print('Converting a NON-DEPRESSED Corpus to a matrix of TF-IDF features')
        self.Non_Depressed_Corpus = Corpus(csv_file, '0')
        self.Non_Depressed_Docs = self.Non_Depressed_Corpus.Topic_of_Rows
        self.non_depressed_sum_tf_idf_weights_all_terms = self.Non_Depressed_Corpus.sum_tf_idf_weights_all_terms

        print('Converting a DEPRESSED Corpus to a matrix of TF-IDF features')
        self.Depressed_Corpus = Corpus(csv_file, '1')
        self.Depressed_Docs = self.Depressed_Corpus.Topic_of_Rows
        self.depressed_sum_tf_idf_weights_all_terms = self.Depressed_Corpus.sum_tf_idf_weights_all_terms
        self.Non_Depressed_Topic_P = log(self.Non_Depressed_Docs / self.Total_Docs)
        self.Depressed_Topic_P = log(self.Depressed_Docs / self.Total_Docs)
    

    def Classify_with_Naive_Bayes(self,tweet):
        probability_non_depressed = 0
        for term in tweet:
            tf_idf_per_term = self.Non_Depressed_Corpus.get_term_tf_idf(term)
            probability_non_depressed += log((tf_idf_per_term + 1) / (self.non_depressed_sum_tf_idf_weights_all_terms + self.feature_names_size))
        probability_non_depressed += self.Non_Depressed_Topic_P
        probability_depressed = 0
        for term in tweet:
            #Laplace smoothing
            tf_idf_per_term = self.Depressed_Corpus.get_term_tf_idf(term)
            probability_depressed += log((tf_idf_per_term + 1) / (self.depressed_sum_tf_idf_weights_all_terms + self.feature_names_size))
        probability_depressed += self.Depressed_Topic_P
        return probability_depressed, probability_non_depressed
    

    def predict(self,testData):
        result = []
        for i, r in testData.iterrows():
            processed_message = Message_Processing(r['message'])
            probability_depressed, probability_non_depressed=self.Classify_with_Naive_Bayes(processed_message)
            result.append(int(probability_depressed>=probability_non_depressed))
        pd.options.mode.chained_assignment = None
        testData['prediction'] = result
        return testData
    

    def Get_Metrics(self,testData):
        print('Calculating precision, re-call, accurancy, F-score on test-data prediction.')
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        for index, row in testData.iterrows():
            label = int(row['label'])
            prediction = row['prediction']
            if(label == 0 and prediction == 0):
            	true_pos += 1
            elif(label == 1 and prediction == 1):
            	true_neg += 1
            elif(label == 1 and prediction == 0):
            	false_pos += 1
            else:
            	false_neg += 1

        precision = true_pos
        precision /= (true_pos + false_pos)
        recall = true_pos 
        recall /= (true_pos + false_neg)
        Fscore = 2 * precision * recall 
        Fscore /= (precision + recall)
        accuracy = (true_pos + true_neg) 
        accuracy /= (true_pos + true_neg + false_pos + false_neg)

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", Fscore)
        print("Accuracy: ", accuracy)


def Retrieve_Songs(flag):
    df = pd.read_csv("songs.csv",encoding='ISO-8859-1')
    track = df['Track.Name']
    artist = df['Artist.Name']
    genre = df['Genre']
    energy = df['Energy']
    dSongs = []
    nSongs = []
    for i in range(50):
        if(energy[i] >= 67):
            dSongs.append([track[i],artist[i],genre[i],energy[i]])
        else:
            nSongs.append([track[i],artist[i],genre[i],energy[i]])
    if(flag == 1):
        return random.sample(dSongs,5)
    else:
        return random.sample(nSongs,5)


if __name__ == '__main__':
    csv_file_name = 'sentiment_tweets3'
    try:
        print('----------------------TRAINING MODEL------------------------------')
        print('Reading Kaggle dataset twitter_sentiment data in file, ' + csv_file_name)
        df = pd.read_csv(csv_file_name + '.csv')
        rng = RandomState()

        print('Split data into 90% training data and 10% test data..')
        trainData = df.sample(frac=0.9, random_state=rng)
        testData = df.loc[~df.index.isin(trainData.index)]
        trainData.to_csv(csv_file_name + '_train.csv', index=False)
        testData.to_csv(csv_file_name + '_test.csv', index=False)

        print('Train the classifier using training data set...')
        ClassifyTweet = ClassifyTweet(csv_file_name + '_train.csv')

        print('----------------------Applying classifier on test data------------------------------')
        results = ClassifyTweet.predict(testData)
        ClassifyTweet.Get_Metrics(results)

        print('----------------------------------------------------------')
        print('\n\n\n*********PREDICTIONS********\n')
        while True:
            tweet = input("Enter a tweet? (quit to exit):\n")
            if tweet == "quit":
                break
            else:
                processed_message = Message_Processing(tweet)
                probability_depressed, probability_non_depressed=ClassifyTweet.Classify_with_Naive_Bayes(processed_message)
                status = int(probability_depressed>=probability_non_depressed)
                print('----------------------------------------------------------')
                if status == 0:
                    print('Depression Status: Person doesn\'t seem to be suffering from depression..')
                else:
                    print('Depression Status: Person does seem to be suffering from depression..')
                print('----------------------------------------------------------')
                print("Suggestions are:")
                suggestions = Retrieve_Songs(status)
                print('----------------------------------------------------------')
                for song in suggestions:
                    print(song[0] + " by " + song[2])
                    print('----------------------------------------------------------')
                print('\n')
    except FileNotFoundError:
        print('Reading Kaggle dataset twitter_sentiment data in file, ' + csv_file_name + ',not found!')
        sys.exit(1)