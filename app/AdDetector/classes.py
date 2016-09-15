from __future__ import division, print_function

#Imports for TextFeaturizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import math

#Extra imports for AdDetectorModel
import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn.decomposition import PCA

class TextFeaturizer(object):
    def __init__(self,articles,ads,max_vocab_size=100):
        self.max_vocab_size = max_vocab_size
        self.article_corpus = articles
        self.ad_corpus = ads
        
        self.vocabulary = self.create_vocabulary(self.article_corpus,self.ad_corpus)
        self.document_frequency = self.get_document_frequency()
        self.feature_words = self.get_feature_words()
        
    def tokenize_text(self,text):
        """
        Creates tokens from raw text
        """
    
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        extra_stops = [str(i) for i in range(2100)]
        extra_stops +=['zero','one','two','three','four','five',
                      'six','seven','eight','nine','ten']
        extra_stops +=['samsung','google','apple','iphone','htc']
        stops = set(stopwords.words("english")+extra_stops)
        clean_tokens = [token.lower() for token in tokens if token.lower() not in stops]
    
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in clean_tokens]
    
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
        return lemmatized_tokens
    
    def create_vocabulary(self,articles,ads):
        """
        Uses the corpus to create a vocabulary of the most common words with
        size < max_vocab_size
        """
        
        #Generate the full vocabulary
        vocabulary = {}
        for article in articles:
            article_text = articles[article]
            tokens = self.tokenize_text(article_text)
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = 0
                vocabulary[token] += 1
        
        for ad in ads:
            ad_text = ads[ad]
            tokens = self.tokenize_text(ad_text)
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = 0
                vocabulary[token] += 1
        
        #Prune vocabulary 
        ctr = 0
        pruned_vocabulary = {}
        for word in sorted(vocabulary, key=vocabulary.get, reverse=True):
            if ctr < self.max_vocab_size:
                pruned_vocabulary[word] = vocabulary[word]
            ctr += 1
            
        return pruned_vocabulary
    
    def get_document_frequency(self):
        """
        Computes the document frequency of words in the vocabulary
        """
        total_words = 0
        for word in self.vocabulary: total_words += self.vocabulary[word]
        document_frequency = {word:self.vocabulary[word]/total_words for word in self.vocabulary}
        return document_frequency
    
    def get_feature_words(self):
        feature_words = [word for word in self.vocabulary]
        feature_words.sort()
        return feature_words
        
    def get_tf_idf(self,text):
        """
        Gets the tf-idf for words in both text and in vocabulary
        """
        tf_idfs = {}
        #text = text.encode("utf-8")
        #text = text.decode("utf-8").encode('ascii','ignore')
        tokens = self.tokenize_text(text)
        total_tokens = len(tokens)
        
        for token in tokens:
            if token not in tf_idfs and token in self.vocabulary:
                    token_number = tokens.count(token)
                    tf = token_number/total_tokens
                    idf = math.log(0+1/self.document_frequency[token])
                    tf_idfs[token] = tf*idf
        return tf_idfs

    def get_features(self,text):
        """
        Gets the feature dictionary and feature words for text
        """
        feature_dict = {word:0 for word in self.feature_words}
        tf_idf_dict = self.get_tf_idf(text)
        for word in tf_idf_dict:
            feature_dict[word] = tf_idf_dict[word]
        feature_vector = [feature_dict[word] for word in self.feature_words]
        return feature_dict, feature_vector

class AdDetectorModel_svm(object):
    def __init__(self,articles,ads,max_vocab_size=100):
        self.article_corpus = articles
        self.ad_corpus = ads
        
        self.featurizer = TextFeaturizer(articles,ads,max_vocab_size)
    
        self.model = None
    
    def load_model(self,model_name):
        """
        Load model with name model_name
        """
        self.model = model_name

    def evaluate_text(self,text):
        """
        Evaluates the model on a single text
        """
        if self.model == None:
            print("You have to load a model first!")
            return None
        text = text.encode('utf-8')
        text = text.decode("utf-8").encode('ascii','ignore')
        _,vec = self.featurizer.get_features(text)
        input_vector = np.zeros((1,len(vec)))
        input_vector[0] = vec
        prediction = self.model.predict_proba(input_vector)
        return prediction[0]
    
    def classify_text(self,text):
        """
        Returns 0 for articles and 1 for ads
        """
        prediction = self.evaluate_text(text)
        if prediction[0] < 0.5:
            return 1
        else:
            return 0

class AdDetectorModel_svm_pca(object):
    def __init__(self,articles,ads,max_vocab_size=500,components=10):
        self.article_corpus = articles
        self.ad_corpus = ads
        
        self.featurizer = TextFeaturizer(articles,ads,max_vocab_size)
        
        self.model = None
        
        self.components = components
        with open('./models/pca_thursday_week_2.pickle','rb') as handle:
            self.pca = pickle.load(handle)
        #self.create_pca_transformer()
        
    def load_model(self,model_name):
        """
        Load model with name model_name
        """
        self.model = model_name

    def create_pca_transformer(self):
        rows_list = []
        for ad in ads:
            ad_text = ads[ad]
            feature_dict,_ = self.featurizer.get_features(ad_text)
            rows_list.append(feature_dict)
            
        for article in articles:
            article_text = articles[article]
            feature_dict,_ = self.featurizer.get_features(article_text)
            rows_list.append(feature_dict)     
        feature_frame = pd.DataFrame(rows_list)
        #print(feature_frame.head())
        pca = PCA(n_components=self.components)
        pca.fit(feature_frame)
        self.pca = pca
        
    def transform_feature(self,feature_vector):
        np_vec = np.zeros((1,len(feature_vector)))
        np_vec[0] = feature_vector
        pca_vector = self.pca.transform(np_vec)
        return pca_vector
    
    def evaluate_text(self,text):
        """
        Evaluates the model on a single text
        """
        if self.model == None:
            print("You have to load a model first!")
            return None
        text = text.encode('utf-8')
        text = text.decode("utf-8").encode('ascii','ignore')
        _,vec = self.featurizer.get_features(text)
        print(vec)
        transformed_vec = self.transform_feature(vec)
        print(transformed_vec)
        prediction = self.model.predict_proba(transformed_vec)
        
        return prediction[0]
    
    def classify_text(self,text):
        """
        Returns 0 for articles and 1 for ads
        """
        prediction = self.evaluate_text(text)
        if prediction[0] < 0.5:
            return 1
        else:
            return 0
        
if __name__ == '__main__':        
    #Load the data
   with open('/home/sine/Dropbox/AdDetector/data/cleaned_ads.pickle','rb') as handle:
       ads = pickle.load(handle)
           
   with open('/home/sine/Dropbox/AdDetector/data/cleaned_articles.pickle','rb') as handle:
       articles = pickle.load(handle)

   with open('./models/svm_pca_thursday_week_2.pickle','rb') as handle:
       model = pickle.load(handle)
      
   admodel = AdDetectorModel_svm_pca(articles,ads)
   admodel.load_model(model)
   
   test_text = """
   LOTTOTECH has been recognized as one of the 10 best companies in the world of gaming in terms of online gaming innovation supply by the nomination panel of Global Gaming Awards 2015. The panel consists of Gambling Insider editorial members and independent media professionals who assessed the whole gaming industry and delivered a shortlist of 10 companies for each category. The shortlists will be then judged by a panel of 50 renowned senior gaming industry executives, who will choose the winners in each category.

    LOTTOTECHs innovative cross-channel lottery platform is the first product of that kind to unify both currently existing business models of digital distribution  the messenger and the insurance model. Through the most advanced technology in the whole industry, the mix of these two models gives LOTTOTECHs clients a unique competitive advantage over all other existing white labels on the market and provides them with a superior means of revenue generation and cost optimization across all business levels.

    This is the second recognition of innovation, performance and achievement LOTTOTECH has received this year.

    Such level of appreciation within just a year since LOTTOTECH has stepped on the market is a solid proof that we are overtaking the competition and establishing ourselves as the industrys preferred B2B lottery platform provider.
   """
   print(admodel.evaluate_text(test_text))
   