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

class SVM_PCA_model(object):
    def __init__(self):
        #Load the data
        with open('/home/sine/Dropbox/AdDetector/data/cleaned_ads.pickle','rb') as handle:
            self.ads = pickle.load(handle)
                
        with open('/home/sine/Dropbox/AdDetector/data/cleaned_articles.pickle','rb') as handle:
            self.articles = pickle.load(handle)
        
        with open('models/svm_pca_thursday_week_2.pickle','rb') as handle:
            self.model = pickle.load(handle)
        
        self.vocabulary = self.create_vocab()
        self.feature_words,self.corpus_frequency = self.get_corpus_info()
        self.feature_frame = self.create_feature_frame()
        self.pca = self.create_pca_transformer()
        
    def tokenize_text(self,text):
        """
        Creates tokens from raw text
        """
    
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        extra_stops = [str(i) for i in range(2100)]
        extra_stops +=['zero','one','two','three','four','five',
                      'six','seven','eight','nine','ten']
        #extra_stops +=['samsung','google','apple','iphone','htc']
        stops = set(stopwords.words("english")+extra_stops)
        clean_tokens = [token.lower() for token in tokens if token.lower() not in stops]
    
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in clean_tokens]
    
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
        return lemmatized_tokens
    
    def create_vocab(self):
        vocabulary = {}
        for ad in self.ads:
            ad_text = self.ads[ad]
            tokens = self.tokenize_text(ad_text)
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = 0
                vocabulary[token] += 1
        
        for article in self.articles:
            article_text = self.articles[article]
            tokens = self.tokenize_text(article_text)
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = 0
                vocabulary[token] += 1
                
        n = 500
        #Prune vocabulary 
        ctr = 0
        pruned_vocabulary = {}
        for word in sorted(vocabulary, key=vocabulary.get, reverse=True):
            if ctr < n:
                pruned_vocabulary[word] = vocabulary[word]
            ctr += 1
        
        return pruned_vocabulary
    
    def get_corpus_info(self):
        total_words = 0
        for word in self.vocabulary: total_words += self.vocabulary[word]
        corpus_frequency = {word:self.vocabulary[word]/total_words for word in self.vocabulary}
        feature_words = [word for word in corpus_frequency]
        feature_words.sort()
        return feature_words,corpus_frequency
    
    def get_tf_idf(self,text):
        tf_idfs = {}
        
        text = text.encode("utf-8")
        text = text.decode("utf-8").encode('ascii','ignore')
        
        tokens = self.tokenize_text(text)
        total_tokens = len(tokens)
        
        for token in tokens:
            if token not in tf_idfs and token in self.corpus_frequency:
                    token_number = tokens.count(token)
                    tf = token_number/total_tokens
                    idf = math.log(0+1/self.corpus_frequency[token])
                    tf_idfs[token] = tf*idf
        return tf_idfs
    
    def get_features(self,text):
        feature_dict = {word:0 for word in self.feature_words}
        tf_idf_dict = self.get_tf_idf(text)
        for word in tf_idf_dict:
            feature_dict[word] = tf_idf_dict[word]
        feature_vector = [feature_dict[word] for word in self.feature_words]
        return feature_dict, feature_vector
    
    def create_feature_frame(self):
        columns = self.feature_words + ['is_ad']
        
        rows_list = []
        for ad in self.ads:
            ad_text = self.ads[ad]
            feature_dict,_ = self.get_features(ad_text)
            feature_dict['is_ad'] = 1
            rows_list.append(feature_dict)
            
        for article in self.articles:
            article_text = self.articles[article]
            feature_dict,_ = self.get_features(article_text)
            feature_dict['is_ad'] = 0
            rows_list.append(feature_dict)
        
        feature_frame = pd.DataFrame(rows_list)
        cols = feature_frame.columns.tolist()
        cols.remove('is_ad')
        new_cols = ['is_ad'] + cols
        feature_frame = feature_frame[new_cols]

        return feature_frame
    
    def create_pca_transformer(self):
        pca_frame = self.feature_frame
        del pca_frame['is_ad']
        components = 10
        pca = PCA(n_components=components)
        pca.fit(pca_frame)
        return pca

    def transform_feature(self,feature_vector):
        np_vec = np.zeros((1,len(feature_vector)))
        np_vec[0] = feature_vector
        pca_vector = self.pca.transform(np_vec)
        return pca_vector

    def classifier(self,x_in):
        predictions = self.model.predict_proba(x_in)
        classes = []
        for prediction in predictions:
            classes.append(prediction[1])
        return classes

    def evaluate_text(self,text):
        _,vec = self.get_features(text)
        transformed_vec = self.transform_feature(vec)
        print(transformed_vec)
        inpt = np.zeros((1,len(transformed_vec[0])))
        inpt[0] = transformed_vec[0]
        return self.classifier(inpt)[0]
    
if __name__ == '__main__':
    model = SVM_PCA_model()
    test_text = "This is some test text to see if things are working"
    _,vec = model.get_features(test_text)
    print(model.evaluate_text(test_text))