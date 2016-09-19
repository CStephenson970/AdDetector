from __future__ import division, print_function
import numpy as np
import cPickle as pickle

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.data import load


from keras.models import load_model
from keras.preprocessing import sequence

class lstm_model(object):
    def __init__(self):
        
        self.max_text_length = 500
        
        with open('./AdDetector/models/word_ids.pickle','rb') as handle: 
            self.word_ids = pickle.load(handle)
            
        self.model = load_model('./AdDetector/models/lstm_model_500.h5')
        self.sentence_tokenizer = load('tokenizers/punkt/english.pickle')
        
    def tokenize_text(self,text):
        """
        Creates tokens from raw text
        """
        tokenizer = RegexpTokenizer(r'\w+')
        
        tokens = tokenizer.tokenize(text)
        extra_stops = [str(i) for i in range(2100)]
        extra_stops +=['zero','one','two','three','four','five',
                      'six','seven','eight','nine','ten']
        stops = set(stopwords.words("english")+extra_stops)
        clean_tokens = [token.lower() for token in tokens if token.lower() not in stops]
    
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in clean_tokens]
    
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
        return lemmatized_tokens
    
    def get_id_list(self,text):
        """
        turns text into a list of word ids
        """
        tokens = self.tokenize_text(text)
        id_list = []
        for token in tokens:
            if token in self.word_ids:
                id_list.append(self.word_ids[token])
        return id_list
    
    def evaluate_text(self,text):
        """
        evaluates the model with text as the input
        """
        text = text.encode('utf-8')
        text = text.decode('utf-8').encode('ascii','ignore')
        id_list = self.get_id_list(text)
        inpt = np.zeros(1, dtype=object)
        inpt[0] = id_list
        inpt = sequence.pad_sequences(inpt, maxlen=self.max_text_length)
        ad_prob = self.model.predict(inpt)[0,0]
        return [1-ad_prob,ad_prob]
    
    def score_sentences(self,text):
        sentences = self.sentence_tokenizer.tokenize(text)
        sentence_scores = {}
        for sentence in sentences:
            evalu = self.evaluate_text(sentence)
            sentence_scores[sentence] = evalu[1]
        return sentence_scores
    
    def get_best_worst(self,text):
        sentence_scores = self.score_sentences(text)
        sentences = []
        for sentence in sorted(sentence_scores, key=sentence_scores.get, reverse=True):
            sentences.append(sentence)
        worst_sentence = sentences[0]
        best_sentence = sentences[-1]
        return best_sentence, worst_sentence
    
if __name__ == '__main__':
    model = lstm_model()
    text = "this is some sample text to see if things are working properly"
    print(model.evaluate_text(text))