from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from unidecode import unidecode
import numpy as np

class TextPreparation(object):
    #0.001, 0.95
    3000 
    60
    def __init__(self,text_series,cv_params={"min_df":0.005,"max_df":0.15},language="english"):
        self.text_series = text_series
        self.cv_params = cv_params
        self.language=language
        language_model_name_dict = {"english":"en_core_web_sm","spanish":"es_core_news_sm","german":"de_core_news_sm"}
        self.model = spacy.load(language_model_name_dict[self.language], disable = ['parser'])

    def _get_stopwords(self):
        nltk.download("stopwords")
        english_stopwords = stopwords.words(self.language)
        self.stopwords = english_stopwords

    
    def _clean_text(self,text):
        self._get_stopwords()
        lower_text = text.str.lower()
        no_accents = lower_text#.apply(unidecode)
        #alpha_text =  no_accents.str.replace(r"^[\.a-zA-Z0-9,!? ]*$", "")
        #nostops = alpha_text.apply(lambda x: " ".join([word for word in x.split() if word not in self.stopwords and (len(word) > 2 )]))#or word == "." or word == " .")]))
        #alpha_text =  no_accents.str.replace(r"(@\[a-z]+)|([^a-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "")
        nostops = no_accents.apply(lambda x: " ".join([word for word in x.split() if word not in self.stopwords and len(word) > 2]))
        return nostops
    
    def _lemmatize_words(self,text):
        def _lemmatize_doc(doc):
            doc_obj = self.model(doc)
            lemmatized = " ".join([token.lemma_ for token in doc_obj])
            return lemmatized
        return text.apply(_lemmatize_doc)
    
    def _filter_words(self,text,max_features=False):
        cv = CountVectorizer(min_df=self.cv_params["min_df"],max_df=self.cv_params["max_df"])
        cv.fit(text)
        print(cv.vocabulary_)
        filtered_text = text.apply(lambda x: " ".join([word for word in x.split() if word in cv.vocabulary_]))
        print(filtered_text)
        filled_text = filtered_text.apply(lambda x: text.values[0] if x == "" else x)
        self.vocab_size=len(cv.vocabulary_)
        return filled_text
    
    
    def _indexes_to_keep(self,text):
        indexes_keep = np.where(text.str.len() > 0)
        self.indexes_to_keep = indexes_keep
        return text.loc[indexes_keep]
    
    
    def prepare_text(self,pipeline=["clean","filter","keep"]):
        functions = {"clean":self._clean_text,"filter":self._filter_words,"lemmatize":self._lemmatize_words,"keep":self._indexes_to_keep}
        text = self.text_series
        for step in pipeline:
            text = functions[step](text)
        
        return text
    
    def get_entities(self,texts):
        entities = set()
        for text in texts:
            data = self.model(text)
            for ent in data.ents:
                entity_text = ent.text
                for entity in entity_text.split(" "):
                    entities.add(entity)
        return entities

